"""Async job execution.

Analyses run in an in-process thread pool: the engine's heavy work is
numpy/sklearn, which releases the GIL, so threads give real concurrency
without requiring Redis. `submit()` is the seam — a Celery or Arq backend
implements the same one-line contract (enqueue a job id) and the API layer
never changes.

The worker owns the job state machine: queued -> running -> succeeded|failed.
A crash is recorded on the job row, never swallowed, and jobs orphaned by a
restart are failed loudly at startup instead of spinning forever.

No database session is held open while the engine runs, so progress updates
and API reads never wait on a long analysis.
"""

from __future__ import annotations

import atexit
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from datagate import blind, signing
from datagate.orchestrator import analyze
from datagate.service.auth import registry_for
from datagate.service.config import settings
from datagate.service.db import (
    LOCAL_USER_ID,
    DatasetRecord,
    DealRecord,
    DealState,
    JobRecord,
    JobStatus,
    ReportRecord,
    SpecRecord,
    session_scope,
)
from datagate.service.storage import get_storage
from datagate.types import ProjectSpec

log = logging.getLogger("datagate.worker")

_executor = ThreadPoolExecutor(
    max_workers=settings.worker_threads, thread_name_prefix="datagate-worker")
atexit.register(lambda: _executor.shutdown(wait=False))
_progress_lock = threading.Lock()


def submit(job_id: str) -> None:
    """Enqueue a job. Swap this body for Celery/Arq dispatch in production."""
    _executor.submit(_run_job, job_id)


def run_job_sync(job_id: str) -> None:
    """Execute a job on the calling thread — used by tests."""
    _run_job(job_id)


def recover_orphaned_jobs() -> int:
    """Fail jobs left queued/running by a previous process. Returns how many."""
    with session_scope() as session:
        orphans = session.query(JobRecord).filter(
            JobRecord.status.in_([JobStatus.QUEUED.value, JobStatus.RUNNING.value])).all()
        for job in orphans:
            job.status = JobStatus.FAILED.value
            job.error = "interrupted by a service restart — run the analysis again"
            job.finished_at = datetime.now(timezone.utc)
            if job.deal_id:
                deal = session.get(DealRecord, job.deal_id)
                if deal is not None and deal.state == DealState.EVALUATING.value:
                    deal.state = DealState.LISTED.value
        session.commit()
        return len(orphans)


def _progress_sink(job_id: str):
    finished: list[str] = []   # names, in completion order, so the UI can say which ran

    def update(analyzer: str, done: int, total: int) -> None:
        with _progress_lock, session_scope() as session:
            finished.append(analyzer)
            job = session.get(JobRecord, job_id)
            if job is not None:
                job.progress = {"done": done, "total": total, "last": analyzer,
                                "finished": list(finished)}
                session.commit()
    return update


def _run_job(job_id: str) -> None:
    # 1) Claim the job and read everything needed, then release the session.
    with session_scope() as session:
        job = session.get(JobRecord, job_id)
        if job is None:
            log.error("job %s vanished before execution", job_id)
            return
        job.status = JobStatus.RUNNING.value
        job.started_at = datetime.now(timezone.utc)
        session.commit()

        dataset = session.get(DatasetRecord, job.dataset_id)
        spec_row = session.get(SpecRecord, job.spec_id)
        reference = (session.get(DatasetRecord, job.reference_dataset_id)
                     if job.reference_dataset_id else None)
        deal = session.get(DealRecord, job.deal_id) if job.deal_id else None
        mode, deal_id = job.mode, job.deal_id
        owner_id = job.owner_id or LOCAL_USER_ID
        deal_title = deal.title if deal else ""
        dataset_ref, dataset_name = dataset.storage_ref, dataset.filename
        reference_ref = reference.storage_ref if reference else None
        reference_name = reference.filename if reference else None
        spec = ProjectSpec.model_validate(spec_row.spec)

    error: str | None = None
    report = quarantine = None
    try:
        storage = get_storage()
        result = analyze(
            str(storage.local_path(dataset_ref)),
            spec=spec,
            reference_path=(str(storage.local_path(reference_ref))
                            if reference_ref else None),
            dataset_name=dataset_name,
            reference_name=reference_name,
            registry=registry_for(owner_id),
            progress=_progress_sink(job_id),
        )
        report, quarantine = result.report, result.quarantine
        if mode == "B":
            # Only the score leaves: row-level findings never reach the buyer.
            report, quarantine = blind.score_only(report, title=deal_title), None
        signing.sign_report(report)
    except Exception as exc:
        log.exception("analysis job %s failed", job_id)
        error = f"{type(exc).__name__}: {exc}"

    # 2) Record the outcome in a fresh session.
    with session_scope() as session:
        job = session.get(JobRecord, job_id)
        if job is None:
            return
        deal = session.get(DealRecord, deal_id) if deal_id else None
        if error is None and report is not None:
            quarantine_ref = None
            if quarantine is not None:
                blob = json.dumps([f.model_dump(mode="json") for f in quarantine])
                quarantine_ref = get_storage().put_bytes(blob.encode("utf8"),
                                                         "quarantine.json")
            record = ReportRecord(
                job_id=job.id,
                verdict=report.decision.verdict.value,
                safety_score=report.safety.score,
                fit_score=report.fit.score,
                payload=report.model_dump(mode="json"),
                quarantine_ref=quarantine_ref,
                blind=1 if mode == "B" else 0,
                engine_version=report.engine_version,
                owner_id=job.owner_id,
            )
            session.add(record)
            session.flush()
            job.report_id = record.id
            job.status = JobStatus.SUCCEEDED.value
            if deal is not None:
                deal.state = DealState.EVALUATED.value
                deal.report_id = record.id
                deal.history = [*deal.history, {
                    "event": "evaluated", "verdict": record.verdict,
                    "at": datetime.now(timezone.utc).isoformat()}]
        else:
            job.status = JobStatus.FAILED.value
            job.error = error or "analysis produced no report"
            if deal is not None and deal.state == DealState.EVALUATING.value:
                deal.state = DealState.LISTED.value
        job.finished_at = datetime.now(timezone.utc)
        session.commit()
