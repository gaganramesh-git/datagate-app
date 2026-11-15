"""FastAPI application — the DataGate service surface.

Route shapes follow the API sketch in the design doc. The service is a thin
shell around the engine: it stores inputs, schedules work, and serves results.
It never computes a score itself.

Mode A (open checker): /datasets, /specs, /analyses, /reports.
Mode B preview (blind evaluation): /deals — a buyer vets a seller's dataset and
receives a signed, score-only report. No payments, identity or escrow yet.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import polars as pl
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy import or_

from datagate import export, signing
from datagate.analyzers.base import registered_analyzers
from datagate.ingest.loader import SUPPORTED_SUFFIXES, load_dataset
from datagate.ingest.profiler import profile_dataset
from datagate.orchestrator import default_analyzers
from datagate.report import render_markdown
from datagate.service import worker
from datagate.service.auth import User, current_user, registry_for
from datagate.service.auth import router as auth_router
from datagate.service.config import settings
from datagate.service.db import (
    DatasetRecord,
    DealRecord,
    DealState,
    JobRecord,
    JobStatus,
    ReportRecord,
    SpecRecord,
    init_db,
    session_scope,
)
from datagate.service.schemas import (
    AnalysisRequest,
    AnalysisStatus,
    DatasetSummary,
    DealCreate,
    DealDecision,
    DealOut,
    EvaluationRequest,
    Meta,
    ReportResponse,
    RowsResponse,
    SpecRequest,
    SpecResponse,
    SpecUpdate,
    VerifyResponse,
)
from datagate.service.storage import UploadTooLarge, get_storage
from datagate.spec import llm as llm_backend
from datagate.spec.compiler import compile_spec
from datagate.types import ENGINE_VERSION, Flag, ProjectSpec, Report, Severity

STATIC_DIR = Path(__file__).parent / "static"
MAX_PREVIEW_ROWS = 100
MAX_CELL_CHARS = 600

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}
# The app ships its own scripts; nothing is loaded from other origins.
_CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
        "base-uri 'none'; form-action 'self'")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    worker.recover_orphaned_jobs()
    yield


app = FastAPI(
    title="DataGate",
    version=ENGINE_VERSION,
    description="Is this dataset safe, and will it help my model?",
    lifespan=lifespan,
)
app.include_router(auth_router)


_UNSAFE = {"POST", "PUT", "PATCH", "DELETE"}


# Belt and braces on top of SameSite=Lax cookies: a state-changing request that
# names a different origin than the host it was sent to is refused.
@app.middleware("http")
async def origin_guard(request, call_next):
    origin = request.headers.get("origin")
    if (request.method in _UNSAFE and origin
            and urlparse(origin).netloc != request.headers.get("host", "")):
        return PlainTextResponse("Cross-origin request refused.", status_code=403)
    return await call_next(request)


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    for name, value in _SECURITY_HEADERS.items():
        response.headers.setdefault(name, value)
    # The interactive API docs load their assets from a CDN; leave them alone.
    if not request.url.path.startswith(("/docs", "/redoc", "/openapi")):
        response.headers.setdefault("Content-Security-Policy", _CSP)
    return response


# For routes that need a signed-in user but not who they are.
AUTH = [Depends(current_user)]


def _not_found(what: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"{what} not found")


def _owned(session, model, row_id: Optional[str], user: User, what: str):
    """A row the user owns. Someone else's row is reported as not found, so ids
    cannot be probed for existence."""
    row = session.get(model, row_id) if row_id else None
    if row is None or row.owner_id != user.id:
        raise _not_found(what)
    return row


# -- meta ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}


@app.get("/meta", response_model=Meta, dependencies=AUTH)
def meta() -> Meta:
    registered_analyzers()  # ensure plugins are imported
    return Meta(
        version=app.version, engine_version=ENGINE_VERSION,
        auth_enabled=settings.auth_enabled, llm_enabled=llm_backend.llm_enabled(),
        signing_public_key=signing.public_key_b64(),
        analyzers=[{"name": a.name, "family": a.family, "dimension": a.dimension,
                    "needs_reference": a.needs_reference} for a in default_analyzers()],
    )


# -- datasets -----------------------------------------------------------------------


@app.post("/datasets", response_model=DatasetSummary)
def create_dataset(file: UploadFile = File(...),
                   user: User = Depends(current_user)) -> DatasetSummary:
    """Upload a dataset. Returns the statistical profile immediately."""
    filename = Path(file.filename or "upload").name
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported format {suffix!r}; supported: {sorted(SUPPORTED_SUFFIXES)}")

    storage = get_storage()
    try:
        ref, size, digest = storage.put_stream(file.file, filename,
                                               settings.max_upload_bytes)
    except UploadTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc))
    if size == 0:
        storage.delete(ref)
        raise HTTPException(status_code=400, detail="uploaded file is empty")

    with session_scope() as session:
        shared = session.query(DatasetRecord).filter_by(storage_ref=ref).count()
    try:
        ds = load_dataset(storage.local_path(ref), name=filename)
        profile = profile_dataset(ds)
    except Exception as exc:
        if not shared:
            storage.delete(ref)
        raise HTTPException(status_code=400, detail=f"could not parse dataset: {exc}")

    # Provenance: have we seen (most of) this data before?
    registry = registry_for(user.id)
    seen = [{"name": m.name, "kind": m.kind, "containment": round(m.containment, 3),
             "exact": m.exact} for m in registry.match(ds, min_containment=0.2)[:5]]
    registry.add(ds, name=filename, kind="upload")

    with session_scope() as session:
        record = DatasetRecord(
            filename=filename, storage_ref=ref, format=ds.format, size_bytes=size,
            n_rows=ds.n_rows, n_cols=ds.n_cols, fingerprint=digest, profile=profile,
            owner_id=user.id)
        session.add(record)
        session.commit()
        return _dataset_summary(record, seen_before=seen)


@app.get("/datasets", response_model=list[DatasetSummary])
def list_datasets(limit: int = Query(50, ge=1, le=200),
                  user: User = Depends(current_user)) -> list[DatasetSummary]:
    with session_scope() as session:
        records = (session.query(DatasetRecord).filter_by(owner_id=user.id)
                   .order_by(DatasetRecord.created_at.desc()).limit(limit).all())
        return [_dataset_summary(r) for r in records]


@app.get("/datasets/{dataset_id}", response_model=DatasetSummary)
def get_dataset(dataset_id: str, user: User = Depends(current_user)) -> DatasetSummary:
    with session_scope() as session:
        return _dataset_summary(_owned(session, DatasetRecord, dataset_id, user, "dataset"))


@app.delete("/datasets/{dataset_id}")
def purge_dataset(dataset_id: str, user: User = Depends(current_user)) -> dict[str, Any]:
    """Delete the stored data. Past reports stay readable; exports stop working."""
    with session_scope() as session:
        record = _owned(session, DatasetRecord, dataset_id, user, "dataset")
        ref, fingerprint = record.storage_ref, record.fingerprint
        record.storage_ref = ""
        session.flush()
        still_used = session.query(DatasetRecord).filter_by(storage_ref=ref).count()
        session.commit()
    if ref and not still_used:
        get_storage().delete(ref)
        registry_for(user.id).remove(fingerprint)
    return {"purged": True, "blob_deleted": bool(ref and not still_used)}


# -- specs --------------------------------------------------------------------------


def _load_for_spec(session, dataset_id: Optional[str], user: Optional[User]):
    """Load a dataset to parse a spec against. `user` is None only for a blind
    evaluation, where the buyer reads the seller's listed data (shape only)."""
    if not dataset_id:
        return None
    if user is not None:
        record = _owned(session, DatasetRecord, dataset_id, user, "dataset")
    else:
        record = session.get(DatasetRecord, dataset_id)
    if record is None:
        raise _not_found("dataset")
    if not record.storage_ref:
        raise HTTPException(status_code=410, detail="dataset data has been purged")
    dataset = load_dataset(get_storage().local_path(record.storage_ref),
                           name=record.filename)
    dataset.profile = record.profile
    return dataset


@app.post("/specs", response_model=SpecResponse)
def create_spec(request: SpecRequest, user: User = Depends(current_user)) -> SpecResponse:
    """Compile the objective + brief into a ProjectSpec.

    May come back with clarifying questions — the client is expected to show the
    parsed spec for review and PATCH corrections before starting an analysis.
    """
    with session_scope() as session:
        dataset = _load_for_spec(session, request.dataset_id, user)
        spec = compile_spec(request.description, dataset,
                            target_column=request.target_column,
                            project_brief=request.project_brief)
        row = SpecRecord(dataset_id=request.dataset_id, spec=spec.model_dump(mode="json"),
                         owner_id=user.id)
        session.add(row)
        session.commit()
        return SpecResponse(id=row.id, spec=spec, needs_clarification=spec.is_ambiguous)


@app.patch("/specs/{spec_id}", response_model=SpecResponse)
def update_spec(spec_id: str, update: SpecUpdate,
                user: User = Depends(current_user)) -> SpecResponse:
    """Accept user corrections to a parsed spec (the spec-review screen)."""
    with session_scope() as session:
        row = _owned(session, SpecRecord, spec_id, user, "spec")
        # A user-edited spec is authoritative: clear the machine's questions.
        spec = update.spec.model_copy(update={"clarifying_questions": []})
        row.spec = spec.model_dump(mode="json")
        session.commit()
        return SpecResponse(id=row.id, spec=spec, needs_clarification=False)


@app.get("/specs/{spec_id}", response_model=SpecResponse)
def get_spec(spec_id: str, user: User = Depends(current_user)) -> SpecResponse:
    with session_scope() as session:
        row = _owned(session, SpecRecord, spec_id, user, "spec")
        spec = ProjectSpec.model_validate(row.spec)
        return SpecResponse(id=row.id, spec=spec, needs_clarification=spec.is_ambiguous)


# -- analyses -----------------------------------------------------------------------


def _require_dataset(session, dataset_id: str, what: str = "dataset",
                     user: Optional[User] = None) -> DatasetRecord:
    """A usable dataset. With `user`, it must be theirs; without, it is a seller's
    listed data being read for a blind evaluation."""
    if user is not None:
        record = _owned(session, DatasetRecord, dataset_id, user, what)
    else:
        record = session.get(DatasetRecord, dataset_id)
    if record is None:
        raise _not_found(what)
    if not record.storage_ref:
        raise HTTPException(status_code=410, detail=f"{what} data has been purged")
    return record


@app.post("/analyses", response_model=AnalysisStatus, status_code=202)
def create_analysis(request: AnalysisRequest,
                    user: User = Depends(current_user)) -> AnalysisStatus:
    """Start an async analysis job."""
    if request.mode != "A":
        raise HTTPException(
            status_code=400,
            detail="only Mode A runs here; blind (Mode B) evaluations start from /deals")
    with session_scope() as session:
        _require_dataset(session, request.dataset_id, user=user)
        _owned(session, SpecRecord, request.spec_id, user, "spec")
        if request.reference_dataset_id:
            _require_dataset(session, request.reference_dataset_id, "reference dataset",
                             user=user)
        job = JobRecord(dataset_id=request.dataset_id, spec_id=request.spec_id,
                        reference_dataset_id=request.reference_dataset_id,
                        mode="A", status=JobStatus.QUEUED.value, owner_id=user.id)
        session.add(job)
        session.commit()
        status = _job_status(session, job)
    worker.submit(status.id)
    return status


@app.get("/analyses/{job_id}", response_model=AnalysisStatus)
def get_analysis(job_id: str, user: User = Depends(current_user)) -> AnalysisStatus:
    with session_scope() as session:
        job = _owned(session, JobRecord, job_id, user, "analysis")
        return _job_status(session, job)


@app.get("/analyses", response_model=list[AnalysisStatus])
def list_analyses(limit: int = Query(25, ge=1, le=100),
                  user: User = Depends(current_user)) -> list[AnalysisStatus]:
    with session_scope() as session:
        jobs = (session.query(JobRecord)
                .filter(JobRecord.mode == "A", JobRecord.owner_id == user.id)
                .order_by(JobRecord.created_at.desc()).limit(limit).all())
        return [_job_status(session, j) for j in jobs]


# -- reports ------------------------------------------------------------------------


def _report_context(session, report_id: str, user: User, owner_only: bool = False
                    ) -> tuple[ReportRecord, Optional[JobRecord], Optional[DatasetRecord]]:
    """A report the user may read: their own, or — for a blind, score-only report —
    the one on a listing they sold. Anything else is reported as not found."""
    record = session.get(ReportRecord, report_id)
    allowed = record is not None and record.owner_id == user.id
    if record is not None and not allowed and record.blind and not owner_only:
        deal = session.query(DealRecord).filter_by(report_id=record.id).first()
        allowed = deal is not None and deal.seller_id == user.id
    if record is None or not allowed:
        raise _not_found("report")
    job = session.get(JobRecord, record.job_id)
    dataset = session.get(DatasetRecord, job.dataset_id) if job else None
    return record, job, dataset


def _export_inputs(report_id: str, user: User):
    """Load the dataset and full quarantine list behind a report, or refuse."""
    with session_scope() as session:
        record, _, dataset = _report_context(session, report_id, user, owner_only=True)
        if record.blind:
            raise HTTPException(status_code=403,
                                detail="blind evaluations expose scores only")
        if dataset is None or not dataset.storage_ref:
            raise HTTPException(status_code=410, detail="dataset data has been purged")
        ref, name, quarantine_ref = dataset.storage_ref, dataset.filename, record.quarantine_ref
        embedded = record.payload.get("quarantine_list", [])
    storage = get_storage()
    ds = load_dataset(storage.local_path(ref), name=name)
    if quarantine_ref:
        raw = json.loads(storage.local_path(quarantine_ref).read_text())
    else:
        raw = embedded
    return ds, [Flag.model_validate(f) for f in raw]


def _frame_response(df: pl.DataFrame, filename: str, fmt: str) -> Response:
    if fmt == "jsonl":
        body, media = df.write_ndjson(), "application/x-ndjson"
    else:
        body, media, fmt = df.write_csv(), "text/csv", "csv"
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in filename)
    return Response(content=body, media_type=media, headers={
        "Content-Disposition": f'attachment; filename="{safe}.{fmt}"'})


@app.get("/reports/{report_id}", response_model=ReportResponse)
def get_report(report_id: str, user: User = Depends(current_user)) -> ReportResponse:
    with session_scope() as session:
        record, _, dataset = _report_context(session, report_id, user)
        blind = bool(record.blind)
        deal = (session.query(DealRecord).filter_by(report_id=record.id).first()
                if blind else None)
        return ReportResponse(
            deal_id=deal.id if deal else None, deal_state=deal.state if deal else None,
            id=record.id, job_id=record.job_id, engine_version=record.engine_version,
            created_at=record.created_at,
            dataset_filename=(record.payload.get("dataset_name") if blind
                              else (dataset.filename if dataset else "(unknown)")),
            blind=blind,
            exports_available=bool(not blind and dataset and dataset.storage_ref),
            report=record.payload)


@app.get("/reports/{report_id}/markdown", response_class=PlainTextResponse)
def get_report_markdown(report_id: str, user: User = Depends(current_user)) -> str:
    with session_scope() as session:
        record, _, _ = _report_context(session, report_id, user)
        return render_markdown(Report.model_validate(record.payload))


@app.get("/reports/{report_id}/clean")
def download_clean(report_id: str, format: str = Query("csv", pattern="^(csv|jsonl)$"),
                   redact_pii: bool = False,
                   user: User = Depends(current_user)) -> Response:
    """The dataset with quarantined rows removed (or PII masked in place)."""
    ds, quarantine = _export_inputs(report_id, user)
    cleaned = export.clean_frame(ds, quarantine, min_severity=Severity.WARNING,
                                 redact_pii=redact_pii)
    return _frame_response(cleaned, f"{Path(ds.name).stem}.clean", format)


@app.get("/reports/{report_id}/quarantine")
def download_quarantine(report_id: str,
                        format: str = Query("csv", pattern="^(csv|jsonl)$"),
                        user: User = Depends(current_user)) -> Response:
    """The quarantined rows with the reason each was pulled."""
    ds, quarantine = _export_inputs(report_id, user)
    return _frame_response(export.quarantine_frame(ds, quarantine),
                           f"{Path(ds.name).stem}.quarantine", format)


@app.get("/reports/{report_id}/rows", response_model=RowsResponse)
def preview_rows(report_id: str, indices: str = Query(..., max_length=2000),
                 user: User = Depends(current_user)) -> RowsResponse:
    """Row contents for the quarantine explorer (never available for blind reports)."""
    try:
        wanted = sorted({int(i) for i in indices.split(",") if i.strip()})[:MAX_PREVIEW_ROWS]
    except ValueError:
        raise HTTPException(status_code=400, detail="indices must be comma-separated integers")
    ds, _ = _export_inputs(report_id, user)
    wanted = [i for i in wanted if 0 <= i < ds.n_rows]
    frame = ds.df.with_row_index("row_index")[wanted] if wanted else ds.df.head(0)
    rows = []
    for row in frame.iter_rows(named=True):
        rows.append({k: (v[:MAX_CELL_CHARS] + "…" if isinstance(v, str)
                         and len(v) > MAX_CELL_CHARS else v) for k, v in row.items()})
    return RowsResponse(columns=frame.columns, rows=rows)


@app.post("/reports/verify", response_model=VerifyResponse, dependencies=AUTH)
def verify_report(payload: dict[str, Any]) -> VerifyResponse:
    """Check a report's digest and signature against this installation's key."""
    result = signing.verify_report(payload.get("report", payload),
                                   trusted_public_key=signing.public_key_b64())
    return VerifyResponse(ok=result.ok, digest_ok=result.digest_ok,
                          signature_ok=result.signature_ok, trusted=result.trusted,
                          detail=result.detail)


# -- deals: blind evaluation (Mode B preview) ---------------------------------------


def _event(name: str, **extra: Any) -> dict[str, Any]:
    return {"event": name, "at": datetime.now(timezone.utc).isoformat(), **extra}


# While a deal is in one of these states anyone signed in can see it in the
# marketplace; in the others only its seller and evaluating buyer can.
_PUBLIC_DEAL_STATES = (DealState.LISTED.value, DealState.EVALUATED.value,
                       DealState.DECLINED.value)


def _is_party(deal: DealRecord, user: User) -> bool:
    return user.id in (deal.seller_id, deal.evaluator_id)


def _visible_deal(session, deal_id: str, user: User) -> DealRecord:
    deal = session.get(DealRecord, deal_id)
    if deal is None or not (_is_party(deal, user) or deal.state in _PUBLIC_DEAL_STATES):
        raise _not_found("deal")
    return deal


def _deal_out(session, deal: DealRecord, user: User) -> DealOut:
    """The deal as `user` may see it. A stranger sees the listing; the evaluation
    outcome, its report and the deal's history belong to the two parties."""
    party = _is_party(deal, user)
    dataset = session.get(DatasetRecord, deal.dataset_id)
    report = (session.get(ReportRecord, deal.report_id)
              if party and deal.report_id else None)
    columns = {name: col.get("kind", "other")
               for name, col in ((dataset.profile or {}).get("columns", {}).items()
                                 if dataset else [])}
    return DealOut(
        id=deal.id, title=deal.title, description=deal.description,
        asking_price=deal.asking_price,
        state=deal.state if party else DealState.LISTED.value,
        n_rows=dataset.n_rows if dataset else 0, n_cols=dataset.n_cols if dataset else 0,
        format=dataset.format if dataset else "", columns=columns,
        fingerprint_prefix=(dataset.fingerprint[:12] if dataset else ""),
        report_id=deal.report_id if party else None,
        verdict=report.verdict if report else None,
        safety_score=report.safety_score if report else None,
        fit_score=report.fit_score if report else None,
        history=(deal.history or []) if party else [],
        is_seller=deal.seller_id == user.id, is_evaluator=deal.evaluator_id == user.id,
        created_at=deal.created_at)


@app.post("/deals", response_model=DealOut, status_code=201)
def create_deal(request: DealCreate, user: User = Depends(current_user)) -> DealOut:
    """Seller: list a dataset for blind evaluation."""
    with session_scope() as session:
        _require_dataset(session, request.dataset_id, user=user)
        deal = DealRecord(dataset_id=request.dataset_id, title=request.title,
                          description=request.description,
                          asking_price=request.asking_price, history=[_event("listed")],
                          seller_id=user.id)
        session.add(deal)
        session.commit()
        return _deal_out(session, deal, user)


@app.get("/deals", response_model=list[DealOut])
def list_deals(limit: int = Query(50, ge=1, le=200),
               user: User = Depends(current_user)) -> list[DealOut]:
    with session_scope() as session:
        deals = (session.query(DealRecord)
                 .filter(or_(DealRecord.state.in_(_PUBLIC_DEAL_STATES),
                             DealRecord.seller_id == user.id,
                             DealRecord.evaluator_id == user.id))
                 .order_by(DealRecord.created_at.desc()).limit(limit).all())
        return [_deal_out(session, d, user) for d in deals]


@app.get("/deals/{deal_id}", response_model=DealOut)
def get_deal(deal_id: str, user: User = Depends(current_user)) -> DealOut:
    with session_scope() as session:
        return _deal_out(session, _visible_deal(session, deal_id, user), user)


@app.post("/deals/{deal_id}/evaluations", response_model=AnalysisStatus, status_code=202)
def evaluate_deal(deal_id: str, request: EvaluationRequest,
                  user: User = Depends(current_user)) -> AnalysisStatus:
    """Buyer: run the engine against the listed data; only the score comes back."""
    with session_scope() as session:
        deal = _visible_deal(session, deal_id, user)
        if deal.state not in (DealState.LISTED.value, DealState.EVALUATED.value,
                              DealState.DECLINED.value):
            raise HTTPException(status_code=409,
                                detail=f"deal is {deal.state}; it cannot be evaluated now")
        _require_dataset(session, deal.dataset_id, "listed dataset")
        if request.reference_dataset_id:
            _require_dataset(session, request.reference_dataset_id, "reference dataset",
                             user=user)

        dataset = _load_for_spec(session, deal.dataset_id, None)
        spec = compile_spec(request.description, dataset,
                            target_column=request.target_column,
                            project_brief=request.project_brief)
        spec_row = SpecRecord(dataset_id=deal.dataset_id, spec=spec.model_dump(mode="json"),
                              owner_id=user.id)
        session.add(spec_row)
        session.flush()
        job = JobRecord(dataset_id=deal.dataset_id, spec_id=spec_row.id,
                        reference_dataset_id=request.reference_dataset_id,
                        deal_id=deal.id, mode="B", status=JobStatus.QUEUED.value,
                        owner_id=user.id)
        session.add(job)
        deal.state = DealState.EVALUATING.value
        deal.evaluator_id = user.id
        deal.history = [*(deal.history or []), _event("evaluation_started")]
        session.commit()
        status = _job_status(session, job, reveal_filename=False)
    worker.submit(status.id)
    return status


@app.post("/deals/{deal_id}/decision", response_model=DealOut)
def decide_deal(deal_id: str, request: DealDecision,
                user: User = Depends(current_user)) -> DealOut:
    """Buyer: accept or decline after reading the score-only report."""
    with session_scope() as session:
        deal = _visible_deal(session, deal_id, user)
        if deal.state != DealState.EVALUATED.value:
            raise HTTPException(status_code=409,
                                detail="a deal can only be decided after it is evaluated")
        if deal.evaluator_id != user.id:
            raise HTTPException(
                status_code=403,
                detail="only the buyer who ran the evaluation can accept or decline")
        accepted = request.decision == "accept"
        deal.state = (DealState.ACCEPTED if accepted else DealState.DECLINED).value
        deal.history = [*(deal.history or []),
                        _event("accepted" if accepted else "declined")]
        session.commit()
        return _deal_out(session, deal, user)


@app.post("/deals/{deal_id}/withdraw", response_model=DealOut)
def withdraw_deal(deal_id: str, user: User = Depends(current_user)) -> DealOut:
    """Seller: take the listing down."""
    with session_scope() as session:
        deal = _visible_deal(session, deal_id, user)
        if deal.seller_id != user.id:
            raise HTTPException(status_code=403,
                                detail="only the seller can withdraw a listing")
        if deal.state in (DealState.ACCEPTED.value, DealState.EVALUATING.value):
            raise HTTPException(status_code=409,
                                detail=f"a deal that is {deal.state} cannot be withdrawn")
        deal.state = DealState.WITHDRAWN.value
        deal.history = [*(deal.history or []), _event("withdrawn")]
        session.commit()
        return _deal_out(session, deal, user)


# -- helpers ------------------------------------------------------------------------


def _dataset_summary(record: DatasetRecord,
                     seen_before: Optional[list[dict[str, Any]]] = None) -> DatasetSummary:
    profile = record.profile or {}
    return DatasetSummary(
        id=record.id, filename=record.filename, format=record.format,
        size_bytes=record.size_bytes, n_rows=record.n_rows, n_cols=record.n_cols,
        fingerprint=record.fingerprint, created_at=record.created_at,
        available=bool(record.storage_ref),
        columns=profile.get("columns", {}),
        label_candidates=profile.get("label_candidates", []),
        seen_before=seen_before or [])


def _job_status(session, job: JobRecord, reveal_filename: bool = True) -> AnalysisStatus:
    dataset = session.get(DatasetRecord, job.dataset_id)
    report = session.get(ReportRecord, job.report_id) if job.report_id else None
    return AnalysisStatus(
        id=job.id, status=job.status, mode=job.mode, dataset_id=job.dataset_id,
        spec_id=job.spec_id, reference_dataset_id=job.reference_dataset_id,
        report_id=job.report_id, error=job.error, progress=job.progress,
        created_at=job.created_at, started_at=job.started_at, finished_at=job.finished_at,
        dataset_filename=(dataset.filename if dataset and reveal_filename
                          and job.mode == "A" else None),
        verdict=report.verdict if report else None,
        safety_score=report.safety_score if report else None,
        fit_score=report.fit_score if report else None)


class _NoCacheStatic(StaticFiles):
    """Serve the SPA with no-cache so edits show up without a hard reload.

    The bundle is tiny and fetched from localhost; correctness beats caching
    here. A production build would fingerprint filenames and cache aggressively.
    """

    def is_not_modified(self, response_headers, request_headers) -> bool:  # type: ignore[override]
        return False

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response


# The SPA is mounted last so API routes take precedence.
if STATIC_DIR.exists():
    app.mount("/static", _NoCacheStatic(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(
            STATIC_DIR / "index.html",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
