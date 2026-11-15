"""Analysis Orchestrator: load, profile, run every analyzer, aggregate, decide.

Analyzers run in a small thread pool (their heavy work is numpy/sklearn, which
releases the GIL) and results are collected in a fixed order, so parallelism
never changes the report. One crashing analyzer is recorded and skipped — it
cannot sink the run.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional

from datagate.aggregate import aggregate
from datagate.analyzers.base import Analyzer
from datagate.analyzers.fit.alignment import AlignmentAnalyzer
from datagate.analyzers.fit.coverage import CoverageAnalyzer
from datagate.analyzers.fit.learnability import LearnabilityAnalyzer
from datagate.analyzers.fit.marginal_gain import MarginalGainAnalyzer
from datagate.analyzers.fit.shift import ShiftAnalyzer
from datagate.analyzers.fit.structure import StructureAnalyzer
from datagate.analyzers.fit.target_leakage import TargetLeakageAnalyzer
from datagate.analyzers.fit.valuation import ValuationAnalyzer
from datagate.analyzers.security.contamination import ContaminationAnalyzer
from datagate.analyzers.security.duplicates import DuplicateAnalyzer
from datagate.analyzers.security.hidden import HiddenContentAnalyzer
from datagate.analyzers.security.labels import LabelAnalyzer
from datagate.analyzers.security.pii import PIIAnalyzer
from datagate.analyzers.security.poisoning import PoisoningAnalyzer
from datagate.analyzers.security.provenance import ProvenanceAnalyzer
from datagate.analyzers.security.quality import QualityAnalyzer
from datagate.context import AnalysisContext
from datagate.ingest.loader import Dataset, load_dataset
from datagate.ingest.profiler import profile_dataset
from datagate.registry import Registry
from datagate.signing import report_digest
from datagate.spec.compiler import compile_spec
from datagate.types import Flag, GatePolicy, ProjectSpec, Report, ScoredEvidence

log = logging.getLogger("datagate.orchestrator")

MAX_REPORT_FLAGS = 2_000        # quarantine rows embedded in the report
MAX_EVIDENCE_FLAGS = 200        # flags kept per analyzer in the report
MAX_WORKERS = 4

ProgressFn = Callable[[str, int, int], None]


@dataclass
class AnalysisResult:
    """The report plus what is too big to embed in it."""

    report: Report
    quarantine: list[Flag]      # complete, one flag per row
    dataset: Dataset
    reference: Optional[Dataset] = None


def default_analyzers(registry: Optional[Registry] = None) -> list[Analyzer]:
    return [
        # Security — the metal detector.
        QualityAnalyzer(),
        LabelAnalyzer(),
        DuplicateAnalyzer(),
        PIIAnalyzer(),
        HiddenContentAnalyzer(),
        PoisoningAnalyzer(),
        ContaminationAnalyzer(),
        ProvenanceAnalyzer(registry),
        # Fit — the taste test.
        StructureAnalyzer(),
        LearnabilityAnalyzer(),
        TargetLeakageAnalyzer(),
        CoverageAnalyzer(),
        ShiftAnalyzer(),
        AlignmentAnalyzer(),
        ValuationAnalyzer(),
        MarginalGainAnalyzer(),
    ]


def analyze(
    dataset_path: str,
    description: str = "",
    target_column: Optional[str] = None,
    reference_path: Optional[str] = None,
    spec: Optional[ProjectSpec] = None,
    project_brief: Optional[str] = None,
    analyzers: Optional[list[Analyzer]] = None,
    policy: Optional[GatePolicy] = None,
    seed: int = 7,
    progress: Optional[ProgressFn] = None,
    dataset_name: Optional[str] = None,
    reference_name: Optional[str] = None,
    registry: Optional[Registry] = None,
    parallel: bool = True,
) -> AnalysisResult:
    started = time.perf_counter()
    dataset = load_dataset(dataset_path, name=dataset_name)
    profile_dataset(dataset)
    reference = load_dataset(reference_path, name=reference_name) if reference_path else None
    if spec is None:
        spec = compile_spec(description, dataset, target_column=target_column,
                            project_brief=project_brief)
    policy = policy or GatePolicy()

    ctx = AnalysisContext(dataset, spec, reference=reference, seed=seed)
    plugins = analyzers if analyzers is not None else default_analyzers(registry)
    timings: dict[str, float] = {}
    errors: dict[str, str] = {}
    done = 0

    def run_one(analyzer: Analyzer) -> ScoredEvidence:
        nonlocal done
        t0 = time.perf_counter()
        try:
            result = analyzer.run(ctx)
        except Exception as exc:
            log.warning("analyzer %s crashed", analyzer.name, exc_info=True)
            errors[analyzer.name] = f"{type(exc).__name__}: {exc}"
            result = analyzer.skip(f"crashed: {type(exc).__name__}: {exc}")
        timings[analyzer.name] = round(time.perf_counter() - t0, 3)
        done += 1
        if progress is not None:
            try:
                progress(analyzer.name, done, len(plugins))
            except Exception:
                log.debug("progress callback failed", exc_info=True)
        return result

    if parallel and len(plugins) > 1:
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(plugins)),
                                thread_name_prefix="datagate-analyzer") as pool:
            evidence = list(pool.map(run_one, plugins))
    else:
        evidence = [run_one(a) for a in plugins]

    agg = aggregate(evidence, policy=policy, spec=spec)

    gain = next((e.metrics.get("gain_text") for e in evidence
                 if e.analyzer == "marginal_gain" and not e.skipped), None)

    # Keep the report compact: per-analyzer flags are capped (counts stay in metrics).
    slim = [e.model_copy(update={"flags": e.flags[:MAX_EVIDENCE_FLAGS]}) for e in evidence]

    report = Report(
        dataset_path=str(dataset_path),
        dataset_name=dataset.name,
        dataset_rows=dataset.n_rows,
        dataset_columns=dataset.n_cols,
        dataset_fingerprint=dataset.fingerprint,
        reference_name=reference.name if reference else None,
        reference_rows=reference.n_rows if reference else None,
        reference_fingerprint=reference.fingerprint if reference else None,
        spec=spec,
        safety=agg.safety,
        fit=agg.fit,
        decision=agg.decision,
        policy=policy,
        predicted_gain_range=(f"{gain} (linear-proxy estimate on your task data)"
                              if gain else None),
        fit_breakdown=agg.fit_breakdown,
        safety_breakdown=agg.safety_breakdown,
        recommendations=agg.recommendations,
        quarantine_list=agg.quarantine[:MAX_REPORT_FLAGS],
        quarantine_total=len(agg.quarantine),
        quarantine_by_reason=agg.quarantine_by_reason,
        evidence=slim,
        seed=seed,
        runtime={"seconds": round(time.perf_counter() - started, 3),
                 "analyzer_seconds": timings, "errors": errors},
    )
    report.digest = report_digest(report)
    return AnalysisResult(report=report, quarantine=agg.quarantine,
                          dataset=dataset, reference=reference)


def run_analysis(dataset_path: str, description: str = "", **kwargs) -> Report:
    """Convenience wrapper returning only the report."""
    return analyze(dataset_path, description, **kwargs).report
