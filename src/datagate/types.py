"""Core data contracts for DataGate.

The plugin contract is deliberately tiny: every analyzer produces a
ScoredEvidence. The aggregator only ever consumes ScoredEvidence. New
analyzers, modalities, and checks slot in without touching the core.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

ENGINE_VERSION = "0.2.0"


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    GENERATION = "generation"
    RETRIEVAL = "retrieval"
    UNKNOWN = "unknown"


class Modality(str, Enum):
    TABULAR = "tabular"
    TEXT = "text"
    IMAGE = "image"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ProjectSpec(BaseModel):
    """Structured intent parsed from the user's project inputs.

    Two free-text inputs feed this:
    - ``raw_description`` — the task objective ("what do you want to train?").
      Primary driver of task_type, modality, and target.
    - ``project_brief`` — the wider project context (the idea, the area of use).
      Mainly sharpens domain, deployment_context, and success_metric, which the
      fit analyzers and report framing key off.
    """

    raw_description: str
    project_brief: Optional[str] = None
    task_type: TaskType = TaskType.UNKNOWN
    modality: Modality = Modality.UNKNOWN
    target_column: Optional[str] = None
    domain: Optional[str] = None
    deployment_context: Optional[str] = None
    success_metric: Optional[str] = None
    # Which fields the LLM backend contributed, for transparency in the UI.
    llm_assisted_fields: list[str] = Field(default_factory=list)
    clarifying_questions: list[str] = Field(default_factory=list)

    @property
    def is_ambiguous(self) -> bool:
        return bool(self.clarifying_questions)


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK[self.value]


_SEVERITY_RANK = {"info": 0, "warning": 1, "critical": 2}


class Flag(BaseModel):
    """A single flagged sample with the reason it was flagged."""

    row_index: int
    reason: str
    severity: Severity = Severity.WARNING
    detail: Optional[str] = None


class ScoredEvidence(BaseModel):
    """The universal analyzer output: one scored signal plus its evidence."""

    analyzer: str
    family: str  # "security" | "fit"
    score: float = Field(ge=0.0, le=1.0)  # 1.0 = perfectly clean / perfect fit
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    # Human-facing name of the dimension this signal measures.
    dimension: Optional[str] = None
    flags: list[Flag] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    # Findings serious enough that the gate must not pass the dataset as-is.
    blockers: list[str] = Field(default_factory=list)
    # Concrete next steps, phrased from this analyzer's own evidence.
    recommendations: list[str] = Field(default_factory=list)
    skipped: bool = False
    skip_reason: Optional[str] = None
    # Skipped only because no reference dataset was supplied.
    needs_reference: bool = False


class CompositeScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    label: str  # human framing, e.g. "likely safe"


class Verdict(str, Enum):
    PASS = "pass"
    REVIEW = "review"
    BLOCK = "block"


class GatePolicy(BaseModel):
    """Thresholds the gate applies. Versioned with every report for reproducibility."""

    pass_safety: float = 0.80   # at or above: safety does not hold the dataset back
    block_safety: float = 0.50  # below: the dataset is blocked outright
    min_fit: float = 0.50       # below: flagged for review as a weak fit
    min_confidence: float = 0.30  # below: not enough evidence for a verdict


class GateDecision(BaseModel):
    verdict: Verdict
    headline: str
    reasons: list[str] = Field(default_factory=list)


class Report(BaseModel):
    """Final output of an analysis run. Deterministic given inputs + configs.

    Everything except ``runtime``, ``digest`` and ``signature`` is a pure
    function of the inputs, the seed and the policy; ``digest`` is the SHA-256
    of that deterministic part, and ``signature`` (optional) signs the digest.
    """

    dataset_path: str
    dataset_name: str = ""
    dataset_rows: int
    dataset_columns: int
    dataset_fingerprint: str = ""
    reference_name: Optional[str] = None
    reference_rows: Optional[int] = None
    reference_fingerprint: Optional[str] = None
    spec: ProjectSpec
    safety: CompositeScore
    fit: CompositeScore
    decision: GateDecision
    policy: GatePolicy = Field(default_factory=GatePolicy)
    predicted_gain_range: Optional[str] = None
    fit_breakdown: dict[str, float] = Field(default_factory=dict)
    safety_breakdown: dict[str, float] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)
    # Worst-first, capped for report size; the totals below are uncapped.
    quarantine_list: list[Flag] = Field(default_factory=list)
    quarantine_total: int = 0
    quarantine_by_reason: dict[str, int] = Field(default_factory=dict)
    evidence: list[ScoredEvidence] = Field(default_factory=list)
    seed: int = 7
    engine_version: str = ENGINE_VERSION
    runtime: dict[str, Any] = Field(default_factory=dict)
    digest: Optional[str] = None
    signature: Optional[dict[str, str]] = None
