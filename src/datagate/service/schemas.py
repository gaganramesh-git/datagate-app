"""API request/response models.

Wire schemas are deliberately separate from the engine's internal types, except
where the engine type *is* the contract (ProjectSpec, Report) — those are
embedded verbatim so the API can never drift from what the engine computed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from datagate.types import ProjectSpec


class DatasetSummary(BaseModel):
    id: str
    filename: str
    format: str
    size_bytes: int
    n_rows: int
    n_cols: int
    fingerprint: str
    created_at: datetime
    available: bool = True          # False once the data has been purged
    columns: dict[str, Any] = Field(default_factory=dict)
    label_candidates: list[str] = Field(default_factory=list)
    # Known datasets this upload overlaps (name, kind, containment, exact).
    seen_before: list[dict[str, Any]] = Field(default_factory=list)


class SpecRequest(BaseModel):
    description: str = Field(min_length=1, max_length=10_000,
                             description="Training objective: what you want to train")
    project_brief: Optional[str] = Field(
        default=None, max_length=100_000,
        description="Wider project context: the idea and area of use")
    dataset_id: Optional[str] = None
    target_column: Optional[str] = None


class SpecResponse(BaseModel):
    id: str
    spec: ProjectSpec
    needs_clarification: bool


class SpecUpdate(BaseModel):
    """User corrections from the spec-review screen. Any field may be edited."""

    spec: ProjectSpec


class AnalysisRequest(BaseModel):
    dataset_id: str
    spec_id: str
    reference_dataset_id: Optional[str] = None
    mode: str = "A"


class AnalysisStatus(BaseModel):
    id: str
    status: str
    mode: str
    dataset_id: str
    spec_id: str
    reference_dataset_id: Optional[str] = None
    report_id: Optional[str] = None
    error: Optional[str] = None
    progress: Optional[dict[str, Any]] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    # Denormalized for the history view.
    dataset_filename: Optional[str] = None
    verdict: Optional[str] = None
    safety_score: Optional[float] = None
    fit_score: Optional[float] = None


class ReportResponse(BaseModel):
    id: str
    job_id: str
    engine_version: str
    created_at: datetime
    # The engine's payload records the content-addressed storage path, which is
    # what makes a report reproducible. The original filename is carried
    # alongside it for display rather than overwriting that.
    dataset_filename: str
    blind: bool = False
    deal_id: Optional[str] = None      # set on blind reports: the deal they evaluated
    deal_state: Optional[str] = None
    exports_available: bool = True
    report: dict[str, Any]


class RowsResponse(BaseModel):
    columns: list[str]
    rows: list[dict[str, Any]]


class VerifyResponse(BaseModel):
    ok: bool
    digest_ok: bool
    signature_ok: bool
    trusted: Optional[bool] = None
    detail: str


class Meta(BaseModel):
    version: str
    engine_version: str
    auth_enabled: bool
    llm_enabled: bool
    signing_public_key: str
    analyzers: list[dict[str, Any]]


class DealCreate(BaseModel):
    dataset_id: str
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=5_000)
    asking_price: str = Field(default="", max_length=64)


class DealOut(BaseModel):
    """What a prospective buyer may see: the shape of the data, never its rows."""

    id: str
    title: str
    description: str
    asking_price: str
    state: str
    n_rows: int
    n_cols: int
    format: str
    columns: dict[str, str]            # column name -> kind
    fingerprint_prefix: str
    report_id: Optional[str] = None
    # The latest evaluation's outcome, so a buyer never decides blind.
    verdict: Optional[str] = None
    safety_score: Optional[float] = None
    fit_score: Optional[float] = None
    history: list[dict[str, Any]] = Field(default_factory=list)
    # Who is asking: the seller who listed it, or the buyer behind the latest
    # blind evaluation. Everyone else sees only the listing itself.
    is_seller: bool = False
    is_evaluator: bool = False
    created_at: datetime


class EvaluationRequest(BaseModel):
    description: str = Field(min_length=1, max_length=10_000)
    project_brief: Optional[str] = Field(default=None, max_length=100_000)
    target_column: Optional[str] = None
    reference_dataset_id: Optional[str] = None


class DealDecision(BaseModel):
    decision: str = Field(pattern="^(accept|decline)$")
