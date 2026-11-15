"""Analyzer plugin contract: run(ctx) -> ScoredEvidence.

An analyzer reads the shared AnalysisContext and returns exactly one
ScoredEvidence. It never raises for "this check doesn't apply" — it returns a
skip, which carries zero confidence and drops out of the aggregate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from datagate.context import AnalysisContext
from datagate.types import Flag, ScoredEvidence

# Per-analyzer cap on stored flags. Counts in ``metrics`` are always uncapped.
MAX_FLAGS = 50_000

_REGISTRY: dict[str, type["Analyzer"]] = {}


class Analyzer(ABC):
    name: str = "analyzer"
    family: str = "security"      # or "fit"
    dimension: str = ""           # human-facing label for the report
    needs_reference: bool = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "name", None) and cls.name != "analyzer":
            _REGISTRY[cls.name] = cls

    @abstractmethod
    def run(self, ctx: AnalysisContext) -> ScoredEvidence: ...

    def evidence(self, score: float, confidence: float, summary: str,
                 flags: Optional[list[Flag]] = None, **kwargs: Any) -> ScoredEvidence:
        return ScoredEvidence(
            analyzer=self.name, family=self.family, dimension=self.dimension or self.name,
            score=min(1.0, max(0.0, float(score))),
            confidence=min(1.0, max(0.0, float(confidence))),
            summary=summary, flags=(flags or [])[:MAX_FLAGS], **kwargs,
        )

    def skip(self, reason: str, needs_reference: bool = False) -> ScoredEvidence:
        """A neutral, zero-confidence result that the aggregator ignores."""
        return ScoredEvidence(
            analyzer=self.name, family=self.family, dimension=self.dimension or self.name,
            score=1.0, confidence=0.0, summary=f"Skipped: {reason}",
            skipped=True, skip_reason=reason, needs_reference=needs_reference,
        )

    def skip_no_reference(self) -> ScoredEvidence:
        return self.skip("needs a sample of your own task data (reference dataset)",
                         needs_reference=True)


def registered_analyzers() -> dict[str, type[Analyzer]]:
    return dict(_REGISTRY)
