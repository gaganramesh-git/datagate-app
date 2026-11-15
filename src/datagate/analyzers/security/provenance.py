"""Provenance: does this dataset overlap datasets we already know?

Compares the candidate's row sketch against the local registry
(`datagate registry add ...`), in both directions:

- *containment* — what share of the candidate is the known dataset
  ("this is mostly a re-upload of X");
- *coverage* — what share of the known dataset is inside the candidate
  ("your whole eval set is in here"), which can be total even when the known
  set is a sliver of a large candidate.

Overlap with a registered benchmark or eval set is contamination; overlap with
a public dataset means this is (partly) a re-upload. The language is
deliberately "matches", never "ownership proven".
"""

from __future__ import annotations

from typing import Optional

from datagate.analyzers.base import Analyzer
from datagate.context import AnalysisContext
from datagate.registry import Match, Registry
from datagate.scoring import pct, plural, risk_score
from datagate.types import ScoredEvidence

_SENSITIVE_KINDS = ("benchmark", "eval")


def _describe(m: Match) -> str:
    return (f"{pct(m.coverage, 0)} of {m.kind} set {m.name!r} is in this dataset "
            f"({pct(m.containment, 0)} of its rows)")


class ProvenanceAnalyzer(Analyzer):
    name = "provenance"
    family = "security"
    dimension = "Provenance"

    def __init__(self, registry: Optional[Registry] = None) -> None:
        self.registry = registry

    def run(self, ctx: AnalysisContext) -> ScoredEvidence:
        registry = self.registry or Registry()
        entries = registry.entries()
        # A dataset must not match itself, nor the reference it is compared against.
        ignore = {ctx.dataset.fingerprint}
        if ctx.reference is not None:
            ignore.add(ctx.reference.fingerprint)
        known = [e for e in entries if e.fingerprint not in ignore]
        if not known:
            return self.skip("no known datasets registered to compare against "
                             "(`datagate registry add`)")

        matches = registry.match(ctx.dataset, ignore_fingerprints=frozenset(ignore))
        if not matches:
            return self.evidence(
                score=1.0, confidence=0.3,
                summary=f"no overlap with {plural(len(known), 'known dataset')}",
                metrics={"known_datasets": len(known), "matches": []},
            )

        sensitive = [m for m in matches if m.kind in _SENSITIVE_KINDS]
        worst = max((max(m.containment, m.coverage) for m in sensitive), default=0.0)
        recommendations, blockers = [], []
        for m in matches[:3]:
            if m.kind in _SENSITIVE_KINDS:
                recommendations.append(
                    f"About {_describe(m)} — evaluating on it after training on this data "
                    "would be contaminated. Remove the overlapping rows first.")
            elif m.kind == "public":
                recommendations.append(
                    f"≈{pct(m.containment, 0)} of rows match public dataset {m.name!r} — "
                    "this is at least partly a re-upload of freely available data.")
        heavy = [m for m in sensitive if max(m.containment, m.coverage) >= 0.5]
        if heavy:
            blockers.append(f"Contaminated: about {_describe(heavy[0])}.")

        top = matches[0]
        if top.kind in _SENSITIVE_KINDS:
            summary = "about " + _describe(top)
        else:
            summary = (f"≈{pct(top.containment, 0)} of rows match known {top.kind} dataset "
                       f"{top.name!r}")
        return self.evidence(
            score=risk_score(worst, ok=0.02, bad=0.5),
            confidence=0.8,
            summary=(summary + (" (identical file)" if top.exact else "")
                     + (f"; {plural(len(matches) - 1, 'other match', 'other matches')}"
                        if len(matches) > 1 else "")),
            metrics={"known_datasets": len(known),
                     "matches": [{"name": m.name, "kind": m.kind,
                                  "containment": round(m.containment, 4),
                                  "coverage": round(m.coverage, 4), "exact": m.exact}
                                 for m in matches[:10]]},
            blockers=blockers, recommendations=recommendations,
        )
