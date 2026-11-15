"""Score-only reports: the "only the score leaves" guarantee of marketplace mode.

In a blind evaluation a buyer vets a seller's dataset before purchase. The
buyer must learn whether the data is safe and fits their task — and nothing
that would let them reconstruct the data without paying for it.

``score_only`` derives a buyer-safe report from a full one:
- no row indices, no quarantined rows, no per-row flags;
- analyzer metrics reduced to plain numbers (tokens, column values and other
  content-bearing fields are dropped);
- recommendations that quote dataset content are dropped;
- the storage path is removed; counts and scores are kept.

The result gets its own digest and is meant to be signed, so both parties can
verify what was issued.
"""

from __future__ import annotations

from datagate.signing import report_digest
from datagate.types import Report

BLIND_NOTE = ("Blind evaluation: scores and aggregate counts only. Row-level findings "
              "stay with the seller until the deal is accepted.")


def _numeric_only(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items()
            if isinstance(v, (int, float, bool)) and not isinstance(v, str)}


def score_only(report: Report, title: str = "listed dataset") -> Report:
    evidence = [
        e.model_copy(update={
            "flags": [],
            "metrics": _numeric_only(e.metrics),
            "recommendations": [],
        })
        for e in report.evidence
    ]
    blind = report.model_copy(deep=True, update={
        "dataset_path": "",
        "dataset_name": title,
        "evidence": evidence,
        "quarantine_list": [],
        "recommendations": [BLIND_NOTE],
        "runtime": {},
        "signature": None,
    })
    blind.digest = report_digest(blind)
    return blind
