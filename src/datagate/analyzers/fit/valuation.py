"""Fit: per-row data value against your task, via exact KNN-Shapley.

Jia et al. (2019) showed the Shapley value of each training point under a
K-nearest-neighbor utility has a closed form — no retraining, no sampling.
With your reference data as the validation set, every candidate row gets a
value: positive when it helps classify your data correctly, negative when it
sits near your data but carries a conflicting label.

The values sum to the KNN utility — the share of your task rows the candidate
would classify correctly — which is the score, measured against what random
neighbors would achieve. Negative-value rows are reported, not quarantined:
they are a fit problem for *this* task, not a safety problem.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from datagate.analyzers.base import Analyzer
from datagate.context import AnalysisContext
from datagate.scoring import clip01, pct, sample_indices
from datagate.types import Flag, ScoredEvidence, Severity

K = 5
MAX_CANDIDATE = 4_000
MAX_REFERENCE = 400
MAX_REPORTED = 200


def knn_shapley(dist: np.ndarray, y_train: np.ndarray, y_val: np.ndarray,
                k: int = K) -> np.ndarray:
    """Exact KNN-Shapley values of the training points, averaged over validation points.

    ``dist`` is (n_val, n_train). Recursion, per validation point with training
    points sorted nearest-first (1-indexed i):
        s_N = 1[y_N = y_val] / N
        s_i = s_{i+1} + (1[y_i = y_val] - 1[y_{i+1} = y_val]) / K * min(K, i) / i
    """
    n_val, n = dist.shape
    order = np.argsort(dist, axis=1, kind="stable")
    match = (y_train[order] == y_val[:, None]).astype(float)

    ranks = np.arange(1, n)                       # i = 1 .. N-1
    weights = np.minimum(k, ranks) / (k * ranks)
    steps = (match[:, :-1] - match[:, 1:]) * weights
    sorted_values = np.empty_like(match)
    sorted_values[:, -1] = match[:, -1] / n
    sorted_values[:, :-1] = (sorted_values[:, -1:]
                             + np.cumsum(steps[:, ::-1], axis=1)[:, ::-1])

    values = np.zeros(n)
    np.add.at(values, order, sorted_values)
    return values / n_val


class ValuationAnalyzer(Analyzer):
    name = "valuation"
    family = "fit"
    dimension = "Data value"
    needs_reference = True

    def run(self, ctx: AnalysisContext) -> ScoredEvidence:
        joint = ctx.joint()
        if joint is None:
            return self.skip_no_reference()
        if ctx.task_kind != "classification" or not joint.has_labels:
            return self.skip("data valuation needs a classification target in both the "
                             "candidate and your reference data")
        cpos, rpos = np.where(joint.labeled_c)[0], np.where(joint.labeled_r)[0]
        if len(cpos) < K + 1 or len(rpos) < 10:
            return self.skip("too few labeled rows on one side")
        if len(cpos) > MAX_CANDIDATE:
            cpos = cpos[sample_indices(len(cpos), MAX_CANDIDATE, ctx.seed)]
        if len(rpos) > MAX_REFERENCE:
            rpos = rpos[sample_indices(len(rpos), MAX_REFERENCE, ctx.seed + 1)]

        yc, yr = joint.yc[cpos].astype(str), joint.yr[rpos].astype(str)
        values = knn_shapley(cdist(joint.emb_r[rpos], joint.emb_c[cpos]), yc, yr)

        utility = float(values.sum())
        # What random neighbors would score: P(candidate label == reference label).
        labels = np.unique(np.r_[yc, yr])
        chance = float(sum((yc == c).mean() * (yr == c).mean() for c in labels))
        helpful = float((values > 0).mean())
        harmful_mask = values < 0
        harmful = float(harmful_mask.mean())

        worst = np.argsort(values)[:MAX_REPORTED]
        flags = [
            Flag(row_index=int(joint.cand_idx[cpos[i]]), reason="negative_value",
                 severity=Severity.INFO,
                 detail=f"conflicts with similar rows of your task data (label {yc[i]!r})")
            for i in worst if values[i] < 0
        ]
        recs = []
        if harmful > 0.10:
            recs.append(
                f"{pct(harmful, 0)} of candidate rows have negative value for your task — "
                "they resemble your data but carry a different label. Review them before "
                "mixing this data in.")
        return self.evidence(
            score=clip01((utility - chance) / max(1e-9, 1.0 - chance)),
            confidence=0.7 if len(rpos) >= 100 else 0.5,
            summary=(f"nearest-neighbor utility on your task data is {utility:.2f} "
                     f"(random would be {chance:.2f}); {pct(helpful, 0)} of candidate rows "
                     f"help, {pct(harmful, 0)} hurt"),
            flags=flags,
            metrics={"knn_utility": utility, "chance": chance,
                     "helpful_fraction": helpful, "harmful_fraction": harmful,
                     "harmful_rows": int(harmful_mask.sum()),
                     "candidate_rows": int(len(cpos)), "reference_rows": int(len(rpos))},
            recommendations=recs,
        )
