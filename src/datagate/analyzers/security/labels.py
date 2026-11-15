"""Label-error detection via confident learning (Northcutt et al., 2021).

Out-of-fold class probabilities come from the shared proxy classifier. Each
class gets its own confidence threshold — the mean predicted probability of
that class among rows labeled with it — which makes the rule robust to class
imbalance. A row is a suspected mislabel when the model is *confidently* in a
different class than the given label.

The check abstains when the proxy can barely predict the labels at all: a
model that knows nothing cannot tell you which labels are wrong.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from datagate.analyzers.base import Analyzer
from datagate.context import AnalysisContext
from datagate.scoring import pct, risk_score
from datagate.types import Flag, ScoredEvidence, Severity

MIN_MARGIN = 0.10   # confident class must beat the given label by at least this
MIN_LIFT = 0.15     # proxy must clear chance by this much to be trusted


class LabelAnalyzer(Analyzer):
    name = "labels"
    family = "security"
    dimension = "Label quality"

    def run(self, ctx: AnalysisContext) -> ScoredEvidence:
        if ctx.task_kind != "classification":
            return self.skip("label-error detection needs a classification target")
        oof = ctx.oof()
        if oof is None:
            return self.skip("not enough labeled rows per class to cross-validate")

        lift = (oof.balanced_accuracy - oof.chance) / max(1e-9, 1.0 - oof.chance)
        if lift < MIN_LIFT:
            return self.skip(
                "the proxy model cannot predict these labels well enough to judge them "
                f"(balanced accuracy {oof.balanced_accuracy:.2f}, chance {oof.chance:.2f})")

        proba, classes, y = oof.proba, oof.classes, oof.y
        col = {c: i for i, c in enumerate(classes)}
        given = np.array([col[v] for v in y])
        rows = np.arange(len(y))

        thresholds = np.array([proba[given == j, j].mean() for j in range(len(classes))])
        confident = proba >= thresholds[None, :]
        best = np.where(confident, proba, -1.0).argmax(axis=1)
        p_given = proba[rows, given]
        p_best = proba[rows, best]
        suspect = confident.any(axis=1) & (best != given) & (p_best - p_given >= MIN_MARGIN)

        view = ctx.view()
        order = np.argsort(-(p_best - p_given))
        flags = [
            Flag(row_index=int(view.idx[oof.pos[i]]),
                 reason="suspected_mislabel", severity=Severity.WARNING,
                 detail=(f"labeled {classes[given[i]]!r}, looks like {classes[best[i]]!r} "
                         f"(p={p_best[i]:.2f} vs {p_given[i]:.2f})"))
            for i in order if suspect[i]
        ]

        n_suspect = int(suspect.sum())
        rate = n_suspect / len(y)
        pairs = Counter((str(classes[given[i]]), str(classes[best[i]]))
                        for i in np.where(suspect)[0])
        sampled = len(view.idx) < ctx.dataset.n_rows

        recommendations = []
        if n_suspect:
            recommendations.append(
                f"Review the {n_suspect:,} rows with suspected label errors "
                f"({pct(rate)} of those checked); relabel or drop them before training.")

        return self.evidence(
            score=risk_score(rate, ok=0.005, bad=0.15),
            confidence=min(0.9, 0.5 + 0.4 * lift),
            summary=(f"{n_suspect:,} suspected label errors in {len(y):,} checked rows "
                     f"({pct(rate)})" + (" — sampled" if sampled else "")),
            flags=flags,
            metrics={
                "checked_rows": int(len(y)),
                "suspected_mislabels": n_suspect,
                "mislabel_rate": rate,
                "proxy_balanced_accuracy": oof.balanced_accuracy,
                "sampled": sampled,
                "top_confusions": [
                    {"given": g, "likely": b, "count": c}
                    for (g, b), c in pairs.most_common(5)
                ],
            },
            recommendations=recommendations,
        )
