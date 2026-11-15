"""Fit: is there learnable signal at all?

A small proxy model is cross-validated on a sample; the score is its lift over
chance. This answers "do the features say anything about the target?" — it is
*not* a prediction of the user's final accuracy, and makes no gain claim.
(The marginal-gain analyzer does that, and only when it has the user's own
task data to measure against.)
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold, cross_val_score

from datagate.analyzers.base import Analyzer
from datagate.context import AnalysisContext
from datagate.scoring import clip01
from datagate.types import ScoredEvidence


class LearnabilityAnalyzer(Analyzer):
    name = "learnability"
    family = "fit"
    dimension = "Learnable signal"

    def run(self, ctx: AnalysisContext) -> ScoredEvidence:
        kind = ctx.task_kind
        if kind is None:
            return self.skip("no prediction target — learnability applies to "
                             "classification and regression")
        target = ctx.target

        if kind == "classification":
            oof = ctx.oof()
            if oof is None:
                return self.skip("not enough labeled rows per class to cross-validate")
            lift = clip01((oof.balanced_accuracy - oof.chance) / max(1e-9, 1 - oof.chance))
            metric, value, baseline = "balanced accuracy", oof.balanced_accuracy, oof.chance
            n_rows = len(oof.pos)
        else:
            view = ctx.view()
            pos = np.where(view.labeled)[0]
            if len(pos) < 100 or view.X.shape[1] == 0:
                return self.skip("too few labeled rows or no feature columns")
            X = ctx.model_matrix(view)[pos]
            y = view.y[pos].astype(float)
            if np.std(y) == 0:
                return self.skip("target is constant")
            cv = KFold(n_splits=3, shuffle=True, random_state=ctx.seed)
            r2 = float(np.mean(cross_val_score(
                ctx.regressor(view.featurizer.has_text), X, y, cv=cv, scoring="r2")))
            lift = clip01(r2)
            metric, value, baseline = "R²", r2, 0.0
            n_rows = len(pos)

        strength = ("strong" if lift >= 0.6 else "moderate" if lift >= 0.3
                    else "weak" if lift >= 0.1 else "almost no")
        recs = []
        if lift < 0.1:
            recs.append(f"The features carry almost no signal about {target!r}. Check that "
                        "the right target column is selected and that labels are not random.")
        return self.evidence(
            score=lift, confidence=0.7,
            summary=(f"a quick proxy model reaches {metric} {value:.2f} on held-out folds "
                     f"(chance {baseline:.2f}) — {strength} learnable signal about {target!r}"),
            metrics={"metric": metric, "proxy_score": value, "chance": baseline,
                     "lift": lift, "rows": n_rows},
            recommendations=recs,
        )
