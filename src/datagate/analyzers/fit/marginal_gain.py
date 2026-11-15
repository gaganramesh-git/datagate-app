"""Fit: the empirical taste test — what does adding this data actually do?

Two cheap experiments with a linear proxy model on the shared feature space,
both scored on held-out folds of *your* task data:

- **Transfer**: train on the candidate alone, test on your data. Can this
  dataset teach your task from scratch?
- **Marginal gain**: train on part of your data, then on that part plus the
  candidate; the difference on your held-out fold is the gain, repeated across
  folds to give a range instead of a point estimate.

The range is what the report quotes as "predicted marginal gain". It is a
proxy-model estimate against a small sample of your data — a direction and a
rough size, not a promise about your production model.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold

from datagate.analyzers.base import Analyzer
from datagate.context import AnalysisContext
from datagate.scoring import clip01, sample_indices
from datagate.types import ScoredEvidence

MAX_CANDIDATE = 10_000
MIN_REFERENCE = 40


class MarginalGainAnalyzer(Analyzer):
    name = "marginal_gain"
    family = "fit"
    dimension = "Marginal gain"
    needs_reference = True

    def run(self, ctx: AnalysisContext) -> ScoredEvidence:
        joint = ctx.joint()
        if joint is None:
            return self.skip_no_reference()
        kind = ctx.task_kind
        if kind is None or not joint.has_labels:
            return self.skip("marginal gain needs the target column in both the candidate "
                             "and your reference data")
        cpos, rpos = np.where(joint.labeled_c)[0], np.where(joint.labeled_r)[0]
        if len(rpos) < MIN_REFERENCE or len(cpos) < 20 or joint.Xc.shape[1] == 0:
            return self.skip(f"needs at least {MIN_REFERENCE} labeled reference rows")
        if len(cpos) > MAX_CANDIDATE:
            cpos = cpos[sample_indices(len(cpos), MAX_CANDIDATE, ctx.seed)]
        Xc, Xr = joint.Xc[cpos], joint.Xr[rpos]

        if kind == "classification":
            yc, yr = joint.yc[cpos].astype(str), joint.yr[rpos].astype(str)
            _, counts = np.unique(yr, return_counts=True)
            if len(counts) < 2 or counts.min() < 2:
                return self.skip("reference needs at least two classes with 2+ rows each")
            folds = StratifiedKFold(n_splits=int(min(4, counts.min())), shuffle=True,
                                    random_state=ctx.seed)
            metric, chance = "balanced accuracy", 1.0 / len(counts)

            def fit_score(X_train, y_train, X_test, y_test) -> float:
                if len(np.unique(y_train)) < 2:
                    return chance
                model = LogisticRegression(C=5.0, max_iter=300).fit(X_train, y_train)
                return float(balanced_accuracy_score(y_test, model.predict(X_test)))
        else:
            yc, yr = joint.yc[cpos].astype(float), joint.yr[rpos].astype(float)
            folds = KFold(n_splits=4, shuffle=True, random_state=ctx.seed)
            metric, chance = "R²", 0.0

            def fit_score(X_train, y_train, X_test, y_test) -> float:
                model = Ridge(alpha=1.0).fit(X_train, y_train)
                return float(r2_score(y_test, model.predict(X_test)))

        transfer = fit_score(Xc, yc, Xr, yr)

        base_scores, with_scores = [], []
        for train, test in folds.split(Xr, yr):
            base_scores.append(fit_score(Xr[train], yr[train], Xr[test], yr[test]))
            with_scores.append(fit_score(
                sp.vstack([Xr[train], Xc], format="csr"), np.r_[yr[train], yc],
                Xr[test], yr[test]))
        gains = np.array(with_scores) - np.array(base_scores)
        lo, hi, mean = float(gains.min()), float(gains.max()), float(gains.mean())

        transfer_lift = clip01((transfer - chance) / max(1e-9, 1.0 - chance))
        marginal = clip01(0.5 + mean / 0.10 * 0.5)
        unit = "pts" if kind == "classification" else ""
        scale = 100 if kind == "classification" else 1
        gain_text = (f"{scale * lo:+.1f} to {scale * hi:+.1f} {unit} {metric}".replace("  ", " ")
                     if kind == "classification"
                     else f"{lo:+.3f} to {hi:+.3f} {metric}")

        recs = []
        if hi < 0:
            recs.append("Adding this data made the proxy model *worse* on your task in "
                        "every fold — do not mix it in without filtering.")
        elif transfer_lift < 0.2:
            recs.append("A model trained only on this data barely beats chance on your "
                        "task — it is unlikely to help much.")
        return self.evidence(
            score=0.6 * transfer_lift + 0.4 * marginal,
            confidence=0.75 if len(rpos) >= 200 else 0.6,
            summary=(f"trained on the candidate alone, a proxy scores {metric} "
                     f"{transfer:.2f} on your task data (chance {chance:.2f}); adding it to "
                     f"a sample of your data changes the score by {gain_text}"),
            metrics={"metric": metric, "transfer_score": transfer, "chance": chance,
                     "baseline_score": float(np.mean(base_scores)),
                     "with_candidate_score": float(np.mean(with_scores)),
                     "gain_mean": mean, "predicted_gain_range": [lo, hi],
                     "gain_text": gain_text, "folds": int(len(gains)),
                     "reference_rows": int(len(rpos))},
            recommendations=recs,
        )
