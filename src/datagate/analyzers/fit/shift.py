"""Fit: distribution match, by adversarial validation.

Train a classifier to tell candidate rows from your own task data. If it
cannot (AUC ≈ 0.5), the two are drawn from the same distribution. If it can,
its strongest features say *what* is different — usually more useful than the
number itself.

Data from another source is almost always somewhat distinguishable, so the
score only starts dropping once the gap is substantial, and this signal
carries moderate confidence: a detectable shift is a reason to look, not a
verdict that the data is useless.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from datagate.analyzers.base import Analyzer
from datagate.context import AnalysisContext
from datagate.scoring import clip01, sample_indices
from datagate.types import ScoredEvidence

MAX_CANDIDATE = 10_000


class ShiftAnalyzer(Analyzer):
    name = "shift"
    family = "fit"
    dimension = "Distribution match"
    needs_reference = True

    def run(self, ctx: AnalysisContext) -> ScoredEvidence:
        joint = ctx.joint()
        if joint is None:
            return self.skip_no_reference()
        Xc, Xr = joint.Xc, joint.Xr
        if Xc.shape[1] == 0:
            return self.skip("no shared feature columns between candidate and reference")
        if min(Xc.shape[0], Xr.shape[0]) < 30:
            return self.skip("too few rows for a distribution test (<30)")

        if Xc.shape[0] > MAX_CANDIDATE:
            Xc = Xc[sample_indices(Xc.shape[0], MAX_CANDIDATE, ctx.seed)]
        X = sp.vstack([Xc, Xr], format="csr")
        origin = np.r_[np.zeros(Xc.shape[0], dtype=int), np.ones(Xr.shape[0], dtype=int)]

        model = LogisticRegression(C=1.0, max_iter=300, class_weight="balanced")
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=ctx.seed)
        proba = cross_val_predict(model, X, origin, cv=cv, method="predict_proba")[:, 1]
        auc = float(roc_auc_score(origin, proba))
        auc = max(auc, 1.0 - auc)

        coef = model.fit(X, origin).coef_.ravel()
        names = joint.featurizer.feature_names()
        order = np.argsort(coef)
        typical_candidate = [names[i] for i in order[:6] if coef[i] < 0]
        typical_reference = [names[i] for i in order[::-1][:6] if coef[i] > 0]

        level = ("indistinguishable" if auc < 0.6 else "mild shift" if auc < 0.75
                 else "clear shift" if auc < 0.9 else "strong shift")
        recs = []
        if auc >= 0.9:
            recs.append(
                "The candidate is easy to tell apart from your task data. Typical of the "
                f"candidate: {', '.join(typical_candidate[:5]) or '—'}; typical of your "
                f"data: {', '.join(typical_reference[:5]) or '—'}. Check these are "
                "differences you can live with.")
        return self.evidence(
            score=1.0 - clip01((auc - 0.6) / 0.4),
            confidence=0.55,
            summary=(f"a classifier separates the candidate from your task data with AUC "
                     f"{auc:.2f} ({level})"),
            metrics={"auc": auc, "level": level,
                     "typical_of_candidate": typical_candidate,
                     "typical_of_reference": typical_reference},
            recommendations=recs,
        )
