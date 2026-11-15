"""Fit: directional value — does a taste of this data steer a model toward your goal?

Learning from data nudges a model in some direction. We measure that direction
for the candidate and for your own task data, from the same starting point,
and compare:

- **Gradient alignment**: cosine between the two first-step gradients of a
  linear model. Aligned → learning from the candidate moves toward your task.
  Orthogonal → irrelevant. Opposed → actively harmful (e.g. a flipped label
  convention).
- **Taste test**: take a few gradient steps on the candidate only and measure
  how much your task's loss actually drops, relative to the same number of
  steps on your own data (the best a taste could do).

Everything is closed-form numpy on the shared feature space — no training run,
no GPU. A neural encoder would sharpen the features, not change the method.
"""

from __future__ import annotations

import numpy as np

from datagate.analyzers.base import Analyzer
from datagate.context import AnalysisContext
from datagate.scoring import clip01
from datagate.types import ScoredEvidence

STEPS = 25
MIN_ROWS = 20


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class _Linear:
    """Softmax (classification) or least-squares (regression) linear model."""

    def __init__(self, classification: bool) -> None:
        self.classification = classification

    def predict(self, X, theta: np.ndarray) -> np.ndarray:
        out = np.asarray(X @ theta)
        return _softmax(out) if self.classification else out

    def grad(self, X, Y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return np.asarray(X.T @ (self.predict(X, theta) - Y)) / X.shape[0]

    def loss(self, X, Y: np.ndarray, theta: np.ndarray) -> float:
        p = self.predict(X, theta)
        if self.classification:
            return float(-np.mean(np.sum(Y * np.log(np.clip(p, 1e-12, None)), axis=1)))
        return float(0.5 * np.mean((p - Y) ** 2))

    def descend(self, X, Y: np.ndarray, steps: int) -> np.ndarray:
        theta = np.zeros((X.shape[1], Y.shape[1]))
        mean_sq_norm = float(X.multiply(X).sum() / X.shape[0])
        lr = 1.0 / max(mean_sq_norm, 1e-9)
        for _ in range(steps):
            theta -= lr * self.grad(X, Y, theta)
        return theta


class AlignmentAnalyzer(Analyzer):
    name = "alignment"
    family = "fit"
    dimension = "Directional value"
    needs_reference = True

    def run(self, ctx: AnalysisContext) -> ScoredEvidence:
        joint = ctx.joint()
        if joint is None:
            return self.skip_no_reference()
        kind = ctx.task_kind
        if kind is None or not joint.has_labels:
            return self.skip("directional value needs the target column in both the "
                             "candidate and your reference data")
        mc, mr = joint.labeled_c, joint.labeled_r
        if mc.sum() < MIN_ROWS or mr.sum() < MIN_ROWS or joint.Xc.shape[1] == 0:
            return self.skip("too few labeled rows on one side")
        Xc, Xr = joint.Xc[np.where(mc)[0]], joint.Xr[np.where(mr)[0]]

        if kind == "classification":
            yc, yr = joint.yc[mc].astype(str), joint.yr[mr].astype(str)
            if not set(yc) & set(yr):
                return self.evidence(
                    score=0.05, confidence=0.8,
                    summary="candidate and reference labels do not overlap — the "
                            "candidate cannot teach your task as labeled",
                    metrics={"cosine": None},
                    recommendations=["Map the candidate's labels onto your label set."])
            classes = np.unique(np.r_[yc, yr])
            index = {c: i for i, c in enumerate(classes)}
            Yc = np.eye(len(classes))[[index[v] for v in yc]]
            Yr = np.eye(len(classes))[[index[v] for v in yr]]
        else:
            yc, yr = joint.yc[mc].astype(float), joint.yr[mr].astype(float)
            mu, sd = float(np.mean(yr)), float(np.std(yr)) or 1.0
            Yc, Yr = ((yc - mu) / sd)[:, None], ((yr - mu) / sd)[:, None]

        model = _Linear(kind == "classification")
        zero = np.zeros((Xc.shape[1], Yc.shape[1]))
        gc, gr = model.grad(Xc, Yc, zero), model.grad(Xr, Yr, zero)
        denom = np.linalg.norm(gc) * np.linalg.norm(gr)
        cosine = float(np.sum(gc * gr) / denom) if denom > 0 else 0.0

        start = model.loss(Xr, Yr, zero)
        after_candidate = model.loss(Xr, Yr, model.descend(Xc, Yc, STEPS))
        after_own = model.loss(Xr, Yr, model.descend(Xr, Yr, STEPS))
        best_drop = start - after_own
        efficiency = (start - after_candidate) / best_drop if best_drop > 1e-9 else 0.0
        efficiency = float(np.clip(efficiency, -1.0, 1.5))

        direction = ("aligned" if cosine >= 0.3 else "weakly aligned" if cosine >= 0.1
                     else "orthogonal" if cosine > -0.1 else "opposed")
        recs, blockers = [], []
        if direction == "opposed":
            recs.append("Learning from this data pushes a model *away* from your task — "
                        "check for an inverted or different label convention.")
            blockers.append("Learning from this data moves a model away from your task "
                            f"(gradient cosine {cosine:+.2f}).")
        elif direction == "orthogonal":
            recs.append("This data barely moves a model toward your task; it may be "
                        "about something else.")
        return self.evidence(
            score=0.5 * clip01((cosine + 0.1) / 0.9) + 0.5 * clip01(efficiency),
            confidence=0.7,
            summary=(f"learning direction is {direction} with your task (gradient cosine "
                     f"{cosine:+.2f}); a taste of the candidate achieves "
                     f"{100 * efficiency:.0f}% of the loss reduction your own data would"),
            metrics={"cosine": cosine, "direction": direction,
                     "taste_efficiency": efficiency, "reference_loss_start": start,
                     "reference_loss_after_candidate": after_candidate,
                     "reference_loss_after_own_data": after_own},
            recommendations=recs, blockers=blockers,
        )
