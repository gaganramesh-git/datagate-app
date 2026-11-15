"""Fit: gap coverage — does the candidate reach the regions your task lives in?

Both datasets are embedded in one space. A reference row is *covered* when the
candidate has neighbors about as close to it as the reference's own neighbors
are (the radius is calibrated from the reference itself, so there is no magic
distance threshold). The mirror question — what share of the candidate lies
near your task data at all — is its *relevance*.

High coverage + high relevance: the candidate is about your task.
High relevance, low coverage: on-topic but missing whole regions.
Low relevance: mostly off-topic for this task, whatever its quality.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from datagate.analyzers.base import Analyzer
from datagate.context import AnalysisContext
from datagate.scoring import pct
from datagate.types import ScoredEvidence

K = 5


class CoverageAnalyzer(Analyzer):
    name = "coverage"
    family = "fit"
    dimension = "Gap coverage"
    needs_reference = True

    def run(self, ctx: AnalysisContext) -> ScoredEvidence:
        joint = ctx.joint()
        if joint is None:
            return self.skip_no_reference()
        Ec, Er = joint.emb_c, joint.emb_r
        if len(Ec) < K + 1 or len(Er) < K + 1 or Ec.shape[1] < 1:
            return self.skip("too few rows to compare neighborhoods")

        # Radius: how close a reference row's own neighbors typically are.
        own = NearestNeighbors(n_neighbors=K + 1).fit(Er).kneighbors(Er)[0][:, 1:].mean(axis=1)
        radius = max(float(np.median(own)), 1e-9)

        to_cand = NearestNeighbors(n_neighbors=K).fit(Ec).kneighbors(Er)[0].mean(axis=1)
        coverage = float((to_cand <= radius).mean())

        to_ref = NearestNeighbors(n_neighbors=1).fit(Er).kneighbors(Ec)[0][:, 0]
        relevance = float((to_ref <= radius).mean())

        uncovered = joint.ref_idx[np.argsort(-to_cand)][: int((to_cand > radius).sum())]
        recs = []
        if coverage < 0.5:
            recs.append(f"The candidate reaches only {pct(coverage, 0)} of your task data's "
                        "neighborhoods — it is missing whole regions your model needs.")
        if relevance < 0.5:
            recs.append(f"Only {pct(relevance, 0)} of candidate rows resemble your task "
                        "data; filter the rest out or expect little benefit from them.")

        return self.evidence(
            score=0.6 * coverage + 0.4 * relevance,
            confidence=0.7 if len(Er) >= 50 else 0.5,
            summary=(f"candidate covers {pct(coverage, 0)} of your task data's regions; "
                     f"{pct(relevance, 0)} of candidate rows are relevant to it"),
            metrics={"coverage": coverage, "relevance": relevance, "radius": radius,
                     "reference_rows": int(len(Er)), "candidate_rows": int(len(Ec)),
                     "uncovered_reference_rows": [int(i) for i in uncovered[:25]]},
            recommendations=recs,
        )
