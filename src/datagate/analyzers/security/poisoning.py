"""Poisoning & backdoor heuristics. Best-effort by design — say so in the report.

Two model-agnostic signals, neither of which needs a training run on the
suspect data:

1. **Class-geometry outliers.** Within each class, rows far from the class
   centroid, or extreme along the class's top singular direction (the
   "spectral signature" of Tran et al., 2018), are unlike their own class.
   Label-flipped and off-distribution injected rows look like this.

2. **Backdoor trigger tokens** (text). A dirty-label backdoor stamps a trigger
   token onto rows of other classes and relabels them. Such a token is almost
   perfectly predictive of one label — but so is any honest strong word. The
   discriminating test is counterfactual: *mask the token and re-read the row*.
   For an honest word the rest of the text still supports the label; for a
   trigger the rest of the text argues for a different class.

Limits, stated plainly: clean-label backdoors (trigger added to rows that
already carry the target label) and feature-space attacks on tabular data are
not detectable this way. A clean result lowers risk; it does not prove safety.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

from datagate.analyzers.base import Analyzer
from datagate.context import AnalysisContext, View
from datagate.scoring import pct, plural, risk_score, robust_z
from datagate.types import Flag, ScoredEvidence, Severity

OUTLIER_Z = 6.0
SPECTRAL_DIRECTIONS = 8
MIN_CLASS_ROWS = 30


MAX_INJECTED_SHARE = 0.06      # a tight cluster up to this share of a class is suspect
MAX_SUBPOPULATION_SHARE = 0.30


def _classify_axis(z: np.ndarray) -> str | None:
    """What do the extreme rows along this axis look like?

    - ``"outliers"``: a handful of isolated extreme points.
    - ``"cluster"``: a small, *uniformly* extreme group — nearly every unusual
      row is very unusual. This is what injected rows look like.
    - ``"subpopulation"``: the same tight shape but a sizeable share of the class.
      On real data that is boilerplate (one author's signature, a template), so
      it is reported, not quarantined.
    - ``None``: a spread-out natural group (a topic, a common phrase) whose tail
      merely crosses the threshold. Ignored.
    """
    moderate = z > OUTLIER_Z / 2
    share = moderate.mean()
    if share == 0 or not (z > OUTLIER_Z).any():
        return None
    if share <= 0.01:
        return "outliers"
    if (z > OUTLIER_Z).sum() / moderate.sum() < 0.5:
        return None
    if share <= MAX_INJECTED_SHARE:
        return "cluster"
    return "subpopulation" if share <= MAX_SUBPOPULATION_SHARE else None
MAX_TRIGGER_CANDIDATES = 400


class PoisoningAnalyzer(Analyzer):
    name = "poisoning"
    family = "security"
    dimension = "Poisoning & backdoors"

    def run(self, ctx: AnalysisContext) -> ScoredEvidence:
        if ctx.task_kind != "classification":
            return self.skip("poisoning heuristics need a classification target")
        view = ctx.view()
        if view.y is None or view.X.shape[1] == 0:
            return self.skip("no usable feature columns")
        pos = np.where(view.labeled)[0]
        if len(pos) < 100:
            return self.skip("too few labeled rows (<100)")

        y = view.y[pos].astype(str)
        emb = ctx.embeddings()[pos]

        flags: dict[int, Flag] = {}
        outliers, subpopulations = self._geometry_outliers(emb, y)
        for local, detail in outliers:
            row = int(view.idx[pos[local]])
            flags[row] = Flag(row_index=row, reason="class_outlier",
                              severity=Severity.WARNING, detail=detail)
        n_outliers = len(flags)

        triggers = self._trigger_tokens(view, pos, y) if view.featurizer.has_text else []
        n_trigger_rows = 0
        for trig in triggers:
            for local in trig["rows"]:
                row = int(view.idx[pos[local]])
                flags[row] = Flag(
                    row_index=row, reason="backdoor_trigger", severity=Severity.CRITICAL,
                    detail=(f"token {' '.join(trig['tokens'])!r} forces label "
                            f"{trig['label']!r}; the rest of the text disagrees"))
                n_trigger_rows += 1

        n = len(pos)
        score = risk_score(n_outliers / n, ok=0.005, bad=0.05)
        blockers, recommendations = [], []
        strong = [t for t in triggers if t["support"] >= max(10, int(0.005 * n))]
        if triggers:
            score = min(score, 0.1 if strong else 0.4)
            names = "; ".join(f"{' '.join(t['tokens'])!r} → {t['label']!r} "
                              f"({t['support']} rows)" for t in triggers[:5])
            recommendations.append(
                f"Inspect rows containing the suspected trigger "
                f"{'token' if len(triggers) == 1 else 'tokens'} {names}. "
                "If you did not put them there, treat the dataset as poisoned.")
        if strong:
            blockers.append(
                f"{plural(len(strong), 'suspected backdoor trigger')}: a token that forces a label "
                f"the rest of the text contradicts ({n_trigger_rows:,} rows).")
        if n_outliers:
            recommendations.append(
                f"Review {n_outliers:,} rows that sit far from their own class — typical "
                "of label-flipped or injected off-distribution samples.")
        for sub in subpopulations[:3]:
            recommendations.append(
                f"{sub['rows']:,} {sub['label']!r} rows ({pct(sub['share_of_class'], 0)} of "
                "the class) form one tight, distinct group — usually boilerplate such as a "
                "repeated signature or template. A model may latch onto it; consider "
                "stripping it.")

        sampled = len(view.idx) < ctx.dataset.n_rows
        ordered = sorted(flags.values(), key=lambda f: (-f.severity.rank, f.row_index))
        return self.evidence(
            score=score,
            confidence=0.75 if triggers else 0.5,
            summary=(f"{n_outliers:,} class-geometry outliers ({pct(n_outliers / n)} of "
                     f"{n:,} analyzed rows)"
                     + (f" and {plural(len(triggers), 'suspected backdoor trigger')}" if triggers
                        else "; no trigger tokens found")
                     + " — heuristic, cannot rule out stealthy attacks"),
            flags=ordered,
            metrics={
                "analyzed_rows": n, "class_outliers": n_outliers,
                "trigger_rows": n_trigger_rows, "sampled": sampled,
                "subpopulations": subpopulations,
                "triggers": [{"tokens": t["tokens"], "label": t["label"],
                              "support": t["support"], "probability_drop": t["drop"]}
                             for t in triggers],
            },
            blockers=blockers, recommendations=recommendations,
        )

    @staticmethod
    def _geometry_outliers(emb: np.ndarray, y: np.ndarray
                           ) -> tuple[list[tuple[int, str]], list[dict]]:
        """Returns (suspect rows, large tight sub-populations worth a look)."""
        out: list[tuple[int, str]] = []
        subpopulations: list[dict] = []
        for cls in np.unique(y):
            members = np.where(y == cls)[0]
            if len(members) < MIN_CLASS_ROWS:
                continue
            centered = emb[members] - emb[members].mean(axis=0)
            axes = [robust_z(np.linalg.norm(centered, axis=1))]
            if centered.shape[1] >= 2:
                # A small injected cluster rarely owns the *top* direction, but it
                # does own one of the leading ones — where nearly every honest row
                # projects to ~0 and the cluster sits far out.
                _, _, vt = np.linalg.svd(centered, full_matrices=False)
                axes += [np.abs(robust_z(centered @ d)) for d in vt[:SPECTRAL_DIRECTIONS]]

            z = np.zeros(len(members))
            for z_axis in axes:
                kind = _classify_axis(z_axis)
                if kind in ("outliers", "cluster"):
                    z = np.maximum(z, z_axis)
                elif kind == "subpopulation":
                    subpopulations.append({
                        "label": str(cls),
                        "rows": int((z_axis > OUTLIER_Z).sum()),
                        "share_of_class": round(float((z_axis > OUTLIER_Z).mean()), 3)})
            for i in np.where(z > OUTLIER_Z)[0]:
                out.append((int(members[i]),
                            f"unlike other {str(cls)!r} rows (robust z={z[i]:.1f})"))
        return out, subpopulations

    @staticmethod
    def _trigger_tokens(view: View, pos: np.ndarray, y: np.ndarray) -> list[dict]:
        n_text = view.featurizer.n_text_features
        if n_text == 0:
            return []
        X = view.X[pos].tocsr()
        n = X.shape[0]
        present = (X[:, :n_text] > 0).astype(np.float32).tocsc()
        doc_freq = np.asarray(present.sum(axis=0)).ravel()

        min_support = max(8, int(0.003 * n))
        cand = np.where((doc_freq >= min_support) & (doc_freq <= 0.2 * n))[0]
        if len(cand) == 0:
            return []

        classes, y_idx = np.unique(y, return_inverse=True)
        k = len(classes)
        prior = np.bincount(y_idx, minlength=k) / n
        onehot = sp.csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n), y_idx)),
                               shape=(n, k))
        counts = np.asarray((present[:, cand].T @ onehot).todense())
        top = counts.argmax(axis=1)
        purity = counts.max(axis=1) / doc_freq[cand]
        keep = (purity >= 0.9) & (purity - prior[top] >= 0.2)
        cand, top = cand[keep], top[keep]
        if len(cand) == 0:
            return []
        if len(cand) > MAX_TRIGGER_CANDIDATES:
            order = np.argsort(-doc_freq[cand])[:MAX_TRIGGER_CANDIDATES]
            cand, top = cand[order], top[order]

        model = LogisticRegression(C=5.0, max_iter=300).fit(X, y)
        names = view.featurizer.feature_names()

        # Tokens that (almost) always co-occur form one multi-token trigger.
        sub = present[:, cand]
        co = np.asarray((sub.T @ sub).todense())
        df_c = doc_freq[cand]

        found: list[dict] = []
        seen: set[tuple[int, ...]] = set()
        for a, j in enumerate(cand):
            jaccard = co[a] / np.maximum(1.0, df_c[a] + df_c - co[a])
            group = tuple(sorted(int(g) for g in cand[jaccard >= 0.9]))
            if group in seen:
                continue
            seen.add(group)

            rows = present.indices[present.indptr[j]:present.indptr[j + 1]]
            c = int(top[a])
            rows = rows[y_idx[rows] == c]
            if len(rows) < min_support:
                continue

            masked = X[rows].tocsc(copy=True)
            for g in group:
                masked.data[masked.indptr[g]:masked.indptr[g + 1]] = 0.0
            masked.eliminate_zeros()
            masked = masked.tocsr()

            p_full = model.predict_proba(X[rows])[:, c]
            proba_masked = model.predict_proba(masked)
            p_masked = proba_masked[:, c]
            flipped = proba_masked.argmax(axis=1) != c
            drop = float((p_full - p_masked).mean())
            residual = float(p_masked.mean() - prior[c])
            if drop >= 0.30 and residual <= -0.10 and flipped.mean() >= 0.6:
                found.append({
                    "tokens": [names[g] for g in group],
                    "label": str(classes[c]),
                    "rows": rows[flipped].tolist(),
                    "support": int(len(rows)),
                    "drop": round(drop, 3),
                })
        return sorted(found, key=lambda t: -t["support"])
