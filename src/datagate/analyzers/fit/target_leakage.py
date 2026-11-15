"""Fit: target leakage — one column that gives the answer away.

A single feature that predicts the target almost perfectly is usually a
leaked copy of the label, a post-outcome field, or an ID that encodes the
class. A model trained on it looks brilliant offline and fails in production.
Legitimately decisive features exist, so this is a "verify" signal, not a
verdict: it lowers the fit score modestly and says exactly which column to check.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from datagate.analyzers.base import Analyzer
from datagate.context import AnalysisContext
from datagate.scoring import plural, sample_indices
from datagate.types import ScoredEvidence

THRESHOLD = 0.985
MAX_COLUMNS = 200


class TargetLeakageAnalyzer(Analyzer):
    name = "target_leakage"
    family = "fit"
    dimension = "Leakage risk"

    def run(self, ctx: AnalysisContext) -> ScoredEvidence:
        kind, target, ds = ctx.task_kind, ctx.target, ctx.dataset
        if kind is None or target is None:
            return self.skip("no prediction target")
        numeric = ds.numeric_columns(exclude=[target])[:MAX_COLUMNS]
        categorical = ds.categorical_columns(exclude=[target])[:MAX_COLUMNS]
        if not numeric and not categorical:
            return self.skip("no structured feature columns to check")

        df = ds.df.filter(pl.col(target).is_not_null())
        if df.height < 100:
            return self.skip("too few labeled rows (<100)")
        if df.height > ctx.max_rows:
            df = df[sample_indices(df.height, ctx.max_rows, ctx.seed)]
        n = df.height

        suspects: list[tuple[str, float]] = []
        if kind == "classification":
            y = df[target].cast(pl.String).to_numpy()
            _, counts = np.unique(y, return_counts=True)
            if len(counts) < 2 or counts.min() < 3:
                return self.skip("target classes too small to test for leakage")
            majority = counts.max() / n

            for col in categorical:
                if df[col].n_unique() > 0.5 * n:
                    continue
                grouped = df.group_by([col, target]).len()
                purity = float(grouped.group_by(col).agg(pl.col("len").max())["len"].sum()) / n
                if purity >= THRESHOLD and purity - majority >= 0.1:
                    suspects.append((col, purity))

            for col in numeric:
                x = df[col].cast(pl.Float64, strict=False).to_numpy().astype(float)
                x[~np.isfinite(x)] = np.nan
                if np.all(np.isnan(x)):
                    continue
                x = np.where(np.isnan(x), np.nanmedian(x), x).reshape(-1, 1)
                tree = DecisionTreeClassifier(max_depth=3, random_state=ctx.seed)
                acc = float(np.mean(cross_val_score(tree, x, y, cv=3,
                                                    scoring="balanced_accuracy")))
                if acc >= THRESHOLD and acc - 1.0 / len(counts) >= 0.1:
                    suspects.append((col, acc))
        else:
            y = df[target].cast(pl.Float64, strict=False).to_numpy().astype(float)
            ok = np.isfinite(y)
            if ok.sum() < 100 or np.std(y[ok]) == 0:
                return self.skip("regression target unusable for a leakage test")
            for col in numeric:
                x = df[col].cast(pl.Float64, strict=False).to_numpy().astype(float)
                both = ok & np.isfinite(x)
                if both.sum() < 100 or np.std(x[both]) == 0:
                    continue
                r = abs(float(np.corrcoef(x[both], y[both])[0, 1]))
                if r >= 0.995:
                    suspects.append((col, r))

        if not suspects:
            # Absence of leakage is not evidence of fit: give it almost no weight.
            return self.evidence(
                score=1.0, confidence=0.15,
                summary=(f"no single column gives away {target!r} "
                         f"({len(numeric) + len(categorical)} columns checked)"),
                metrics={"columns_checked": len(numeric) + len(categorical), "suspects": []},
            )

        suspects.sort(key=lambda s: -s[1])
        names = ", ".join(f"{c!r} ({v:.3f})" for c, v in suspects[:5])
        return self.evidence(
            score=0.4, confidence=0.6,
            summary=(f"{plural(len(suspects), 'column')} "
                     f"{'predicts' if len(suspects) == 1 else 'predict'} {target!r} almost perfectly on "
                     f"their own: {names} — possible target leakage"),
            metrics={"columns_checked": len(numeric) + len(categorical),
                     "suspects": [{"column": c, "score": round(v, 4)} for c, v in suspects]},
            recommendations=[
                f"Verify that {', '.join(repr(c) for c, _ in suspects[:3])} would be known "
                "at prediction time. If not, drop it — the model would learn a shortcut "
                "that does not exist in production."],
        )
