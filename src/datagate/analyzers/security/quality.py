"""Record-quality scan: rows that are structurally unusable.

Missing labels, mostly-empty rows, blank text and non-finite numbers are
quarantined. Extreme numeric outliers are reported but not quarantined — an
outlier is not necessarily an error, so the call stays with the user.
"""

from __future__ import annotations

import math

import polars as pl

from datagate.analyzers.base import Analyzer
from datagate.context import AnalysisContext
from datagate.ingest.loader import KIND_ID
from datagate.scoring import pct, risk_score
from datagate.types import Flag, ScoredEvidence, Severity

OUTLIER_Z = 8.0  # robust (median/MAD) z-score

_REASONS = [  # (column, reason, severity), in priority order
    ("missing_label", "missing_label", Severity.WARNING),
    ("mostly_null", "mostly_null_row", Severity.WARNING),
    ("empty_text", "empty_text", Severity.WARNING),
    ("non_finite", "non_finite_value", Severity.WARNING),
    ("outlier", "extreme_outlier", Severity.INFO),
]


class QualityAnalyzer(Analyzer):
    name = "quality"
    family = "security"
    dimension = "Record quality"

    def run(self, ctx: AnalysisContext) -> ScoredEvidence:
        ds = ctx.dataset
        df = ds.df
        n = df.height
        if n == 0:
            return self.skip("empty dataset")

        target = ctx.target
        false = pl.lit(False)

        null_threshold = max(1, math.ceil(0.5 * df.width))
        mostly_null = (pl.sum_horizontal([pl.col(c).is_null().cast(pl.Int32)
                                          for c in df.columns]) >= null_threshold)

        text_cols = ds.text_columns(exclude=[target] if target else [])
        empty_text = (pl.any_horizontal([
            (pl.col(c).str.strip_chars().str.len_chars() == 0).fill_null(False)
            for c in text_cols]) if text_cols else false)

        float_cols = [c for c, dt in df.schema.items() if dt.is_float()]
        non_finite = (pl.any_horizontal([
            (pl.col(c).is_infinite() | pl.col(c).is_nan()).fill_null(False)
            for c in float_cols]) if float_cols else false)

        outlier_exprs, outlier_columns = [], []
        for c in ds.numeric_columns(exclude=[target] if target else []):
            s = df[c].drop_nulls().cast(pl.Float64)
            s = s.filter(s.is_finite())
            if s.len() < 30:
                continue
            median = s.median()
            mad = (s - median).abs().median()
            if not mad:
                continue
            outlier_exprs.append(
                (((pl.col(c).cast(pl.Float64) - median).abs() / (1.4826 * mad)) > OUTLIER_Z)
                .fill_null(False))
            outlier_columns.append(c)
        outlier = pl.any_horizontal(outlier_exprs) if outlier_exprs else false

        missing_label = pl.col(target).is_null() if target else false

        marked = (
            df.with_row_index("_idx")
            .select("_idx", missing_label.alias("missing_label"),
                    mostly_null.alias("mostly_null"), empty_text.alias("empty_text"),
                    non_finite.alias("non_finite"), outlier.alias("outlier"))
            .filter(pl.any_horizontal(pl.exclude("_idx")))
        )

        counts = {col: int(marked[col].sum()) for col, _, _ in _REASONS}
        flags: list[Flag] = []
        n_bad = 0
        for row in marked.iter_rows(named=True):
            for col, reason, severity in _REASONS:
                if row[col]:
                    flags.append(Flag(row_index=int(row["_idx"]), reason=reason,
                                      severity=severity))
                    n_bad += severity is not Severity.INFO
                    break
        flags.sort(key=lambda f: (-f.severity.rank, f.row_index))

        # Column-level observations: not row flags, but worth acting on.
        notes: list[str] = []
        kinds = ds.kinds()
        for c in df.columns:
            if c == target:
                continue
            s = df[c]
            if n > 1 and s.n_unique() == 1:
                notes.append(f"column {c!r} is constant")
            elif s.null_count() / n > 0.5:
                notes.append(f"column {c!r} is {pct(s.null_count() / n, 0)} empty")
            elif kinds.get(c) == KIND_ID:
                notes.append(f"column {c!r} looks like an identifier and is ignored")

        rate = n_bad / n
        recommendations = []
        if n_bad:
            recommendations.append(
                f"Drop or repair the {n_bad:,} structurally unusable rows "
                "(missing label, mostly empty, blank text or non-finite values).")
        if counts["outlier"]:
            recommendations.append(
                f"Inspect {counts['outlier']:,} rows with extreme values in "
                f"{', '.join(repr(c) for c in outlier_columns[:4])} — "
                "keep them if they are real, fix them if they are entry errors.")

        parts = [f"{counts[col]:,} {label}" for col, label in [
            ("missing_label", "missing labels"), ("mostly_null", "mostly-empty rows"),
            ("empty_text", "blank texts"), ("non_finite", "non-finite values"),
            ("outlier", "extreme outliers")] if counts[col]]
        return self.evidence(
            score=risk_score(rate, ok=0.01, bad=0.30),
            confidence=0.9,
            summary=(f"{n_bad:,} of {n:,} rows are structurally unusable ({pct(rate)}): "
                     + ", ".join(parts) if parts
                     else f"all {n:,} rows are structurally sound"),
            flags=flags,
            metrics={"counts": counts, "unusable_rows": n_bad, "unusable_rate": rate,
                     "column_notes": notes},
            recommendations=recommendations,
        )
