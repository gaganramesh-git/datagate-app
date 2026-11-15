"""Statistical profiling: schema inference + per-column stats.

The profile feeds the spec compiler, the UI's dataset preview and the report.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from datagate.ingest.loader import KIND_CATEGORICAL, Dataset


def profile_dataset(ds: Dataset) -> dict[str, Any]:
    df = ds.df
    kinds = ds.kinds()
    columns: dict[str, dict[str, Any]] = {}

    for name in df.columns:
        s = df[name]
        dtype = s.dtype
        null_count = int(s.null_count())
        col: dict[str, Any] = {
            "dtype": str(dtype),
            "kind": kinds[name],
            "null_count": null_count,
            "null_fraction": (null_count / df.height) if df.height else 0.0,
            "n_unique": int(s.n_unique()),
        }
        if dtype.is_numeric():
            col.update({"min": _f(s.min()), "max": _f(s.max()),
                        "mean": _f(s.mean()), "std": _f(s.std())})
        elif dtype == pl.String:
            lengths = s.str.len_chars()
            col.update({
                "mean_length": _f(lengths.mean()),
                "max_length": _f(lengths.max()),
                "empty_count": int((lengths == 0).sum()),
            })
        if kinds[name] == KIND_CATEGORICAL or col["n_unique"] <= 20:
            counts = (s.drop_nulls().cast(pl.String).value_counts(sort=True).head(10))
            value_col = counts.columns[0]
            col["top_values"] = [
                {"value": v, "count": int(c)}
                for v, c in zip(counts[value_col].to_list(), counts["count"].to_list())
            ]
        columns[name] = col

    # Candidate label columns: categories, or integers with few distinct values.
    label_candidates = [
        name for name, col in columns.items()
        if col["n_unique"] > 1 and (
            col["kind"] == KIND_CATEGORICAL
            or (df[name].dtype.is_integer() and col["n_unique"] <= 50)
        )
    ]

    profile = {
        "n_rows": df.height,
        "n_cols": df.width,
        "columns": columns,
        "label_candidates": label_candidates,
    }
    ds.profile = profile
    return profile


def _f(v: Any) -> Any:
    return float(v) if v is not None else None
