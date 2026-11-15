"""Exports: the cleaned dataset and the quarantine list.

A gate that only points at problems leaves the work undone. The cleaned export
is the candidate with quarantined rows removed (optionally with PII masked in
place instead of dropped); the quarantine export is those rows with the reason
each was pulled, for review.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from datagate.analyzers.security.pii import redact_text
from datagate.ingest.loader import Dataset
from datagate.types import Flag, Severity


def _is_pii_only(flag: Flag) -> bool:
    return all(r.strip().startswith("pii:") for r in flag.reason.split(";"))


def quarantine_frame(dataset: Dataset, quarantine: list[Flag]) -> pl.DataFrame:
    """Quarantined rows with their index, severity and reasons, worst first."""
    if not quarantine:
        return pl.DataFrame({"row_index": [], "severity": [], "reasons": [], "detail": []})
    meta = pl.DataFrame({
        "row_index": [f.row_index for f in quarantine],
        "severity": [f.severity.value for f in quarantine],
        "reasons": [f.reason for f in quarantine],
        "detail": [f.detail or "" for f in quarantine],
    }, schema={"row_index": pl.UInt32, "severity": pl.String,
               "reasons": pl.String, "detail": pl.String})
    rows = dataset.df.with_row_index("row_index")
    return meta.join(rows, on="row_index", how="left")


def clean_frame(dataset: Dataset, quarantine: list[Flag],
                min_severity: Severity = Severity.WARNING,
                redact_pii: bool = False) -> pl.DataFrame:
    """The dataset minus quarantined rows.

    With ``redact_pii``, rows quarantined *only* for PII are kept and their
    PII/secret spans replaced with placeholders; everything else is dropped.
    """
    drop: set[int] = set()
    redact: set[int] = set()
    for flag in quarantine:
        if flag.severity.rank < min_severity.rank:
            continue
        if redact_pii and _is_pii_only(flag):
            redact.add(flag.row_index)
        else:
            drop.add(flag.row_index)

    df = dataset.df.with_row_index("_idx")
    if redact:
        mask = pl.col("_idx").is_in(list(redact))
        df = df.with_columns([
            pl.when(mask)
            .then(pl.col(c).map_elements(redact_text, return_dtype=pl.String,
                                         skip_nulls=True))
            .otherwise(pl.col(c)).alias(c)
            for c in dataset.string_columns()
        ])
    if drop:
        df = df.filter(~pl.col("_idx").is_in(list(drop)))
    return df.drop("_idx")


def write_frame(df: pl.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        df.write_parquet(p)
    elif suffix in (".jsonl", ".ndjson"):
        df.write_ndjson(p)
    elif suffix == ".json":
        df.write_json(p)
    elif suffix == ".tsv":
        df.write_csv(p, separator="\t")
    else:
        df.write_csv(p)
    return p
