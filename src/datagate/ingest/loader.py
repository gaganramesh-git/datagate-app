"""Multi-format dataset loading -> polars DataFrame, plus column-kind inference.

Column *kinds* matter more than dtypes for analysis: a string column can be
free text (TF-IDF it), a category (one-hot it) or an identifier (ignore it),
and treating one as another quietly wrecks every downstream check.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import polars as pl

SUPPORTED_SUFFIXES = {".csv", ".tsv", ".parquet", ".json", ".jsonl", ".ndjson", ".txt"}

KIND_TEXT = "text"
KIND_CATEGORICAL = "categorical"
KIND_NUMERIC = "numeric"
KIND_ID = "id"
KIND_OTHER = "other"


@dataclass
class Dataset:
    """A loaded dataset plus the lightweight metadata analyzers rely on."""

    path: str
    df: pl.DataFrame
    format: str
    name: str = ""
    fingerprint: str = ""
    profile: dict[str, Any] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def n_rows(self) -> int:
        return self.df.height

    @property
    def n_cols(self) -> int:
        return self.df.width

    def kinds(self) -> dict[str, str]:
        if "kinds" not in self.cache:
            self.cache["kinds"] = infer_kinds(self.df)
        return self.cache["kinds"]

    def columns_of(self, kind: str, exclude: Iterable[str] = ()) -> list[str]:
        skip = set(exclude)
        return [c for c, k in self.kinds().items() if k == kind and c not in skip]

    def string_columns(self) -> list[str]:
        """Every string-typed column, whatever its kind — what content scanners read."""
        return [c for c, dt in self.df.schema.items() if dt == pl.String]

    def text_columns(self, exclude: Iterable[str] = ()) -> list[str]:
        return self.columns_of(KIND_TEXT, exclude)

    def numeric_columns(self, exclude: Iterable[str] = ()) -> list[str]:
        return self.columns_of(KIND_NUMERIC, exclude)

    def categorical_columns(self, exclude: Iterable[str] = ()) -> list[str]:
        return self.columns_of(KIND_CATEGORICAL, exclude)


def infer_kinds(df: pl.DataFrame) -> dict[str, str]:
    n = df.height
    kinds: dict[str, str] = {}
    for name, dtype in df.schema.items():
        s = df[name]
        n_unique = int(s.drop_nulls().n_unique()) if n else 0
        if dtype == pl.Boolean:
            kinds[name] = KIND_CATEGORICAL
        elif dtype.is_numeric():
            # Integers that never repeat are row identifiers, not measurements.
            if dtype.is_integer() and n >= 20 and n_unique == n:
                kinds[name] = KIND_ID
            else:
                kinds[name] = KIND_NUMERIC
        elif dtype == pl.String:
            kinds[name] = _string_kind(s, n, n_unique)
        else:
            kinds[name] = KIND_OTHER
    return kinds


def _string_kind(s: pl.Series, n: int, n_unique: int) -> str:
    if n == 0 or n_unique == 0:
        return KIND_OTHER
    non_null = s.drop_nulls()
    mean_len = float(non_null.str.len_chars().mean() or 0.0)
    space_frac = float(non_null.str.contains(" ", literal=True).mean() or 0.0)
    unique_ratio = n_unique / n
    looks_like_prose = mean_len > 40 or space_frac >= 0.3
    if n_unique <= 50 and unique_ratio <= 0.5 and not (looks_like_prose and unique_ratio > 0.2):
        return KIND_CATEGORICAL
    if not looks_like_prose:
        if unique_ratio >= 0.95:
            return KIND_ID
        if unique_ratio <= 0.05:
            return KIND_CATEGORICAL
    return KIND_TEXT


def file_fingerprint(path: str | Path) -> str:
    """SHA-256 of the file bytes — the dataset's identity for reproducibility."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: str | Path, name: Optional[str] = None) -> Dataset:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    suffix = p.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported format {suffix!r}. Supported: {sorted(SUPPORTED_SUFFIXES)}")

    if suffix in (".csv", ".tsv"):
        df = pl.read_csv(
            p,
            separator="\t" if suffix == ".tsv" else ",",
            infer_schema_length=10_000,
            ignore_errors=True,
            truncate_ragged_lines=True,
            encoding="utf8-lossy",
        )
    elif suffix == ".parquet":
        df = pl.read_parquet(p)
    elif suffix == ".json":
        df = pl.read_json(p)
    elif suffix == ".txt":
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        df = pl.DataFrame({"text": [ln for ln in lines if ln.strip()]})
    else:  # .jsonl / .ndjson
        df = pl.read_ndjson(p)

    if df.width == 0:
        raise ValueError("Dataset has no columns")

    return Dataset(
        path=str(p),
        df=df,
        format=suffix.lstrip("."),
        name=name or p.name,
        fingerprint=file_fingerprint(p),
    )
