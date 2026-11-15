"""One shared feature space for every analyzer.

Text columns become TF-IDF, numeric columns are standardized, categorical
columns are one-hot encoded, identifier-like columns are ignored. Analyzers
never featurize on their own, so a candidate dataset and the user's reference
data are always compared in the *same* space.

The dense embedding (`embed`) is where a neural encoder (sentence-transformers,
CLIP) would slot in; the default is LSA over the shared matrix, which needs no
model download and is fully deterministic.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from datagate.ingest.loader import Dataset

MAX_TEXT_FEATURES = 20_000
MAX_CATEGORIES = 50


@dataclass
class ColumnPlan:
    text: list[str] = field(default_factory=list)
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.text or self.numeric or self.categorical)


def plan_for(ds: Dataset, exclude: Iterable[str] = ()) -> ColumnPlan:
    skip = set(exclude)
    return ColumnPlan(
        text=ds.text_columns(skip),
        numeric=ds.numeric_columns(skip),
        categorical=ds.categorical_columns(skip),
    )


def concat_text(df: pl.DataFrame, cols: list[str]) -> list[str]:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return [""] * df.height
    return df.select(
        pl.concat_str(
            [pl.col(c).cast(pl.String, strict=False).fill_null("") for c in cols],
            separator=" ",
        ).alias("_t")
    )["_t"].to_list()


def numeric_block(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    """Float matrix with NaN for missing/unparseable/non-finite values."""
    if not cols:
        return np.zeros((df.height, 0))
    block = df.select(
        [pl.col(c).cast(pl.Float64, strict=False) for c in cols]
    ).to_numpy().astype(float, copy=True)
    block[~np.isfinite(block)] = np.nan
    return block


class Featurizer:
    """Fit once on one or more frames, transform any frame into the shared space."""

    def __init__(self, numeric: list[str], categorical: list[str],
                 max_text_features: int = MAX_TEXT_FEATURES) -> None:
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.max_text_features = max_text_features
        self.tfidf: Optional[TfidfVectorizer] = None
        self.num_mean = np.zeros(0)
        self.num_std = np.ones(0)
        self.cat_values: dict[str, list[str]] = {}

    @property
    def has_text(self) -> bool:
        return self.tfidf is not None

    def fit(self, frames: list[tuple[pl.DataFrame, list[str]]]) -> "Featurizer":
        """``frames`` is a list of (dataframe, that frame's text columns)."""
        docs: list[str] = []
        for df, text_cols in frames:
            if text_cols:
                docs.extend(concat_text(df, text_cols))
        if any(d.strip() for d in docs):
            try:
                self.tfidf = TfidfVectorizer(
                    max_features=self.max_text_features, sublinear_tf=True,
                    dtype=np.float32,
                ).fit(docs)
            except ValueError:  # nothing but stop-length tokens
                self.tfidf = None

        # Structured columns must exist in every frame to be comparable.
        self.numeric = [c for c in self.numeric if all(c in df.columns for df, _ in frames)]
        self.categorical = [c for c in self.categorical
                            if all(c in df.columns for df, _ in frames)]

        if self.numeric:
            stacked = np.vstack([numeric_block(df, self.numeric) for df, _ in frames])
            with np.errstate(all="ignore"):
                mean = np.nanmean(stacked, axis=0)
                std = np.nanstd(stacked, axis=0)
            self.num_mean = np.where(np.isfinite(mean), mean, 0.0)
            self.num_std = np.where(np.isfinite(std) & (std > 0), std, 1.0)

        for col in self.categorical:
            counts: Counter[str] = Counter()
            for df, _ in frames:
                counts.update(v for v in df[col].cast(pl.String, strict=False).to_list()
                              if v is not None)
            self.cat_values[col] = [v for v, _ in counts.most_common(MAX_CATEGORIES)]
        return self

    def transform(self, df: pl.DataFrame, text_cols: list[str]) -> sp.csr_matrix:
        blocks: list[sp.spmatrix] = []
        if self.tfidf is not None:
            blocks.append(self.tfidf.transform(concat_text(df, text_cols)))
        if self.numeric:
            block = (numeric_block(df, self.numeric) - self.num_mean) / self.num_std
            block = np.clip(np.nan_to_num(block, nan=0.0), -10.0, 10.0)
            blocks.append(sp.csr_matrix(block.astype(np.float32)))
        for col in self.categorical:
            blocks.append(self._one_hot(df, col))
        if not blocks:
            return sp.csr_matrix((df.height, 0), dtype=np.float32)
        return sp.hstack(blocks, format="csr", dtype=np.float32)

    def transform_dense(self, df: pl.DataFrame) -> np.ndarray:
        """Raw numeric + ordinal-coded categoricals, NaN for missing — for tree models."""
        parts = [numeric_block(df, self.numeric)]
        for col in self.categorical:
            index = {v: i for i, v in enumerate(self.cat_values.get(col, []))}
            values = df[col].cast(pl.String, strict=False).to_list()
            parts.append(np.array([[index.get(v, np.nan)] for v in values], dtype=float)
                         .reshape(df.height, 1))
        return np.hstack(parts) if parts else np.zeros((df.height, 0))

    def feature_names(self) -> list[str]:
        names: list[str] = []
        if self.tfidf is not None:
            names.extend(self.tfidf.get_feature_names_out().tolist())
        names.extend(self.numeric)
        for col in self.categorical:
            names.extend(f"{col}={v}" for v in self.cat_values.get(col, []))
        return names

    @property
    def n_text_features(self) -> int:
        return len(self.tfidf.vocabulary_) if self.tfidf is not None else 0

    def _one_hot(self, df: pl.DataFrame, col: str) -> sp.csr_matrix:
        vocab = self.cat_values.get(col, [])
        index = {v: i for i, v in enumerate(vocab)}
        values = df[col].cast(pl.String, strict=False).to_list()
        rows, cols = [], []
        for r, v in enumerate(values):
            j = index.get(v)
            if j is not None:
                rows.append(r)
                cols.append(j)
        data = np.ones(len(rows), dtype=np.float32)
        return sp.csr_matrix((data, (rows, cols)), shape=(df.height, len(vocab)))


def embed(mats: list[sp.csr_matrix], normalize: bool, dim: int = 64,
          seed: int = 7) -> list[np.ndarray]:
    """Dense, low-dimensional embeddings for geometry-based checks.

    All matrices are embedded with one projection so distances between them are
    meaningful. ``normalize`` L2-normalizes rows (cosine geometry — right for
    text); leave it off for purely tabular features.
    """
    sizes = [m.shape[0] for m in mats]
    stacked = sp.vstack(mats, format="csr")
    n, d = stacked.shape
    if n == 0 or d == 0:
        return [np.zeros((s, 1)) for s in sizes]
    k = min(dim, d - 1, n - 1)
    if d <= 128 or k < 2:
        dense = np.asarray(stacked.todense(), dtype=float)
    else:
        dense = TruncatedSVD(n_components=k, random_state=seed).fit_transform(stacked)
    if normalize:
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        dense = dense / np.where(norms > 0, norms, 1.0)
    out, start = [], 0
    for s in sizes:
        out.append(dense[start:start + s])
        start += s
    return out
