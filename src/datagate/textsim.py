"""Text normalization and MinHash helpers shared by the overlap checks."""

from __future__ import annotations

from typing import Iterable, Optional

import polars as pl
from datasketch import MinHash, MinHashLSH

NUM_PERM = 64
NEAR_DUP_THRESHOLD = 0.85


def normalized_expr(cols: list[str]) -> pl.Expr:
    """Lowercased, punctuation- and whitespace-insensitive content key for a row."""
    joined = pl.concat_str(
        [pl.col(c).cast(pl.String, strict=False).fill_null("") for c in cols],
        separator=" ",
    )
    return (joined.str.to_lowercase()
            .str.replace_all(r"[^\p{L}\p{N}]+", " ")
            .str.strip_chars())


def shingles(text: str, k: int = 3) -> list[bytes]:
    tokens = text.split()
    if len(tokens) < k:
        return [" ".join(tokens).encode("utf8")] if tokens else []
    return [" ".join(tokens[i:i + k]).encode("utf8") for i in range(len(tokens) - k + 1)]


def minhash(text: str) -> Optional[MinHash]:
    """MinHash of a normalized text, or None when it is too short to compare."""
    grams = shingles(text)
    if len(grams) < 2:
        return None
    mh = MinHash(num_perm=NUM_PERM)
    mh.update_batch(grams)
    return mh


def build_lsh(items: Iterable[tuple[int, str]],
              threshold: float = NEAR_DUP_THRESHOLD) -> tuple[MinHashLSH, dict[int, MinHash]]:
    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    sigs: dict[int, MinHash] = {}
    for key, text in items:
        mh = minhash(text)
        if mh is None:
            continue
        sigs[key] = mh
        lsh.insert(key, mh)
    return lsh, sigs
