"""Known-dataset registry: has this data been seen before?

Each registered dataset is stored as a compact *bottom-k sketch* of its row
hashes — a few kilobytes, no raw rows. Two sketches are enough to estimate what
fraction of one dataset's rows also appear in the other, which is what
benchmark-contamination and re-upload detection need. Row hashing is
column-order-insensitive, so a re-export with shuffled columns still matches.

This is the provenance index the marketplace mode builds on: it can say
"≈92% of these rows match public dataset X", never "ownership proven".
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import polars as pl

from datagate import paths
from datagate.ingest.loader import Dataset

SKETCH_SIZE = 4096
MAX_SKETCH_ROWS = 2_000_000
VALID_KINDS = ("public", "benchmark", "eval", "internal", "upload")


@dataclass
class Entry:
    name: str
    kind: str
    fingerprint: str
    rows: int
    sketch: list[int]


@dataclass
class Match:
    name: str
    kind: str
    containment: float   # share of the candidate's rows found in the known dataset
    exact: bool          # byte-identical file
    coverage: float = 0.0  # share of the known dataset's rows found in the candidate


def registry_for_user(owner_id: str, local_id: str = "local") -> "Registry":
    """The registry one account sees: its own uploads plus the shared entries.

    The implicit local user keeps using the shared registry directly, exactly as
    before accounts existed.
    """
    if owner_id == local_id:
        return Registry()
    return Registry(root=paths.registry_dir() / "users" / owner_id,
                    shared=[paths.registry_dir()])


def row_sketch(dataset: Dataset) -> list[int]:
    """The SKETCH_SIZE smallest distinct 64-bit row hashes, sorted."""
    if "sketch" in dataset.cache:
        return dataset.cache["sketch"]
    df = dataset.df.head(MAX_SKETCH_ROWS)
    if df.height == 0:
        return []
    keys = df.select(
        pl.concat_list([
            pl.col(c).cast(pl.String, strict=False).fill_null("")
            .str.to_lowercase().str.strip_chars()
            for c in df.columns
        ]).list.sort().list.join("\x1f").alias("k")
    )["k"].to_list()
    hashes = {
        int.from_bytes(hashlib.blake2b(k.encode("utf8"), digest_size=8).digest(), "big")
        for k in keys if k.strip("\x1f")
    }
    sketch = sorted(hashes)[:SKETCH_SIZE]
    dataset.cache["sketch"] = sketch
    return sketch


def containment(a: list[int], b: list[int]) -> float:
    """Estimated share of A's rows that also appear in B.

    Both sketches are complete below the smaller of their maxima, so restricting
    to that range gives an unbiased comparison.
    """
    if not a or not b:
        return 0.0
    tau = min(a[-1], b[-1])
    a_low = {h for h in a if h <= tau}
    if not a_low:
        return 0.0
    b_low = {h for h in b if h <= tau}
    return len(a_low & b_low) / len(a_low)


class Registry:
    """Sketches live under `root`. `shared` roots are read-only extras that `entries`
    and `match` also see — how one user's uploads are checked against the shared
    benchmark entries without ever seeing another user's uploads."""

    def __init__(self, root: Optional[Path] = None,
                 shared: Sequence[Path] = ()) -> None:
        self.root = Path(root) if root else paths.registry_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        self.shared = [Path(p) for p in shared]

    def _path(self, fingerprint: str) -> Path:
        return self.root / f"{fingerprint[:32]}.json"

    def add(self, dataset: Dataset, name: Optional[str] = None,
            kind: str = "internal") -> Entry:
        if kind not in VALID_KINDS:
            raise ValueError(f"kind must be one of {VALID_KINDS}")
        entry = Entry(name=name or dataset.name, kind=kind,
                      fingerprint=dataset.fingerprint, rows=dataset.n_rows,
                      sketch=row_sketch(dataset))
        self._path(entry.fingerprint).write_text(json.dumps(asdict(entry)))
        return entry

    def _read(self, root: Path) -> list[Entry]:
        out = []
        for p in sorted(root.glob("*.json")):
            try:
                out.append(Entry(**json.loads(p.read_text())))
            except (ValueError, TypeError):
                continue  # ignore a corrupt entry rather than fail every analysis
        return out

    def entries(self) -> list[Entry]:
        own = self._read(self.root)
        seen = {e.fingerprint for e in own}
        extra = [e for root in self.shared for e in self._read(root)
                 if e.fingerprint not in seen]
        return own + extra

    def remove(self, name_or_fingerprint: str) -> int:
        removed = 0
        for entry in self._read(self.root):
            if name_or_fingerprint in (entry.name, entry.fingerprint,
                                       entry.fingerprint[:12]):
                self._path(entry.fingerprint).unlink(missing_ok=True)
                removed += 1
        return removed

    def match(self, dataset: Dataset, ignore_fingerprints: frozenset[str] = frozenset(),
              min_containment: float = 0.05) -> list[Match]:
        sketch = row_sketch(dataset)
        matches = []
        for entry in self.entries():
            if entry.fingerprint in ignore_fingerprints:
                continue
            exact = entry.fingerprint == dataset.fingerprint
            c = 1.0 if exact else containment(sketch, entry.sketch)
            # The reverse direction matters just as much: a small eval set fully
            # swallowed by a large candidate is a tiny share of the candidate.
            cov = 1.0 if exact else containment(entry.sketch, sketch)
            if max(c, cov) >= min_containment:
                matches.append(Match(entry.name, entry.kind, c, exact, cov))
        return sorted(matches, key=lambda m: -max(m.containment, m.coverage))
