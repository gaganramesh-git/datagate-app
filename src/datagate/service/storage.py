"""Blob storage behind a small interface.

Analyzers need a real path on disk (Polars reads files), so the contract is
put -> ref and local_path(ref) -> Path. LocalStorage is the dev
implementation; an S3 backend implements local_path by downloading to a temp
file, which is exactly what a remote worker would do anyway.

Uploads are streamed to disk and hashed as they arrive, so a large file never
has to fit in memory and an oversized one is rejected mid-stream.
"""

from __future__ import annotations

import hashlib
import shutil
import uuid
from pathlib import Path
from typing import BinaryIO, Protocol

from datagate.service.config import settings

CHUNK = 1 << 20


class UploadTooLarge(ValueError):
    pass


class Storage(Protocol):
    def put_bytes(self, data: bytes, filename: str) -> str: ...
    def put_stream(self, stream: BinaryIO, filename: str,
                   max_bytes: int) -> tuple[str, int, str]: ...
    def local_path(self, ref: str) -> Path: ...
    def delete(self, ref: str) -> None: ...


def fingerprint(data: bytes) -> str:
    """Content hash — dedup today, provenance signal for marketplace mode."""
    return hashlib.sha256(data).hexdigest()


class LocalStorage:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or settings.storage_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def _blob(self, digest: str, filename: str) -> Path:
        # Content-addressed: identical uploads collapse to one blob.
        return self.root / f"{digest}{Path(filename).suffix.lower()}"

    def put_bytes(self, data: bytes, filename: str) -> str:
        blob = self._blob(fingerprint(data), filename)
        if not blob.exists():
            blob.write_bytes(data)
        return blob.name

    def put_stream(self, stream: BinaryIO, filename: str,
                   max_bytes: int) -> tuple[str, int, str]:
        """Stream to disk. Returns (ref, size_bytes, sha256)."""
        tmp = self.root / f".upload-{uuid.uuid4().hex}"
        digest = hashlib.sha256()
        size = 0
        try:
            with open(tmp, "wb") as out:
                while chunk := stream.read(CHUNK):
                    size += len(chunk)
                    if size > max_bytes:
                        raise UploadTooLarge(f"file exceeds {max_bytes} bytes")
                    digest.update(chunk)
                    out.write(chunk)
            blob = self._blob(digest.hexdigest(), filename)
            if blob.exists():
                tmp.unlink()
            else:
                tmp.replace(blob)
            return blob.name, size, digest.hexdigest()
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def local_path(self, ref: str) -> Path:
        # Refs are bare blob names; never let one walk out of the storage root.
        if not ref or Path(ref).name != ref:
            raise FileNotFoundError(f"invalid blob ref: {ref!r}")
        path = self.root / ref
        if not path.exists():
            raise FileNotFoundError(f"blob not found: {ref}")
        return path

    def delete(self, ref: str) -> None:
        if ref and Path(ref).name == ref:
            (self.root / ref).unlink(missing_ok=True)

    def clear(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)


def get_storage() -> Storage:
    if settings.s3_bucket:
        raise NotImplementedError(
            "S3 storage backend not implemented yet; unset DATAGATE_S3_BUCKET")
    return LocalStorage()
