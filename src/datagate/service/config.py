"""Service configuration, all overridable by environment variable.

Defaults are chosen so the service runs locally with zero infrastructure:
SQLite on disk, local-filesystem blob storage, in-process workers. Each has a
production counterpart (Postgres, S3, Celery/Redis) that swaps in via env var
without touching call sites.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repo-local var/ directory holds the dev database and uploaded blobs.
_DEFAULT_ROOT = Path(os.environ.get("DATAGATE_HOME", Path.cwd() / "var"))


class Settings:
    def __init__(self) -> None:
        self.home: Path = _DEFAULT_ROOT
        self.home.mkdir(parents=True, exist_ok=True)

        # SQLite for dev; set DATABASE_URL=postgresql+psycopg://... for prod.
        self.database_url: str = os.environ.get(
            "DATAGATE_DATABASE_URL", f"sqlite:///{self.home / 'datagate.db'}"
        )
        # Local filesystem blobs; set DATAGATE_S3_BUCKET to use object storage.
        self.storage_dir: Path = Path(
            os.environ.get("DATAGATE_STORAGE_DIR", self.home / "blobs")
        )
        self.s3_bucket: str | None = os.environ.get("DATAGATE_S3_BUCKET")

        self.worker_threads: int = int(os.environ.get("DATAGATE_WORKERS", "2"))
        self.max_upload_bytes: int = int(
            os.environ.get("DATAGATE_MAX_UPLOAD_BYTES", str(512 * 1024 * 1024))
        )
        # Comma-separated keys. Unset = open (local dev only).
        raw_keys = os.environ.get("DATAGATE_API_KEYS", "")
        self.api_keys: set[str] = {k.strip() for k in raw_keys.split(",") if k.strip()}
        # Shared password for the whole site (UI and API), via the browser's
        # built-in login prompt. Unset = no gate (local dev only).
        self.access_password: str = os.environ.get("DATAGATE_ACCESS_PASSWORD", "")

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys)


settings = Settings()
