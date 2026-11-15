"""Where DataGate keeps local state (signing keys, the known-dataset registry).

One root, overridable with DATAGATE_HOME, shared by the CLI and the service.
"""

from __future__ import annotations

import os
from pathlib import Path


def home() -> Path:
    root = Path(os.environ.get("DATAGATE_HOME", Path.cwd() / "var"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def keys_dir() -> Path:
    d = home() / "keys"
    d.mkdir(parents=True, exist_ok=True)
    return d


def registry_dir() -> Path:
    d = home() / "registry"
    d.mkdir(parents=True, exist_ok=True)
    return d
