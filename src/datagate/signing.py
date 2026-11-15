"""Reproducible digests and signatures for reports.

- ``report_digest`` — SHA-256 over the deterministic part of a report
  (everything except timings, the digest itself and the signature). Same
  inputs, seed and policy -> same digest, on any run.
- ``sign_report`` / ``verify_report`` — Ed25519 over that digest, so two
  parties can check a report was not altered after it was issued.

A signature made with the key embedded in the report proves *integrity* only.
To prove *origin*, compare against a public key you obtained out of band
(``datagate keys show``) — ``verify_report`` reports the two separately.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from datagate import paths
from datagate.types import Report

_UNSIGNED_FIELDS = {"runtime", "digest", "signature"}
KEY_FILE = "report_signing_ed25519.pem"


def _normalize(value: Any) -> Any:
    """Make the canonical form survive a round trip through any JSON implementation.

    JavaScript writes the float 1.0 as ``1``; Python then reads it back as an
    int. Without this, a report that merely passed through a browser would no
    longer match its own digest.
    """
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    return value


def _canonical(payload: dict[str, Any]) -> bytes:
    body = _normalize({k: v for k, v in payload.items() if k not in _UNSIGNED_FIELDS})
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf8")


def payload_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def report_digest(report: Report) -> str:
    return payload_digest(report.model_dump(mode="json"))


def load_or_create_key(directory: Optional[Path] = None) -> Ed25519PrivateKey:
    path = (directory or paths.keys_dir()) / KEY_FILE
    if path.exists():
        key = serialization.load_pem_private_key(path.read_bytes(), password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise ValueError(f"{path} is not an Ed25519 private key")
        return key
    key = Ed25519PrivateKey.generate()
    pem = key.private_bytes(serialization.Encoding.PEM,
                            serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption())
    # Create with owner-only permissions from the start.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(pem)
    return key


def public_key_b64(key: Optional[Ed25519PrivateKey] = None) -> str:
    key = key or load_or_create_key()
    raw = key.public_key().public_bytes(serialization.Encoding.Raw,
                                        serialization.PublicFormat.Raw)
    return base64.b64encode(raw).decode("ascii")


def sign_report(report: Report, key: Optional[Ed25519PrivateKey] = None) -> Report:
    key = key or load_or_create_key()
    digest = report_digest(report)
    signature = key.sign(bytes.fromhex(digest))
    report.digest = digest
    report.signature = {
        "algorithm": "ed25519",
        "public_key": public_key_b64(key),
        "signature": base64.b64encode(signature).decode("ascii"),
    }
    return report


@dataclass
class Verification:
    digest_ok: bool          # content matches the recorded digest
    signature_ok: bool       # signature matches the digest and embedded key
    trusted: Optional[bool]  # embedded key equals the key you trust (None: not checked)
    detail: str

    @property
    def ok(self) -> bool:
        return self.digest_ok and self.signature_ok and self.trusted is not False


def verify_report(payload: dict[str, Any],
                  trusted_public_key: Optional[str] = None) -> Verification:
    recorded = payload.get("digest")
    actual = payload_digest(payload)
    if not recorded or recorded != actual:
        return Verification(False, False, None,
                            "content does not match the recorded digest — the report "
                            "was modified or was never finalized")
    sig = payload.get("signature")
    if not sig:
        return Verification(True, False, None,
                            "digest is intact, but the report is not signed")
    try:
        public = Ed25519PublicKey.from_public_bytes(base64.b64decode(sig["public_key"]))
        public.verify(base64.b64decode(sig["signature"]), bytes.fromhex(actual))
    except (InvalidSignature, KeyError, ValueError):
        return Verification(True, False, None, "signature does not verify")

    if trusted_public_key is None:
        return Verification(True, True, None,
                            "signature is valid for the key embedded in the report "
                            "(integrity only — origin not checked)")
    trusted = sig["public_key"].strip() == trusted_public_key.strip()
    return Verification(True, True, trusted,
                        "signature is valid and made by the trusted key" if trusted
                        else "signature is valid but was NOT made by the trusted key")
