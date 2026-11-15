"""Accounts: sign up, sign in, and "who is making this request?".

Deliberately small and dependency-free.

* Passwords are hashed with scrypt (memory-hard, in the standard library) using
  OWASP's recommended cost, a random salt per user, and a constant-time compare.
* A signed-in browser holds a random 256-bit token in an HttpOnly, SameSite=Lax
  cookie. The database stores only its SHA-256, so a leaked database cannot be
  replayed as sessions.
* Failed sign-ins and sign-ups are rate-limited in memory (per client address and
  per email), which blunts password guessing without any extra service.

With ``DATAGATE_REQUIRE_LOGIN`` off, every request is the implicit local user —
the way the app has always worked on a single machine.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError

from datagate.registry import Registry, registry_for_user
from datagate.service.config import settings
from datagate.service.db import (
    LOCAL_USER_ID,
    SessionRecord,
    UserRecord,
    session_scope,
)

COOKIE = "dg_session"
MIN_PASSWORD = 8
MAX_PASSWORD = 128          # also bounds the work a single request can demand
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# scrypt cost: N=2^15, r=8, p=3 (OWASP's minimum-equivalent set) ≈ 32 MiB, ~0.2 s.
_N, _R, _P = 2 ** 15, 8, 3
_MAXMEM = 128 * 1024 * 1024


@dataclass(frozen=True)
class User:
    id: str
    email: str


LOCAL = User(LOCAL_USER_ID, "local")


def registry_for(owner_id: str) -> Registry:
    return registry_for_user(owner_id, LOCAL_USER_ID)


# -- passwords ----------------------------------------------------------------------


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode("utf8"), salt=salt, n=_N, r=_R, p=_P,
                            maxmem=_MAXMEM, dklen=32)
    return f"scrypt${_N}${_R}${_P}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(digest)
        actual = hashlib.scrypt(password.encode("utf8"), salt=base64.b64decode(salt),
                                n=int(n), r=int(r), p=int(p), maxmem=_MAXMEM,
                                dklen=len(expected))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


_dummy_hash: Optional[str] = None


def _burn_time() -> None:
    """Spend the same time as a real check, so an unknown email is not detectable
    by how quickly the sign-in is refused."""
    global _dummy_hash
    if _dummy_hash is None:
        _dummy_hash = hash_password("not-a-real-password")
    verify_password("wrong", _dummy_hash)


# -- rate limiting ------------------------------------------------------------------

_hits: dict[str, list[float]] = {}
_hits_lock = threading.Lock()


def _count(key: str, window: float) -> int:
    now = time.monotonic()
    with _hits_lock:
        recent = [t for t in _hits.get(key, []) if now - t < window]
        if recent:
            _hits[key] = recent
        else:
            _hits.pop(key, None)
        return len(recent)


def _record(key: str) -> None:
    with _hits_lock:
        _hits.setdefault(key, []).append(time.monotonic())


def _client_ip(request: Request) -> str:
    # Behind a platform proxy the last X-Forwarded-For entry is the one the proxy
    # itself appended (earlier entries are client-supplied and spoofable).
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def _refuse_if_throttled(*checks: tuple[str, int, float]) -> None:
    for key, limit, window in checks:
        if _count(key, window) >= limit:
            raise HTTPException(status_code=429,
                                detail="Too many attempts. Wait a few minutes and try again.")


# -- sessions -----------------------------------------------------------------------


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf8")).hexdigest()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _start_session(request: Request, response: Response, user_id: str) -> None:
    token = secrets.token_urlsafe(32)
    lifetime = timedelta(days=settings.session_days)
    with session_scope() as session:
        session.query(SessionRecord).filter(SessionRecord.expires_at < _now()).delete()
        session.add(SessionRecord(id=_hash_token(token), user_id=user_id,
                                  expires_at=_now() + lifetime))
        session.commit()
    secure = (request.url.scheme == "https"
              or request.headers.get("x-forwarded-proto", "") == "https")
    response.set_cookie(COOKIE, token, max_age=int(lifetime.total_seconds()), path="/",
                        httponly=True, samesite="lax", secure=secure)


def _session_user(request: Request) -> Optional[User]:
    token = request.cookies.get(COOKIE)
    if not token:
        return None
    with session_scope() as session:
        row = (session.query(SessionRecord, UserRecord)
               .join(UserRecord, UserRecord.id == SessionRecord.user_id)
               .filter(SessionRecord.id == _hash_token(token),
                       SessionRecord.expires_at > _now()).first())
        return User(row[1].id, row[1].email) if row else None


def _check_api_key(supplied: Optional[str]) -> None:
    """Optional API-key gate for the no-accounts mode (programmatic clients)."""
    if not settings.auth_enabled:
        return
    given = (supplied or "").encode("utf8")
    if not any(hmac.compare_digest(given, key.encode("utf8")) for key in settings.api_keys):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def current_user(request: Request,
                 x_api_key: Optional[str] = Header(default=None)) -> User:
    """The signed-in user, or the implicit local user when accounts are off."""
    if settings.require_login:
        user = _session_user(request)
        if user is None:
            raise HTTPException(status_code=401, detail="Sign in to continue.")
        return user
    _check_api_key(x_api_key)
    return LOCAL


# -- routes -------------------------------------------------------------------------

router = APIRouter(prefix="/auth", tags=["accounts"])


class Credentials(BaseModel):
    email: str
    password: str


class UserOut(BaseModel):
    id: str
    email: str


class Me(BaseModel):
    login_required: bool
    user: Optional[UserOut] = None


def _require_accounts() -> None:
    if not settings.require_login:
        raise HTTPException(status_code=400,
                            detail="Accounts are not enabled on this server.")


def _normalize_email(raw: str) -> str:
    email = raw.strip().lower()
    if len(email) > 254 or not _EMAIL.match(email):
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    return email


_TAKEN = "That email already has an account. Sign in instead."


@router.post("/signup", response_model=Me, status_code=201)
def signup(creds: Credentials, request: Request, response: Response) -> Me:
    _require_accounts()
    ip = _client_ip(request)
    _refuse_if_throttled((f"signup:{ip}", 10, 3600.0))
    email = _normalize_email(creds.email)
    if not MIN_PASSWORD <= len(creds.password) <= MAX_PASSWORD:
        raise HTTPException(
            status_code=400,
            detail=f"Use a password of {MIN_PASSWORD} to {MAX_PASSWORD} characters.")
    _record(f"signup:{ip}")
    password_hash = hash_password(creds.password)
    with session_scope() as session:
        if session.query(UserRecord).filter_by(email=email).first():
            raise HTTPException(status_code=409, detail=_TAKEN)
        user = UserRecord(email=email, password_hash=password_hash)
        session.add(user)
        try:
            session.commit()
        except IntegrityError:   # lost a race with an identical sign-up
            session.rollback()
            raise HTTPException(status_code=409, detail=_TAKEN)
        out = UserOut(id=user.id, email=user.email)
    _start_session(request, response, out.id)
    return Me(login_required=True, user=out)


@router.post("/login", response_model=Me)
def login(creds: Credentials, request: Request, response: Response) -> Me:
    _require_accounts()
    ip = _client_ip(request)
    email = creds.email.strip().lower()
    _refuse_if_throttled((f"login-ip:{ip}", 10, 600.0), (f"login-email:{email}", 20, 600.0))
    out: Optional[UserOut] = None
    if len(creds.password) <= MAX_PASSWORD:
        with session_scope() as session:
            user = session.query(UserRecord).filter_by(email=email).first()
            if user is not None and verify_password(creds.password, user.password_hash):
                out = UserOut(id=user.id, email=user.email)
            elif user is None:
                _burn_time()
    if out is None:
        _record(f"login-ip:{ip}")
        _record(f"login-email:{email}")
        raise HTTPException(status_code=401, detail="Wrong email or password.")
    _start_session(request, response, out.id)
    return Me(login_required=True, user=out)


@router.post("/logout")
def logout(request: Request, response: Response) -> dict[str, bool]:
    token = request.cookies.get(COOKIE)
    if token:
        with session_scope() as session:
            session.query(SessionRecord).filter_by(id=_hash_token(token)).delete()
            session.commit()
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


@router.get("/me", response_model=Me)
def me(request: Request) -> Me:
    """Who am I? Never an error: the page asks this before it knows whether to show
    the sign-in screen, so "login required, nobody signed in" is a normal answer."""
    if not settings.require_login:
        return Me(login_required=False, user=UserOut(id=LOCAL.id, email=LOCAL.email))
    user = _session_user(request)
    return Me(login_required=True,
              user=UserOut(id=user.id, email=user.email) if user else None)
