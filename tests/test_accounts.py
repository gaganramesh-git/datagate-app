"""Accounts: sign-up and sign-in, and — the part that matters — that one person's
data, reports and history are invisible to everyone else."""

from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from datagate import demo
from datagate.service.auth import hash_password, verify_password
from tests.test_service import _boot, run_analysis, upload, wait

PASSWORD = "correct horse battery"


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    return _boot(tmp_path, monkeypatch, DATAGATE_REQUIRE_LOGIN="1")


@pytest.fixture
def two(api):
    with TestClient(api.app) as a, TestClient(api.app) as b:
        yield a, b


@pytest.fixture
def csv(tmp_path: Path) -> Path:
    path = tmp_path / "reviews.csv"
    demo.make_reviews(400, seed=1).write_csv(path)
    return path


def signup(client: TestClient, email: str, password: str = PASSWORD):
    return client.post("/auth/signup", json={"email": email, "password": password})


# -- passwords ----------------------------------------------------------------------


def test_password_hashing_roundtrip_and_malformed_input():
    stored = hash_password("s3cret-pass")
    assert stored.startswith("scrypt$") and "s3cret-pass" not in stored
    assert verify_password("s3cret-pass", stored)
    assert not verify_password("s3cret-pasS", stored)
    assert hash_password("s3cret-pass") != stored        # a fresh salt every time
    for junk in ("", "scrypt$1$2", "md5$a$b$c$d$e", "scrypt$x$8$3$!!$!!"):
        assert not verify_password("s3cret-pass", junk)


# -- signing up and in --------------------------------------------------------------


def test_everything_needs_a_signed_in_user(api):
    with TestClient(api.app) as client:
        for path in ("/datasets", "/analyses", "/deals", "/meta"):
            assert client.get(path).status_code == 401, path
        assert client.get("/health").status_code == 200
        assert client.get("/").status_code == 200            # the page itself loads
        assert client.get("/auth/me").json() == {"login_required": True, "user": None}


def test_signup_signs_you_in_with_a_safe_cookie(api):
    with TestClient(api.app) as client:
        res = signup(client, "  Ann@Example.com ")
        assert res.status_code == 201
        cookie = res.headers["set-cookie"].lower()
        assert "httponly" in cookie and "samesite=lax" in cookie
        assert client.get("/auth/me").json()["user"]["email"] == "ann@example.com"
        assert client.get("/datasets").status_code == 200


def test_secrets_are_not_stored_in_the_clear(api):
    db = importlib.import_module("datagate.service.db")
    with TestClient(api.app) as client:
        signup(client, "ann@example.com")
        token = client.cookies.get("dg_session")
    with db.session_scope() as session:
        user = session.query(db.UserRecord).one()
        assert PASSWORD not in user.password_hash and user.password_hash.startswith("scrypt$")
        assert [s.id for s in session.query(db.SessionRecord)] != [token]


@pytest.mark.parametrize("email,password,status", [
    ("not-an-email", PASSWORD, 400), ("a@b", PASSWORD, 400), ("ann@example.com", "short", 400),
    ("ann@example.com", "x" * 129, 400)])
def test_signup_rejects_bad_input(api, email, password, status):
    with TestClient(api.app) as client:
        assert signup(client, email, password).status_code == status


def test_one_account_per_email_ignoring_case(two):
    a, b = two
    assert signup(a, "ann@example.com").status_code == 201
    assert signup(b, "ANN@example.com").status_code == 409


def test_login_and_logout_roundtrip(two):
    a, b = two
    signup(a, "ann@example.com")
    assert a.post("/auth/logout").json() == {"ok": True}
    assert a.get("/datasets").status_code == 401
    wrong = a.post("/auth/login", json={"email": "ann@example.com", "password": "nope-nope"})
    unknown = a.post("/auth/login", json={"email": "who@example.com", "password": "nope-nope"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()                # no hint which part was wrong
    good = b.post("/auth/login", json={"email": "Ann@example.com", "password": PASSWORD})
    assert good.status_code == 200 and b.get("/datasets").status_code == 200


def test_wrong_passwords_are_rate_limited(api):
    with TestClient(api.app) as client:
        creds = {"email": "ann@example.com", "password": "wrong-wrong"}
        assert [client.post("/auth/login", json=creds).status_code
                for _ in range(10)] == [401] * 10
        assert client.post("/auth/login", json=creds).status_code == 429


def test_expired_sessions_stop_working(api):
    db = importlib.import_module("datagate.service.db")
    with TestClient(api.app) as client:
        signup(client, "ann@example.com")
        with db.session_scope() as session:
            for row in session.query(db.SessionRecord):
                row.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
            session.commit()
        assert client.get("/datasets").status_code == 401


def test_cross_origin_writes_are_refused(api):
    with TestClient(api.app) as client:
        body = {"email": "ann@example.com", "password": PASSWORD}
        evil = client.post("/auth/signup", json=body, headers={"Origin": "https://evil.example"})
        assert evil.status_code == 403
        same = client.post("/auth/signup", json=body, headers={"Origin": "http://testserver"})
        assert same.status_code == 201


def test_without_accounts_it_is_one_open_local_user(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch):
    with TestClient(_boot(tmp_path, monkeypatch).app) as client:
        assert client.get("/auth/me").json()["login_required"] is False
        assert client.post("/auth/signup", json={"email": "a@b.co", "password": PASSWORD}
                           ).status_code == 400
        assert client.get("/datasets").status_code == 200


def test_data_from_before_accounts_is_adopted_by_the_local_user(api):
    db = importlib.import_module("datagate.service.db")
    db.init_db()                                   # the tables an older version had
    with db.session_scope() as session:
        session.add(db.DatasetRecord(filename="old.csv", storage_ref="x", format="csv"))
        session.commit()
    db.init_db()
    with db.session_scope() as session:
        assert session.query(db.DatasetRecord).one().owner_id == db.LOCAL_USER_ID


# -- privacy between accounts -------------------------------------------------------


def test_people_cannot_see_each_others_data_or_reports(two, csv):
    a, b = two
    signup(a, "ann@example.com")
    signup(b, "bob@example.com")
    status = run_analysis(a, csv)
    dataset_id, report_id, job_id = status["dataset_id"], status["report_id"], status["id"]
    spec_id = status["spec_id"]

    # Ann can see all of it...
    assert [d["id"] for d in a.get("/datasets").json()] == [dataset_id]
    assert a.get(f"/reports/{report_id}").status_code == 200
    # ...Bob sees nothing, and every direct link answers "not found".
    assert b.get("/datasets").json() == [] and b.get("/analyses").json() == []
    for path in (f"/datasets/{dataset_id}", f"/specs/{spec_id}", f"/analyses/{job_id}",
                 f"/reports/{report_id}", f"/reports/{report_id}/markdown",
                 f"/reports/{report_id}/clean", f"/reports/{report_id}/quarantine",
                 f"/reports/{report_id}/rows?indices=0"):
        assert b.get(path).status_code == 404, path
    assert b.delete(f"/datasets/{dataset_id}").status_code == 404
    assert b.post("/specs", json={"description": "x", "dataset_id": dataset_id}
                  ).status_code == 404
    bobs = upload(b, csv)
    assert b.post("/analyses", json={"dataset_id": dataset_id,
                                     "spec_id": status["spec_id"]}).status_code == 404
    assert b.post("/analyses", json={"dataset_id": bobs["id"], "spec_id": spec_id}
                  ).status_code == 404                      # nor can he borrow her spec
    assert a.get(f"/datasets/{dataset_id}").status_code == 200   # and nothing was deleted


def test_the_seen_before_index_never_leaks_another_persons_files(two, csv):
    a, b = two
    signup(a, "ann@example.com")
    signup(b, "bob@example.com")
    assert upload(a, csv)["seen_before"] == []
    assert upload(b, csv)["seen_before"] == []               # Bob learns nothing of Ann
    assert upload(a, csv)["seen_before"]                     # Ann recognises her own


def test_marketplace_roles_and_what_each_person_may_see(api, csv):
    with TestClient(api.app) as seller, TestClient(api.app) as buyer, \
            TestClient(api.app) as stranger:
        for client, email in ((seller, "ann@example.com"), (buyer, "bob@example.com"),
                              (stranger, "cy@example.com")):
            signup(client, email)
        deal = seller.post("/deals", json={"dataset_id": upload(seller, csv)["id"],
                                           "title": "Reviews"}).json()
        assert deal["is_seller"]
        seen = buyer.get("/deals").json()
        assert [d["id"] for d in seen] == [deal["id"]] and not seen[0]["is_seller"]
        # A buyer cannot list someone else's dataset, nor withdraw their listing.
        assert buyer.post("/deals", json={"dataset_id": deal["id"], "title": "x"}
                          ).status_code == 404
        assert buyer.post(f"/deals/{deal['id']}/withdraw").status_code == 403

        started = buyer.post(f"/deals/{deal['id']}/evaluations",
                             json={"description": "classify product reviews by sentiment"})
        assert started.status_code == 202
        report_id = wait(buyer, started.json()["id"])["report_id"]

        mine = buyer.get(f"/deals/{deal['id']}").json()
        assert mine["is_evaluator"] and mine["verdict"] and mine["report_id"] == report_id
        theirs = seller.get(f"/deals/{deal['id']}").json()
        assert theirs["verdict"] and theirs["report_id"] == report_id
        blind = seller.get(f"/reports/{report_id}").json()    # score-only, no rows
        assert blind["blind"] and not blind["exports_available"]
        assert seller.get(f"/reports/{report_id}/clean").status_code == 404

        outsider = stranger.get(f"/deals/{deal['id']}").json()
        assert outsider["state"] == "listed" and outsider["verdict"] is None
        assert outsider["report_id"] is None and outsider["history"] == []
        assert stranger.get(f"/reports/{report_id}").status_code == 404

        # Only the evaluating buyer decides; only the seller withdraws.
        assert seller.post(f"/deals/{deal['id']}/decision",
                           json={"decision": "decline"}).status_code == 403
        assert stranger.post(f"/deals/{deal['id']}/decision",
                             json={"decision": "accept"}).status_code == 403
        assert buyer.post(f"/deals/{deal['id']}/decision",
                          json={"decision": "decline"}).status_code == 200
        assert seller.post(f"/deals/{deal['id']}/withdraw").json()["state"] == "withdrawn"
        assert stranger.get("/deals").json() == []
        assert stranger.get(f"/deals/{deal['id']}").status_code == 404
