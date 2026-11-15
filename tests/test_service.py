"""API tests for the service.

Each test gets an isolated home (database, blobs, registry, signing key) via
environment overrides applied before the service modules import their settings.
"""

from __future__ import annotations

import importlib
import time
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from datagate import demo

SERVICE_MODULES = ("datagate.service.db", "datagate.service.storage",
                   "datagate.service.auth", "datagate.service.worker",
                   "datagate.service.api")
PURPOSE = "classify product reviews by sentiment"


def _boot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **env: str):
    monkeypatch.setenv("DATAGATE_HOME", str(tmp_path))
    monkeypatch.setenv("DATAGATE_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("DATAGATE_STORAGE_DIR", str(tmp_path / "blobs"))
    monkeypatch.delenv("DATAGATE_API_KEYS", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    from datagate.service import config

    importlib.reload(config)
    for name in SERVICE_MODULES:
        importlib.reload(importlib.import_module(name))
    from datagate.service import api as api_module

    return api_module


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    with TestClient(_boot(tmp_path, monkeypatch).app) as c:
        yield c


@pytest.fixture
def files(tmp_path: Path) -> dict[str, Path]:
    dirty, _ = demo.corrupt(demo.make_reviews(600, seed=3), seed=3, backdoor=0.03)
    frames = {"clean.csv": demo.make_reviews(500, seed=1), "dirty.csv": dirty,
              "reference.csv": demo.make_reviews(200, seed=2)}
    out = {}
    for name, frame in frames.items():
        out[name] = tmp_path / name
        frame.write_csv(out[name])
    return out


def upload(client, path: Path) -> dict:
    with open(path, "rb") as fh:
        res = client.post("/datasets", files={"file": (path.name, fh, "text/csv")})
    assert res.status_code == 200, res.text
    return res.json()


def make_spec(client, dataset_id: str, **extra) -> dict:
    res = client.post("/specs", json={"description": PURPOSE, "dataset_id": dataset_id,
                                      **extra})
    assert res.status_code == 200, res.text
    return res.json()


def wait(client, job_id: str, timeout: float = 240) -> dict:
    deadline = time.time() + timeout
    status: dict = {}
    while time.time() < deadline:
        status = client.get(f"/analyses/{job_id}").json()
        if status["status"] in ("succeeded", "failed"):
            return status
        time.sleep(0.3)
    raise AssertionError(f"job did not settle: {status}")


def run_analysis(client, dataset_path: Path, reference_path: Path | None = None) -> dict:
    dataset = upload(client, dataset_path)
    body = {"dataset_id": dataset["id"], "spec_id": make_spec(client, dataset["id"])["id"]}
    if reference_path is not None:
        body["reference_dataset_id"] = upload(client, reference_path)["id"]
    started = client.post("/analyses", json=body)
    assert started.status_code == 202, started.text
    status = wait(client, started.json()["id"])
    assert status["status"] == "succeeded", status.get("error")
    return status


# -- basics -----------------------------------------------------------------------------


def test_health_meta_and_security_headers(client):
    res = client.get("/health")
    assert res.json()["status"] == "ok"
    assert res.headers["X-Content-Type-Options"] == "nosniff"
    assert "script-src 'self'" in res.headers["Content-Security-Policy"]
    meta = client.get("/meta").json()
    assert len(meta["analyzers"]) == 16
    assert meta["signing_public_key"] and meta["llm_enabled"] is False


def test_upload_returns_profile_with_kinds(client, files):
    body = upload(client, files["clean.csv"])
    assert body["n_rows"] == 500 and body["n_cols"] == 2
    assert body["columns"]["text"]["kind"] == "text"
    assert body["columns"]["label"]["kind"] == "categorical"
    assert len(body["fingerprint"]) == 64 and body["seen_before"] == []


def test_reupload_is_recognised_and_shares_one_blob(client, files, tmp_path: Path):
    first = upload(client, files["clean.csv"])
    second = upload(client, files["clean.csv"])
    assert first["id"] != second["id"] and first["fingerprint"] == second["fingerprint"]
    assert second["seen_before"][0]["exact"] is True
    blobs = [p for p in (tmp_path / "blobs").iterdir() if not p.name.startswith(".")]
    assert len(blobs) == 1


@pytest.mark.parametrize("name,content,expected", [
    ("notes.xyz", b"hello", "unsupported format"),
    ("empty.csv", b"", "empty"),
])
def test_upload_rejects_bad_files(client, name, content, expected):
    res = client.post("/datasets", files={"file": (name, content, "text/plain")})
    assert res.status_code == 400 and expected in res.json()["detail"]


def test_upload_size_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, files):
    api = _boot(tmp_path / "small", monkeypatch, DATAGATE_MAX_UPLOAD_BYTES="2000")
    with TestClient(api.app) as c:
        with open(files["clean.csv"], "rb") as fh:
            res = c.post("/datasets", files={"file": ("clean.csv", fh, "text/csv")})
        assert res.status_code == 413
        leftovers = list((tmp_path / "small" / "blobs").iterdir())
        assert leftovers == []  # the partial upload was cleaned up


def test_upload_filename_cannot_escape_storage(client, files):
    with open(files["clean.csv"], "rb") as fh:
        res = client.post("/datasets", files={"file": ("../../evil.csv", fh, "text/csv")})
    assert res.status_code == 200 and res.json()["filename"] == "evil.csv"


# -- specs --------------------------------------------------------------------------------


def test_spec_uses_brief_and_is_editable(client, files):
    dataset = upload(client, files["clean.csv"])
    body = make_spec(client, dataset["id"], project_brief=(
        "An e-commerce marketplace; a real-time production API; we track macro F1."))
    spec = body["spec"]
    assert (spec["task_type"], spec["target_column"]) == ("classification", "label")
    assert (spec["domain"], spec["success_metric"]) == ("e-commerce", "macro F1")
    assert body["needs_clarification"] is False

    patched = client.patch(f"/specs/{body['id']}", json={"spec": {**spec, "domain": "retail"}})
    assert patched.json()["spec"]["domain"] == "retail"
    assert client.get(f"/specs/{body['id']}").json()["spec"]["domain"] == "retail"


def test_ambiguous_spec_returns_questions(client):
    body = client.post("/specs", json={"description": "do something with my data"}).json()
    assert body["needs_clarification"] is True and body["spec"]["clarifying_questions"]


# -- analyses and reports -------------------------------------------------------------------


def test_clean_dataset_passes_end_to_end(client, files):
    status = run_analysis(client, files["clean.csv"], files["reference.csv"])
    assert status["verdict"] == "pass" and status["dataset_filename"] == "clean.csv"
    assert status["progress"]["done"] == status["progress"]["total"] == 16
    assert len(set(status["progress"]["finished"])) == 16   # the UI names each finished check

    body = client.get(f"/reports/{status['report_id']}").json()
    report = body["report"]
    assert body["dataset_filename"] == "clean.csv" and body["exports_available"]
    assert report["decision"]["verdict"] == "pass"
    assert report["reference_name"] == "reference.csv"
    assert report["predicted_gain_range"]
    assert report["signature"]["algorithm"] == "ed25519"

    verified = client.post("/reports/verify", json=report).json()
    assert verified["ok"] and verified["trusted"] is True
    report["fit"]["score"] = 0.01
    assert client.post("/reports/verify", json=report).json()["digest_ok"] is False


def test_dirty_dataset_is_blocked_and_exports_work(client, files):
    status = run_analysis(client, files["dirty.csv"])
    assert status["verdict"] == "block"
    rid = status["report_id"]
    report = client.get(f"/reports/{rid}").json()["report"]
    total = report["quarantine_total"]
    assert total > 0 and report["decision"]["reasons"]

    quarantine = pl.read_csv(client.get(f"/reports/{rid}/quarantine").content)
    assert quarantine.height == total
    assert {"row_index", "severity", "reasons", "text"} <= set(quarantine.columns)

    cleaned = pl.read_csv(client.get(f"/reports/{rid}/clean").content)
    assert cleaned.height == report["dataset_rows"] - total

    redacted = client.get(f"/reports/{rid}/clean?redact_pii=true")
    assert "[SSN]" in redacted.text and "@example.com" not in redacted.text
    assert "attachment" in redacted.headers["Content-Disposition"]

    first = report["quarantine_list"][0]["row_index"]
    rows = client.get(f"/reports/{rid}/rows?indices={first},999999").json()
    assert [r["row_index"] for r in rows["rows"]] == [first]
    assert client.get(f"/reports/{rid}/rows?indices=abc").status_code == 400

    markdown = client.get(f"/reports/{rid}/markdown").text
    assert "**BLOCK**" in markdown and report["digest"] in markdown


def test_history_lists_past_analyses(client, files):
    run_analysis(client, files["clean.csv"])
    history = client.get("/analyses").json()
    assert len(history) == 1
    assert history[0]["verdict"] == "pass" and history[0]["safety_score"] > 0.8


def test_purged_dataset_keeps_report_but_not_exports(client, files, tmp_path: Path):
    status = run_analysis(client, files["dirty.csv"])
    rid, dataset_id = status["report_id"], status["dataset_id"]
    assert client.delete(f"/datasets/{dataset_id}").json()["blob_deleted"] is True
    assert not any(p.suffix == ".csv" for p in (tmp_path / "blobs").iterdir())

    assert client.get(f"/datasets/{dataset_id}").json()["available"] is False
    body = client.get(f"/reports/{rid}").json()
    assert body["report"]["decision"]["verdict"] == "block"
    assert body["exports_available"] is False
    assert client.get(f"/reports/{rid}/clean").status_code == 410
    spec = make_spec(client, upload(client, files["clean.csv"])["id"])
    again = client.post("/analyses", json={"dataset_id": dataset_id, "spec_id": spec["id"]})
    assert again.status_code == 410


def test_failed_job_reports_its_error(client, files, monkeypatch: pytest.MonkeyPatch):
    from datagate.service import worker

    def boom(*args, **kwargs):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr(worker, "analyze", boom)
    dataset = upload(client, files["clean.csv"])
    spec = make_spec(client, dataset["id"])
    job = client.post("/analyses", json={"dataset_id": dataset["id"],
                                         "spec_id": spec["id"]}).json()
    status = wait(client, job["id"])
    assert status["status"] == "failed" and "engine exploded" in status["error"]


def test_orphaned_jobs_are_failed_on_restart(client, files):
    from datagate.service import db, worker

    dataset = upload(client, files["clean.csv"])
    spec = make_spec(client, dataset["id"])
    with db.session_scope() as session:
        job = db.JobRecord(dataset_id=dataset["id"], spec_id=spec["id"],
                           status=db.JobStatus.RUNNING.value,
                           owner_id=db.LOCAL_USER_ID)
        session.add(job)
        session.commit()
        job_id = job.id
    assert worker.recover_orphaned_jobs() == 1
    status = client.get(f"/analyses/{job_id}").json()
    assert status["status"] == "failed" and "restart" in status["error"]


def test_unknown_ids_and_modes(client, files):
    assert client.post("/analyses", json={"dataset_id": "x", "spec_id": "y"}).status_code == 404
    for path in ("/datasets/x", "/analyses/x", "/reports/x", "/specs/x", "/deals/x"):
        assert client.get(path).status_code == 404
    dataset = upload(client, files["clean.csv"])
    spec = make_spec(client, dataset["id"])
    res = client.post("/analyses", json={"dataset_id": dataset["id"],
                                         "spec_id": spec["id"], "mode": "B"})
    assert res.status_code == 400 and "/deals" in res.json()["detail"]


def test_api_key_enforced_when_configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    api = _boot(tmp_path, monkeypatch, DATAGATE_API_KEYS="secret-key")
    with TestClient(api.app) as c:
        assert c.get("/health").status_code == 200  # health stays open
        assert c.get("/datasets").status_code == 401
        assert c.get("/datasets", headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.get("/datasets", headers={"X-API-Key": "secret-key"}).status_code == 200


def test_old_database_gains_new_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import sqlite3

    tmp_path.mkdir(exist_ok=True)
    con = sqlite3.connect(tmp_path / "test.db")
    con.execute("CREATE TABLE jobs (id VARCHAR(36) PRIMARY KEY, dataset_id VARCHAR(36), "
                "spec_id VARCHAR(36), mode VARCHAR(1), status VARCHAR(16))")
    con.execute("INSERT INTO jobs VALUES ('old', 'd', 's', 'A', 'succeeded')")
    con.commit()
    con.close()

    api = _boot(tmp_path, monkeypatch)
    with TestClient(api.app):
        con = sqlite3.connect(tmp_path / "test.db")
        columns = {row[1] for row in con.execute("PRAGMA table_info(jobs)")}
        assert {"progress", "deal_id", "report_id"} <= columns
        assert con.execute("SELECT status FROM jobs WHERE id='old'").fetchone() == ("succeeded",)
        con.close()


# -- blind evaluation (Mode B preview) --------------------------------------------------------


def test_blind_evaluation_only_the_score_leaves(client, files):
    seller_data = upload(client, files["dirty.csv"])
    deal = client.post("/deals", json={"dataset_id": seller_data["id"],
                                       "title": "Retail reviews, 2024",
                                       "asking_price": "$400"}).json()
    assert deal["state"] == "listed"
    assert deal["columns"] == {"text": "text", "label": "categorical"}
    assert "filename" not in deal and deal["fingerprint_prefix"]

    # A deal cannot be decided before it has been evaluated.
    early = client.post(f"/deals/{deal['id']}/decision", json={"decision": "accept"})
    assert early.status_code == 409

    reference = upload(client, files["reference.csv"])
    started = client.post(f"/deals/{deal['id']}/evaluations", json={
        "description": PURPOSE, "reference_dataset_id": reference["id"]})
    assert started.status_code == 202 and started.json()["dataset_filename"] is None
    status = wait(client, started.json()["id"])
    assert status["status"] == "succeeded", status.get("error")

    deal = client.get(f"/deals/{deal['id']}").json()
    assert deal["state"] == "evaluated" and deal["report_id"] == status["report_id"]
    # The buyer sees the outcome on the deal itself, never deciding blind.
    assert deal["verdict"] == "block" and deal["safety_score"] < 0.5

    body = client.get(f"/reports/{deal['report_id']}").json()
    report = body["report"]
    assert body["blind"] is True and body["exports_available"] is False
    assert body["deal_id"] == deal["id"] and body["deal_state"] == "evaluated"
    assert body["dataset_filename"] == "Retail reviews, 2024"
    assert report["decision"]["verdict"] == "block"       # the buyer learns the verdict
    assert report["quarantine_total"] > 0                 # ...and how much is wrong
    assert report["quarantine_list"] == []                # ...but not which rows
    assert all(e["flags"] == [] for e in report["evidence"])
    assert report["dataset_path"] == ""
    text = str(report)
    assert demo.TRIGGER not in text and "@example.com" not in text and "dirty.csv" not in text

    for path in ("clean", "quarantine", "rows?indices=1"):
        assert client.get(f"/reports/{deal['report_id']}/{path}").status_code == 403

    verified = client.post("/reports/verify", json=report).json()
    assert verified["ok"] and verified["trusted"] is True

    declined = client.post(f"/deals/{deal['id']}/decision", json={"decision": "decline"})
    assert declined.json()["state"] == "declined"
    events = [e["event"] for e in declined.json()["history"]]
    assert events == ["listed", "evaluation_started", "evaluated", "declined"]
    assert client.post(f"/deals/{deal['id']}/withdraw").json()["state"] == "withdrawn"
    again = client.post(f"/deals/{deal['id']}/evaluations", json={"description": PURPOSE})
    assert again.status_code == 409

