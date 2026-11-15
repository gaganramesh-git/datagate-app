"""Tests for the two-input Task-Spec Compiler and the optional LLM seam."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from datagate.ingest.loader import load_dataset
from datagate.ingest.profiler import profile_dataset
from datagate.spec import llm
from datagate.spec.compiler import compile_spec
from datagate.types import Modality, TaskType


@pytest.fixture
def reviews(tmp_path: Path):
    rows = [{"text": f"review {i}", "label": "a" if i % 2 else "b"} for i in range(60)]
    p = tmp_path / "r.csv"
    pl.DataFrame(rows).write_csv(p)
    ds = load_dataset(p)
    profile_dataset(ds)
    return ds


def test_brief_enriches_domain_deployment_metric(reviews):
    spec = compile_spec(
        "classify support tickets by sentiment",
        reviews,
        project_brief=(
            "We run an e-commerce marketplace. This model goes into a real-time "
            "production API that routes customer reviews. We care about macro F1."
        ),
    )
    assert spec.task_type is TaskType.CLASSIFICATION
    assert spec.domain == "e-commerce"
    assert spec.deployment_context == "real-time production"
    assert spec.success_metric == "macro F1"


def test_objective_alone_still_works(reviews):
    spec = compile_spec("classify reviews by sentiment", reviews)
    assert spec.task_type is TaskType.CLASSIFICATION
    assert spec.domain is None  # nothing to draw from without a brief
    assert spec.project_brief is None


def test_brief_helps_task_inference_when_objective_is_thin(reviews):
    # A bare objective that lacks task keywords; the brief carries the signal.
    spec = compile_spec(
        "reviews model",
        reviews,
        project_brief="We want to detect and classify spam in the review text.",
    )
    assert spec.task_type is TaskType.CLASSIFICATION


def test_llm_disabled_by_default(monkeypatch):
    monkeypatch.delenv("DATAGATE_USE_LLM", raising=False)
    assert llm.llm_enabled() is False
    assert llm.get_backend() is None


def test_llm_stays_off_without_key(monkeypatch):
    monkeypatch.setenv("DATAGATE_USE_LLM", "1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DATAGATE_LLM_API_KEY", raising=False)
    assert llm.llm_enabled() is False


def test_llm_backend_merges_only_gaps(reviews, monkeypatch):
    class FakeBackend:
        def suggest_fields(self, prompt):
            return {
                "domain": "healthcare",
                "target_column": "label",
                "task_type": "classification",
            }

    monkeypatch.setattr(llm, "get_backend", lambda: FakeBackend())

    # use_llm forced on; user did NOT set a target, heuristic found domain=None.
    spec = compile_spec(
        "classify clinical notes", reviews, project_brief="hospital data",
        use_llm=True,
    )
    assert spec.domain == "healthcare"  # gap filled by the LLM
    assert "domain" in spec.llm_assisted_fields


def test_llm_never_overrides_explicit_target(reviews, monkeypatch):
    class FakeBackend:
        def suggest_fields(self, prompt):
            return {"target_column": "text"}  # tries to override

    monkeypatch.setattr(llm, "get_backend", lambda: FakeBackend())
    spec = compile_spec(
        "classify", reviews, target_column="label", use_llm=True,
    )
    assert spec.target_column == "label"  # user's choice wins
    assert "target_column" not in spec.llm_assisted_fields


def test_sanitize_drops_hallucinated_target():
    clean = llm.sanitize(
        {"target_column": "nonexistent", "domain": "finance", "bogus": "x"},
        columns=["a", "b"],
    )
    assert clean == {"domain": "finance"}


def test_sanitize_keeps_valid_target():
    clean = llm.sanitize({"target_column": "label"}, columns=["label", "text"])
    assert clean == {"target_column": "label"}


def test_metric_multiword_precedence(reviews):
    spec = compile_spec("classify", reviews,
                        project_brief="optimize for macro F1 across classes")
    assert spec.success_metric == "macro F1"


def test_modality_inferred_from_combined_text(reviews):
    spec = compile_spec("build a model", reviews,
                        project_brief="the dataset is product review text in English")
    assert spec.modality is Modality.TEXT
