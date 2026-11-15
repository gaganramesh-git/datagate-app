"""Engine tests: ingestion, each analyzer, the gate, exports, signing, registry.

Datasets come from ``datagate.demo`` — the same seeded generators the eval
harness uses — so every "detected" assertion is against planted ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from datagate import demo, export, signing
from datagate.aggregate import aggregate
from datagate.analyzers.fit.valuation import knn_shapley
from datagate.analyzers.security.pii import luhn_valid, redact_text
from datagate.analyzers.security.provenance import ProvenanceAnalyzer
from datagate.ingest.loader import infer_kinds, load_dataset
from datagate.ingest.profiler import profile_dataset
from datagate.orchestrator import analyze, default_analyzers
from datagate.registry import Registry, containment, row_sketch
from datagate.report import render_markdown
from datagate.scoring import risk_score, robust_z
from datagate.types import GatePolicy, ScoredEvidence, Severity, Verdict

PURPOSE = "classify product reviews by sentiment"


def write(df: pl.DataFrame, path: Path) -> str:
    df.write_csv(path)
    return str(path)


def run(path: str, purpose: str = PURPOSE, **kwargs):
    """Analyze without the provenance check, which reads machine-local state."""
    analyzers = [a for a in default_analyzers() if not isinstance(a, ProvenanceAnalyzer)]
    return analyze(path, purpose, analyzers=analyzers, **kwargs)


def evidence(result, name: str) -> ScoredEvidence:
    return next(e for e in result.report.evidence if e.analyzer == name)


def flagged(result, needle: str) -> set[int]:
    return {f.row_index for f in result.quarantine if needle in f.reason}


@pytest.fixture(scope="module")
def workdir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("engine")


@pytest.fixture(scope="module")
def reference_csv(workdir: Path) -> str:
    return write(demo.make_reviews(300, seed=2), workdir / "reference.csv")


@pytest.fixture(scope="module")
def clean(workdir: Path, reference_csv: str):
    path = write(demo.make_reviews(800, seed=1), workdir / "clean.csv")
    return run(path, target_column="label", reference_path=reference_csv)


@pytest.fixture(scope="module")
def dirty(workdir: Path, reference_csv: str):
    df, planted = demo.corrupt(demo.make_reviews(1200, seed=3), seed=3, backdoor=0.03,
                               hidden_rows=6, injection_rows=6)
    path = write(df, workdir / "dirty.csv")
    return run(path, target_column="label", reference_path=reference_csv), planted


# -- ingestion ------------------------------------------------------------------


def test_load_profile_and_kinds(workdir: Path):
    df, _ = demo.make_tabular(200, seed=1)
    ds = load_dataset(write(df, workdir / "tab.csv"))
    prof = profile_dataset(ds)
    kinds = ds.kinds()
    assert prof["n_rows"] == 200
    assert kinds["customer_id"] == "id"
    assert kinds["plan"] == "categorical"
    assert kinds["income"] == "numeric"
    assert "churned" in prof["label_candidates"]
    assert len(ds.fingerprint) == 64


def test_text_column_is_text_and_label_is_categorical():
    kinds = infer_kinds(demo.make_reviews(200, seed=1))
    assert kinds == {"text": "text", "label": "categorical"}


def test_loader_rejects_unknown_format(tmp_path: Path):
    p = tmp_path / "data.xyz"
    p.write_text("hello")
    with pytest.raises(ValueError):
        load_dataset(p)


def test_loader_reads_plain_text_and_jsonl(tmp_path: Path):
    txt = tmp_path / "lines.txt"
    txt.write_text("first line\n\nsecond line\n")
    assert load_dataset(txt).df["text"].to_list() == ["first line", "second line"]
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text('{"a": 1, "b": "x"}\n{"a": 2, "b": "y"}\n')
    assert load_dataset(jsonl).n_rows == 2


# -- scoring helpers --------------------------------------------------------------


def test_risk_score_is_calibrated_not_a_clean_fraction():
    assert risk_score(0.0, ok=0.005, bad=0.15) == 1.0
    assert risk_score(0.2, ok=0.005, bad=0.15) == 0.0
    # 5% label noise is a meaningfully noisy dataset, not "95% fine".
    assert 0.6 < risk_score(0.05, ok=0.005, bad=0.15) < 0.75


def test_robust_z_ignores_the_outliers_it_measures():
    values = np.r_[np.random.default_rng(0).normal(0, 1, 500), [50.0]]
    assert robust_z(values)[-1] > 20


# -- the gate on clean vs dirty data -------------------------------------------------


def test_clean_data_passes(clean):
    r = clean.report
    assert r.decision.verdict is Verdict.PASS
    assert r.safety.score >= 0.9
    assert r.fit.score >= 0.75
    assert len(clean.quarantine) <= 0.01 * r.dataset_rows
    assert {e.family for e in r.evidence} == {"security", "fit"}


def test_dirty_data_is_blocked_with_reasons(dirty):
    result, _ = dirty
    r = result.report
    assert r.decision.verdict is Verdict.BLOCK
    assert r.safety.score < r.policy.block_safety
    assert r.safety.label == "critical findings — blocked"
    assert any("SSN" in reason for reason in r.decision.reasons)
    assert r.recommendations


def test_every_planted_problem_is_detected(dirty):
    result, planted = dirty

    def recall(needle: str, key: str) -> float:
        return len(flagged(result, needle) & planted[key]) / len(planted[key])

    assert recall("suspected_mislabel", "mislabels") >= 0.8
    assert recall("duplicate", "duplicates") >= 0.95
    assert recall("pii:", "pii") == 1.0
    assert recall("backdoor_trigger", "backdoor") >= 0.9
    assert recall("hidden:", "hidden") == 1.0
    assert recall("hidden:", "injection") == 1.0
    everything = {f.row_index for f in result.quarantine}
    assert len(everything & planted["gibberish"]) / len(planted["gibberish"]) >= 0.9


def test_dirty_data_false_flag_rate_is_low(dirty):
    result, planted = dirty
    touched = set().union(*planted.values())
    stray = {f.row_index for f in result.quarantine} - touched
    assert len(stray) / (result.dataset.n_rows - len(touched)) <= 0.03


def test_backdoor_trigger_token_is_named(dirty):
    result, _ = dirty
    triggers = evidence(result, "poisoning").metrics["triggers"]
    assert [t["tokens"] for t in triggers] == [[demo.TRIGGER]]
    assert triggers[0]["label"] == "positive"


def test_honest_strong_words_are_not_called_triggers(clean):
    assert evidence(clean, "poisoning").metrics["triggers"] == []


def test_pii_values_never_appear_in_the_report(dirty):
    result, _ = dirty
    payload = json.dumps(result.report.model_dump(mode="json"))
    assert "@example.com" not in payload
    assert "123-45-" not in payload


def test_analysis_is_deterministic(workdir: Path):
    path = write(demo.corrupt(demo.make_reviews(500, seed=5), seed=5)[0], workdir / "det.csv")
    first = run(path, target_column="label").report
    second = run(path, target_column="label", parallel=False).report
    assert first.digest == second.digest
    assert first.digest and len(first.digest) == 64


# -- individual security analyzers ----------------------------------------------------


def test_quality_flags_unusable_rows(tmp_path: Path):
    rows = demo.make_reviews(300, seed=1).to_dicts()
    rows[3]["label"] = None
    rows[10]["text"] = "   "
    result = run(write(pl.DataFrame(rows), tmp_path / "q.csv"), target_column="label")
    assert 3 in flagged(result, "missing_label")
    assert 10 in flagged(result, "empty_text")


def test_conflicting_labels_are_flagged(tmp_path: Path):
    rows = demo.make_reviews(300, seed=1).to_dicts()
    twin = dict(rows[0])
    twin["label"] = "negative" if twin["label"] == "positive" else "positive"
    rows.append(twin)
    result = run(write(pl.DataFrame(rows), tmp_path / "c.csv"), target_column="label")
    assert {0, 300} <= flagged(result, "conflicting_label")


def test_near_duplicates_are_found(tmp_path: Path):
    rows = demo.make_reviews(300, seed=1).to_dicts()
    rows.append({"text": rows[5]["text"] + " really", "label": rows[5]["label"]})
    result = run(write(pl.DataFrame(rows), tmp_path / "n.csv"), target_column="label")
    assert 300 in flagged(result, "near_duplicate")


def test_low_cardinality_tabular_repeats_are_not_duplicates(tmp_path: Path):
    rows = [{"color": c, "size": s, "y": "a" if c == "red" else "b"}
            for c in ("red", "blue") for s in ("s", "m", "l")] * 40
    result = run(write(pl.DataFrame(rows), tmp_path / "lc.csv"),
                 "classify items", target_column="y")
    ev = evidence(result, "duplicates")
    assert ev.metrics.get("treated_as_natural") is True
    assert not flagged(result, "duplicate")


def test_luhn_separates_cards_from_order_numbers():
    assert luhn_valid("4111 1111 1111 1111")
    assert not luhn_valid("4111 1111 1111 1112")
    assert not luhn_valid("1234")


def test_zip_code_next_to_phone_number_is_not_a_card(tmp_path: Path):
    """Found on real newsgroup text: these digits pass Luhn but are not card-shaped."""
    assert luhn_valid("37933 615-966-9200")
    rows = demo.make_reviews(200, seed=1).to_dicts()
    rows[0]["text"] += " Perceptics, Knoxville, TN 37933 615-966-9200"
    result = run(write(pl.DataFrame(rows), tmp_path / "zip.csv"), target_column="label")
    assert evidence(result, "pii").metrics["counts"]["credit_card"] == 0
    assert result.report.decision.verdict is not Verdict.BLOCK


def test_large_boilerplate_group_is_reported_not_quarantined(tmp_path: Path):
    """One author's signature on 12% of a class is a sub-population, not an attack."""
    rows = demo.make_reviews(1000, seed=1).to_dicts()
    signature = " -- sent from the desk of gordon banks skepticism is the chastity of intellect"
    stamped = [i for i, r in enumerate(rows) if r["label"] == "positive"][:60]
    for i in stamped:
        rows[i]["text"] += signature
    result = run(write(pl.DataFrame(rows), tmp_path / "sig.csv"), target_column="label")
    ev = evidence(result, "poisoning")
    assert not (flagged(result, "class_outlier") & set(stamped))
    assert ev.metrics["triggers"] == []
    assert result.report.decision.verdict is Verdict.PASS


def test_secrets_and_cards_are_critical(tmp_path: Path):
    rows = demo.make_reviews(200, seed=1).to_dicts()
    rows[1]["text"] += " card 4111 1111 1111 1111"
    rows[2]["text"] += " order 4111 1111 1111 1112"          # fails Luhn: not a card
    rows[3]["text"] += " key " + "AKIA" + "ABCDEFGHIJKLMNOP"  # assembled, not a literal
    result = run(write(pl.DataFrame(rows), tmp_path / "s.csv"), target_column="label")
    pii = evidence(result, "pii")
    assert pii.metrics["counts"]["credit_card"] == 1
    assert pii.metrics["counts"]["aws_access_key"] == 1
    assert pii.blockers
    assert result.report.decision.verdict is Verdict.BLOCK


def test_redact_text_masks_spans():
    out = redact_text("mail a.b@example.com or call 415-555-0199, ssn 123-45-6789")
    assert out == "mail [EMAIL] or call [PHONE], ssn [SSN]"


def test_zero_width_joiner_is_not_flagged(tmp_path: Path):
    rows = demo.make_reviews(200, seed=1).to_dicts()
    rows[0]["text"] += " family \U0001F468‍\U0001F469‍\U0001F467"  # ZWJ emoji
    result = run(write(pl.DataFrame(rows), tmp_path / "z.csv"), target_column="label")
    assert not flagged(result, "hidden:")


def test_contamination_finds_reference_rows(tmp_path: Path):
    reference = demo.make_reviews(200, seed=9)
    candidate = pl.concat([demo.make_reviews(600, seed=8), reference.head(40)])
    result = run(write(candidate, tmp_path / "cand.csv"), target_column="label",
                 reference_path=write(reference, tmp_path / "ref.csv"))
    ev = evidence(result, "contamination")
    assert ev.metrics["reference_rows_overlapping"] >= 40
    assert set(range(600, 640)) <= flagged(result, "overlaps_reference")
    assert result.report.decision.verdict is not Verdict.PASS


# -- fit analyzers ----------------------------------------------------------------------


def test_reference_checks_are_skipped_without_reference(tmp_path: Path):
    result = run(write(demo.make_reviews(400, seed=1), tmp_path / "x.csv"),
                 target_column="label")
    for name in ("coverage", "shift", "alignment", "valuation", "marginal_gain",
                 "contamination"):
        ev = evidence(result, name)
        assert ev.skipped and ev.needs_reference
    assert any("your own task data" in rec for rec in result.report.recommendations)
    assert result.report.predicted_gain_range is None
    # Learnable, well-formed data is not "strong fit" until checked against the task.
    assert result.report.fit.score <= 0.70
    assert "not verified" in result.report.fit.label


def test_off_topic_data_is_a_poor_fit(tmp_path: Path, reference_csv: str, clean):
    off = run(write(demo.make_offtopic(800, seed=4), tmp_path / "off.csv"),
              target_column="label", reference_path=reference_csv)
    assert off.report.fit.score <= 0.4
    assert clean.report.fit.score - off.report.fit.score >= 0.35
    assert off.report.decision.verdict is Verdict.REVIEW


def test_inverted_labels_read_as_opposed(tmp_path: Path, reference_csv: str):
    flipped = demo.flip_labels(demo.make_reviews(800, seed=6))
    result = run(write(flipped, tmp_path / "flip.csv"), target_column="label",
                 reference_path=reference_csv)
    align = evidence(result, "alignment")
    assert align.metrics["direction"] == "opposed"
    assert align.metrics["cosine"] < -0.5
    assert result.report.fit.score <= 0.3
    assert evidence(result, "valuation").metrics["harmful_fraction"] > 0.5


def test_marginal_gain_reports_a_range(clean):
    ev = evidence(clean, "marginal_gain")
    lo, hi = ev.metrics["predicted_gain_range"]
    assert lo <= hi
    assert ev.metrics["transfer_score"] > 0.9
    assert "balanced accuracy" in clean.report.predicted_gain_range


def test_knn_shapley_values_sum_to_the_knn_utility():
    rng = np.random.default_rng(0)
    y_train = rng.integers(0, 2, 40).astype(str)
    y_val = rng.integers(0, 2, 7).astype(str)
    dist = rng.random((7, 40))
    values = knn_shapley(dist, y_train, y_val, k=5)
    nearest = np.argsort(dist, axis=1)[:, :5]
    utility = np.mean([(y_train[nearest[j]] == y_val[j]).mean() for j in range(7)])
    assert values.sum() == pytest.approx(utility)


def test_target_leak_is_named_and_tabular_labels_checked(tmp_path: Path):
    df, planted = demo.make_tabular(1200, seed=2, label_noise=0.08, leak=True)
    result = run(write(df, tmp_path / "leak.csv"), "predict whether a customer will churn",
                 target_column="churned")
    suspects = evidence(result, "target_leakage").metrics["suspects"]
    assert [s["column"] for s in suspects] == ["refund_issued"]


def test_regression_target_is_supported(tmp_path: Path):
    rng = np.random.default_rng(1)
    x1, x2 = rng.normal(size=600), rng.normal(size=600)
    df = pl.DataFrame({"x1": x1, "x2": x2, "price": 3 * x1 - 2 * x2 + rng.normal(0, 0.1, 600)})
    result = run(write(df, tmp_path / "reg.csv"), "regression to predict the price",
                 target_column="price")
    assert result.report.spec.task_type.value == "regression"
    assert evidence(result, "learnability").score > 0.8
    assert evidence(result, "labels").skipped


def test_missing_target_caps_fit(tmp_path: Path):
    result = run(write(demo.make_reviews(300, seed=1).drop("label"), tmp_path / "nt.csv"),
                 "classify reviews by sentiment")
    assert result.report.fit.score <= 0.3
    assert result.report.decision.verdict is Verdict.REVIEW


# -- aggregator -----------------------------------------------------------------------------


def _ev(family: str, score: float, confidence: float = 0.8, **kw) -> ScoredEvidence:
    return ScoredEvidence(analyzer=kw.pop("analyzer", "a"), family=family, score=score,
                          confidence=confidence, summary="s", **kw)


def test_aggregator_ignores_skipped():
    agg = aggregate([_ev("security", 0.9),
                     _ev("security", 1.0, 0.0, skipped=True, analyzer="b"),
                     _ev("fit", 0.6, 0.5, analyzer="c")])
    assert agg.safety.score == pytest.approx(0.908, abs=1e-3)
    assert agg.fit.score == 0.6
    assert agg.quarantine == []


def test_safety_is_weakest_link_not_an_average():
    good = [_ev("security", 1.0, analyzer=str(i)) for i in range(5)]
    agg = aggregate(good + [_ev("security", 0.0, 0.9, analyzer="bad")])
    assert agg.safety.score < 0.6  # a plain mean would say ~0.82


def test_blocker_forces_block_whatever_the_average():
    agg = aggregate([_ev("security", 0.99, blockers=["secrets found"]),
                     _ev("security", 1.0, analyzer="b"), _ev("fit", 0.9, analyzer="c")])
    assert agg.decision.verdict is Verdict.BLOCK
    assert agg.safety.score < 0.5
    assert agg.decision.reasons == ["secrets found"]


def test_no_active_analyzers_means_no_verdict():
    agg = aggregate([_ev("security", 1.0, 0.0, skipped=True)])
    assert agg.safety.confidence == 0.0
    assert "no verdict" in agg.safety.label
    assert agg.decision.verdict is Verdict.REVIEW


def test_policy_thresholds_move_the_verdict():
    evidence_ = [_ev("security", 0.85), _ev("fit", 0.9, analyzer="f")]
    assert aggregate(evidence_).decision.verdict is Verdict.PASS
    strict = GatePolicy(pass_safety=0.95)
    assert aggregate(evidence_, policy=strict).decision.verdict is Verdict.REVIEW


# -- exports, signing, registry, report rendering ---------------------------------------------


def test_clean_export_removes_quarantined_rows(dirty):
    result, _ = dirty
    cleaned = export.clean_frame(result.dataset, result.quarantine)
    assert cleaned.height == result.dataset.n_rows - len(result.quarantine)
    assert cleaned.columns == result.dataset.df.columns
    quarantine = export.quarantine_frame(result.dataset, result.quarantine)
    assert quarantine.height == len(result.quarantine)
    assert {"row_index", "severity", "reasons", "text", "label"} <= set(quarantine.columns)


def test_redacting_export_keeps_rows_and_masks_pii(dirty):
    result, planted = dirty
    dropped = export.clean_frame(result.dataset, result.quarantine)
    redacted = export.clean_frame(result.dataset, result.quarantine, redact_pii=True)
    assert redacted.height > dropped.height
    text = " ".join(redacted["text"].to_list())
    assert "@example.com" not in text and "[EMAIL]" in text and "[SSN]" in text


def test_cleaned_data_passes_a_second_check(dirty, workdir: Path, reference_csv: str):
    result, _ = dirty
    cleaned = export.clean_frame(result.dataset, result.quarantine)
    again = run(write(cleaned, workdir / "cleaned.csv"), target_column="label",
                reference_path=reference_csv)
    assert again.report.decision.verdict is not Verdict.BLOCK
    assert again.report.safety.score > result.report.safety.score + 0.3


def test_signature_roundtrip_and_tamper_detection(clean, tmp_path: Path):
    key = signing.load_or_create_key(tmp_path)
    report = signing.sign_report(clean.report.model_copy(deep=True), key)
    payload = report.model_dump(mode="json")

    ok = signing.verify_report(payload, trusted_public_key=signing.public_key_b64(key))
    assert ok.ok and ok.trusted is True

    untrusted = signing.verify_report(payload, trusted_public_key="AAAA")
    assert untrusted.signature_ok and untrusted.trusted is False and not untrusted.ok

    payload["quarantine_total"] += 1
    assert not signing.verify_report(payload).digest_ok


def test_digest_survives_a_javascript_json_roundtrip(clean):
    """A browser rewrites 1.0 as 1; the digest must not care."""
    payload = clean.report.model_dump(mode="json")
    assert payload["safety"]["score"] == 1.0  # the case that used to break

    def as_javascript_would(value):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, dict):
            return {k: as_javascript_would(v) for k, v in value.items()}
        if isinstance(value, list):
            return [as_javascript_would(v) for v in value]
        return value

    roundtripped = json.loads(json.dumps(as_javascript_would(payload)))
    assert signing.payload_digest(roundtripped) == clean.report.digest
    assert signing.verify_report(roundtripped).digest_ok


def test_signing_key_is_private_and_reused(tmp_path: Path):
    first = signing.public_key_b64(signing.load_or_create_key(tmp_path))
    second = signing.public_key_b64(signing.load_or_create_key(tmp_path))
    assert first == second
    assert (tmp_path / signing.KEY_FILE).stat().st_mode & 0o077 == 0


def test_registry_estimates_containment(tmp_path: Path):
    base = demo.make_reviews(1000, seed=1)
    full = load_dataset(write(base, tmp_path / "full.csv"))
    half_overlap = load_dataset(write(
        pl.concat([base.head(500), demo.make_reviews(500, seed=99)]), tmp_path / "half.csv"))
    shuffled_cols = load_dataset(write(base.select("label", "text"), tmp_path / "swap.csv"))

    assert containment(row_sketch(half_overlap), row_sketch(full)) == pytest.approx(0.5, abs=0.08)
    assert containment(row_sketch(shuffled_cols), row_sketch(full)) == pytest.approx(1.0)

    registry = Registry(tmp_path / "registry")
    registry.add(full, name="public-reviews", kind="benchmark")
    matches = registry.match(half_overlap)
    assert matches[0].name == "public-reviews" and 0.4 < matches[0].containment < 0.6
    assert registry.remove("public-reviews") == 1 and registry.entries() == []


def test_provenance_blocks_a_benchmark_reupload(tmp_path: Path):
    base = demo.make_reviews(600, seed=1)
    registry = Registry(tmp_path / "registry")
    registry.add(load_dataset(write(base, tmp_path / "bench.csv")), name="bench",
                 kind="benchmark")
    renamed = write(pl.concat([base, demo.make_reviews(20, seed=5)]), tmp_path / "mine.csv")
    result = analyze(renamed, PURPOSE, target_column="label", registry=registry)
    ev = evidence(result, "provenance")
    assert ev.metrics["matches"][0]["name"] == "bench"
    assert ev.blockers and result.report.decision.verdict is Verdict.BLOCK


def test_small_eval_set_swallowed_by_a_large_candidate_is_blocked(tmp_path: Path):
    """The eval set is 10% of the candidate's rows, but 100% of it is in there."""
    eval_set = demo.make_reviews(200, seed=9)
    registry = Registry(tmp_path / "registry")
    registry.add(load_dataset(write(eval_set, tmp_path / "eval.csv")), name="my-eval",
                 kind="eval")
    candidate = pl.concat([demo.make_reviews(1800, seed=8), eval_set]).sample(
        fraction=1.0, seed=1, shuffle=True)
    result = analyze(write(candidate, tmp_path / "big.csv"), PURPOSE,
                     target_column="label", registry=registry)
    match = evidence(result, "provenance").metrics["matches"][0]
    assert match["coverage"] > 0.9 and match["containment"] < 0.2
    assert result.report.decision.verdict is Verdict.BLOCK


def test_target_is_found_from_a_stem_in_the_objective(tmp_path: Path):
    df, _ = demo.make_tabular(300, seed=1)
    result = run(write(df, tmp_path / "churn.csv"), "predict whether a customer will churn")
    assert result.report.spec.target_column == "churned"
    assert not result.report.spec.is_ambiguous


def test_markdown_report_is_grounded_in_the_report(dirty):
    result, _ = dirty
    md = render_markdown(result.report)
    assert "**BLOCK**" in md
    assert result.report.digest in md
    assert f"{result.report.safety.score:.2f}" in md
    assert "not guarantees" in md


def test_info_flags_are_not_quarantined():
    agg = aggregate([ScoredEvidence(
        analyzer="q", family="security", score=1.0, confidence=0.9, summary="s",
        flags=[{"row_index": 1, "reason": "extreme_outlier", "severity": Severity.INFO},
               {"row_index": 2, "reason": "empty_text", "severity": Severity.WARNING}])])
    assert [f.row_index for f in agg.quarantine] == [2]


def test_strong_fit_needs_confidence():
    low = [_ev("fit", 0.95, 0.4, analyzer=str(i)) for i in range(4)]
    assert "strong" not in aggregate(low).fit.label
    high = [_ev("fit", 0.95, 0.9, analyzer=str(i)) for i in range(4)]
    assert aggregate(high).fit.label == "strong fit"


def test_pass_names_low_checks_instead_of_repeating_scores():
    evidence_ = [_ev("security", 1.0, 0.9, analyzer="s"), _ev("fit", 0.95, 0.9, analyzer="a"),
                 _ev("fit", 0.9, 0.9, analyzer="b"),
                 _ev("fit", 0.1, 0.5, analyzer="shift", dimension="Distribution match")]
    decision = aggregate(evidence_).decision
    assert decision.verdict is Verdict.PASS
    assert any("Distribution match" in r and "not blocking" in r for r in decision.reasons)


def test_summaries_use_real_plurals(dirty):
    result, _ = dirty
    text = " ".join(e.summary + " ".join(e.blockers) for e in result.report.evidence)
    assert "(s)" not in text and "(es)" not in text


def test_strong_fit_is_qualified_when_a_fit_check_fails():
    evidence_ = [_ev("fit", 1.0, 0.9, analyzer=str(i)) for i in range(5)]
    evidence_.append(_ev("fit", 0.0, 0.9, analyzer="shift"))
    label = aggregate(evidence_).fit.label
    assert label != "strong fit" and "weak check" in label
