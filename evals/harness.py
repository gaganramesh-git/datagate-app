"""DataGate eval harness — how the "honesty" promise stays measurable.

Runs the engine against datasets with *known* planted problems and against
real, clean datasets, then checks:

  A. detection   recall per planted problem + false-flag rate on untouched rows
  B. calibration safety falls monotonically as label noise rises
  C. fit         on-task > off-topic; an inverted label convention reads as opposed
  D. tabular     mislabel recall, target-leak detection, no blockers on clean data
  E. leakage     reference rows hidden in the candidate are found
  F. real data   classic clean datasets are not blocked and barely flagged

Run:  python evals/harness.py            (exit code 1 on any failed check)
      python evals/harness.py --json out.json
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import polars as pl
from sklearn.datasets import load_breast_cancer, load_digits, load_wine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datagate import demo
from datagate.analyzers.security.provenance import ProvenanceAnalyzer
from datagate.orchestrator import analyze, default_analyzers
from datagate.types import Verdict

PURPOSE = "classify product reviews by sentiment"
TMP = Path(tempfile.mkdtemp(prefix="datagate-eval-"))
RESULTS: dict[str, dict] = {}
FAILURES: list[str] = []


def save(df: pl.DataFrame, name: str) -> str:
    path = TMP / name
    df.write_csv(path)
    return str(path)


def run(path: str, purpose: str = PURPOSE, **kwargs):
    # The provenance check reads a machine-local registry; keep evals hermetic.
    analyzers = [a for a in default_analyzers() if not isinstance(a, ProvenanceAnalyzer)]
    return analyze(path, purpose, analyzers=analyzers, **kwargs)


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def flagged_by(result, *reasons: str) -> set[int]:
    return {f.row_index for f in result.quarantine
            if any(r in f.reason for r in reasons)}


def recall(found: set[int], truth: set[int]) -> float:
    return len(found & truth) / len(truth) if truth else 1.0


def scenario_detection() -> None:
    print("\nA. Detection on text with planted problems")
    reference = save(demo.make_reviews(400, seed=8), "reference.csv")
    clean_df = demo.make_reviews(2000, seed=7)
    dirty_df, planted = demo.corrupt(clean_df, seed=7, backdoor=0.02,
                                     hidden_rows=10, injection_rows=10)
    clean = run(save(clean_df, "clean.csv"), target_column="label",
                reference_path=reference)
    dirty = run(save(dirty_df, "dirty.csv"), target_column="label",
                reference_path=reference)

    everything = {f.row_index for f in dirty.quarantine}
    recalls = {
        "mislabels": recall(flagged_by(dirty, "suspected_mislabel"), planted["mislabels"]),
        "duplicates": recall(flagged_by(dirty, "duplicate"), planted["duplicates"]),
        "pii": recall(flagged_by(dirty, "pii:"), planted["pii"]),
        "backdoor": recall(flagged_by(dirty, "backdoor_trigger"), planted["backdoor"]),
        "gibberish": recall(everything, planted["gibberish"]),
        "hidden": recall(flagged_by(dirty, "hidden:"), planted["hidden"]),
        "injection": recall(flagged_by(dirty, "hidden:"), planted["injection"]),
    }
    touched = set().union(*planted.values())
    untouched = dirty.dataset.n_rows - len(touched)
    false_flag = len(everything - touched) / untouched
    clean_flag = len(clean.quarantine) / clean.dataset.n_rows

    floors = {"mislabels": 0.85, "duplicates": 0.95, "pii": 0.99, "backdoor": 0.9,
              "gibberish": 0.9, "hidden": 0.99, "injection": 0.99}
    for name, value in recalls.items():
        check(f"recall {name}", value >= floors[name], f"{value:.1%} (floor {floors[name]:.0%})")
    check("false-flag rate (dirty)", false_flag <= 0.03, f"{false_flag:.2%} (ceiling 3%)")
    check("false-flag rate (clean)", clean_flag <= 0.01, f"{clean_flag:.2%} (ceiling 1%)")
    check("clean passes the gate", clean.report.decision.verdict is Verdict.PASS,
          f"{clean.report.decision.verdict.value}, safety {clean.report.safety.score:.2f}, "
          f"fit {clean.report.fit.score:.2f}")
    check("dirty is blocked", dirty.report.decision.verdict is Verdict.BLOCK,
          f"{dirty.report.decision.verdict.value}, safety {dirty.report.safety.score:.2f}")
    RESULTS["detection"] = {"recall": recalls, "false_flag_dirty": false_flag,
                            "false_flag_clean": clean_flag,
                            "clean_safety": clean.report.safety.score,
                            "dirty_safety": dirty.report.safety.score}


def scenario_calibration() -> None:
    print("\nB. Calibration — safety vs label noise")
    base = demo.make_reviews(2000, seed=21)
    scores, verdicts = [], []
    for noise in (0.0, 0.02, 0.05, 0.10, 0.20):
        df, _ = demo.corrupt(base, seed=21, label_noise=noise, duplicates=0.0,
                             pii_rows=0, gibberish_rows=0)
        r = run(save(df, f"noise_{noise}.csv"), target_column="label").report
        scores.append(r.safety.score)
        verdicts.append(r.decision.verdict.value)
        print(f"    noise {noise:>4.0%}  safety {r.safety.score:.3f}  {r.decision.verdict.value}")
    monotone = all(a >= b - 0.01 for a, b in zip(scores, scores[1:]))
    check("safety is monotone in noise", monotone, " > ".join(f"{s:.2f}" for s in scores))
    check("0% noise passes", verdicts[0] == "pass", verdicts[0])
    check("20% noise does not pass", verdicts[-1] != "pass", verdicts[-1])
    RESULTS["calibration"] = {"noise": [0, 0.02, 0.05, 0.10, 0.20], "safety": scores}


def scenario_fit() -> None:
    print("\nC. Fit — does it tell useful data from useless data?")
    reference = save(demo.make_reviews(400, seed=31), "fit_reference.csv")
    cases = {
        "on-task": demo.make_reviews(2000, seed=32),
        "off-topic": demo.make_offtopic(2000, seed=33),
        "inverted labels": demo.flip_labels(demo.make_reviews(2000, seed=34)),
        "positive only": demo.make_reviews(4000, seed=35).filter(
            pl.col("label") == "positive"),
    }
    fit, reports = {}, {}
    for name, df in cases.items():
        r = run(save(df, f"fit_{name.replace(' ', '_')}.csv"), target_column="label",
                reference_path=reference).report
        fit[name], reports[name] = r.fit.score, r
        print(f"    {name:<16} fit {r.fit.score:.3f} ({r.fit.label})")

    def metric(name: str, analyzer: str, key: str):
        ev = next(e for e in reports[name].evidence if e.analyzer == analyzer)
        return ev.metrics.get(key)

    check("on-task is a strong fit", fit["on-task"] >= 0.75, f"{fit['on-task']:.2f}")
    check("off-topic is a poor fit", fit["off-topic"] <= 0.40, f"{fit['off-topic']:.2f}")
    check("on-task beats off-topic by a wide margin",
          fit["on-task"] - fit["off-topic"] >= 0.35,
          f"{fit['on-task']:.2f} vs {fit['off-topic']:.2f}")
    check("inverted labels read as opposed",
          metric("inverted labels", "alignment", "direction") == "opposed",
          f"direction={metric('inverted labels', 'alignment', 'direction')}, "
          f"cosine={metric('inverted labels', 'alignment', 'cosine'):+.2f}")
    check("inverted labels are a poor fit", fit["inverted labels"] <= 0.35,
          f"{fit['inverted labels']:.2f}")
    cov_full = metric("on-task", "coverage", "coverage")
    cov_narrow = metric("positive only", "coverage", "coverage")
    check("one-sided data leaves a coverage gap", cov_narrow < cov_full - 0.2,
          f"coverage {cov_narrow:.2f} vs {cov_full:.2f}")
    RESULTS["fit"] = fit


def scenario_tabular() -> None:
    print("\nD. Tabular (numeric + categorical)")
    purpose = "predict whether a customer will churn"
    clean_df, _ = demo.make_tabular(2000, seed=41)
    noisy_df, planted = demo.make_tabular(2000, seed=41, label_noise=0.08)
    leaky_df, _ = demo.make_tabular(2000, seed=41, leak=True)

    clean = run(save(clean_df, "tab_clean.csv"), purpose, target_column="churned")
    noisy = run(save(noisy_df, "tab_noisy.csv"), purpose, target_column="churned")
    leaky = run(save(leaky_df, "tab_leaky.csv"), purpose, target_column="churned")

    found = flagged_by(noisy, "suspected_mislabel")
    rec = recall(found, planted["mislabels"])
    precision = len(found & planted["mislabels"]) / len(found) if found else 0.0
    leak_ev = next(e for e in leaky.report.evidence if e.analyzer == "target_leakage")
    suspects = [s["column"] for s in leak_ev.metrics.get("suspects", [])]
    clean_flag = len(clean.quarantine) / clean.dataset.n_rows

    check("clean table is not blocked", clean.report.decision.verdict is not Verdict.BLOCK,
          f"{clean.report.decision.verdict.value}, safety {clean.report.safety.score:.2f}")
    check("clean table false-flag rate", clean_flag <= 0.06, f"{clean_flag:.2%} (ceiling 6%)")
    check("tabular mislabel recall", rec >= 0.5, f"{rec:.1%} (floor 50%)")
    check("tabular mislabel precision", precision >= 0.5, f"{precision:.1%} (floor 50%)")
    check("noisy table scores lower", noisy.report.safety.score < clean.report.safety.score,
          f"{noisy.report.safety.score:.2f} < {clean.report.safety.score:.2f}")
    check("leaked column is named", suspects == ["refund_issued"], f"suspects={suspects}")
    RESULTS["tabular"] = {"mislabel_recall": rec, "mislabel_precision": precision,
                          "clean_false_flag": clean_flag, "leak_suspects": suspects}


def scenario_leakage() -> None:
    print("\nE. Eval contamination")
    reference_df = demo.make_reviews(400, seed=51)
    candidate = pl.concat([demo.make_reviews(1500, seed=52), reference_df.head(100)])
    r = run(save(candidate, "contaminated.csv"), target_column="label",
            reference_path=save(reference_df, "contam_reference.csv"))
    ev = next(e for e in r.report.evidence if e.analyzer == "contamination")
    hit = ev.metrics["reference_rows_overlapping"]
    planted = set(range(1500, 1600))
    rec = recall(flagged_by(r, "overlaps_reference"), planted)
    check("leaked eval rows are found", rec >= 0.99, f"{rec:.1%} of 100 planted rows")
    check("overlap count is accurate", 100 <= hit <= 110, f"{hit} reference rows reported")
    check("contaminated data does not pass", r.report.decision.verdict is not Verdict.PASS,
          r.report.decision.verdict.value)
    RESULTS["leakage"] = {"recall": rec, "reference_rows_overlapping": hit}


def scenario_real_data() -> None:
    print("\nF. Real, clean datasets (false-positive check)")
    out = {}
    for name, loader in (("breast_cancer", load_breast_cancer), ("wine", load_wine),
                         ("digits", load_digits)):
        bunch = loader()
        names = [str(f).replace(" ", "_") for f in bunch.feature_names]
        df = pl.DataFrame(bunch.data, schema=names).with_columns(
            pl.Series("target", [str(bunch.target_names[t]) for t in bunch.target]))
        r = run(save(df, f"real_{name}.csv"), "classify samples into the target class",
                target_column="target")
        rate = len(r.quarantine) / r.dataset.n_rows
        out[name] = {"verdict": r.report.decision.verdict.value,
                     "safety": r.report.safety.score, "flag_rate": rate}
        check(f"{name} is not blocked", r.report.decision.verdict is not Verdict.BLOCK,
              f"{r.report.decision.verdict.value}, safety {r.report.safety.score:.2f}")
        check(f"{name} flag rate", rate <= 0.08, f"{rate:.2%} of rows (ceiling 8%)")
    RESULTS["real_data"] = out


def scenario_real_text() -> None:
    """Real prose is heavy-tailed and full of boilerplate — where heuristics over-fire."""
    print("\nG. Real text corpus (20 Newsgroups, downloaded on first use)")
    try:
        from sklearn.datasets import fetch_20newsgroups

        bunch = fetch_20newsgroups(
            subset="train", remove=("headers", "footers", "quotes"),
            categories=["rec.autos", "sci.med", "comp.graphics", "talk.politics.mideast"])
    except Exception as exc:
        print(f"  [skip] corpus unavailable ({type(exc).__name__}); scenario not run")
        RESULTS["real_text"] = {"skipped": True}
        return
    df = pl.DataFrame({"text": [t.replace("\x00", " ") for t in bunch.data],
                       "label": [bunch.target_names[i] for i in bunch.target]})
    df = df.filter(pl.col("text").str.strip_chars().str.len_chars() > 0)
    r = run(save(df, "newsgroups.csv"), "classify newsgroup posts by topic",
            target_column="label")
    poisoning = next(e for e in r.report.evidence if e.analyzer == "poisoning")
    outlier_rate = len(flagged_by(r, "class_outlier")) / r.dataset.n_rows
    mislabel_rate = len(flagged_by(r, "suspected_mislabel")) / r.dataset.n_rows
    cards = next(e for e in r.report.evidence if e.analyzer == "pii").metrics["counts"]
    check("no false backdoor triggers", poisoning.metrics["triggers"] == [],
          f"triggers={poisoning.metrics['triggers']}")
    check("class-outlier rate on clean prose", outlier_rate <= 0.01,
          f"{outlier_rate:.2%} (ceiling 1%)")
    check("suspected-mislabel rate on clean prose", mislabel_rate <= 0.03,
          f"{mislabel_rate:.2%} (ceiling 3%)")
    check("no phantom payment cards", cards["credit_card"] == 0,
          f"{cards['credit_card']} card(s) reported")
    RESULTS["real_text"] = {"rows": r.dataset.n_rows, "outlier_rate": outlier_rate,
                            "mislabel_rate": mislabel_rate,
                            "safety": r.report.safety.score,
                            "verdict": r.report.decision.verdict.value}


def main() -> int:
    print("== DataGate eval harness ==")
    for scenario in (scenario_detection, scenario_calibration, scenario_fit,
                     scenario_tabular, scenario_leakage, scenario_real_data,
                     scenario_real_text):
        scenario()

    if "--json" in sys.argv:
        target = sys.argv[sys.argv.index("--json") + 1]
        Path(target).write_text(json.dumps(RESULTS, indent=2, default=str))
        print(f"\nresults written to {target}")

    print(f"\n{'PASS' if not FAILURES else 'FAIL'} — {len(FAILURES)} failed check(s)")
    for failure in FAILURES:
        print(f"  - {failure}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
