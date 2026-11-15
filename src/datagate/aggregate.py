"""Evidence Aggregator: analyzer outputs -> composite scores -> a gate decision.

Deterministic and reproducible.

**Safety is a weakest-link property.** Averaging hides the one analyzer that
found something serious, so the composite blends the confidence-weighted mean
with the single worst (confidence-adjusted) signal. Any *blocker* — secrets,
government IDs, a suspected backdoor trigger, invisible smuggled text — caps
the score below the block threshold outright: a dataset with credentials in it
is not "0.97 safe".

**Fit is an average.** Its dimensions trade off against each other, and it
never blocks on its own; a poor fit sends the dataset to review.

The gate then issues PASS / REVIEW / BLOCK with the reasons spelled out.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from datagate.types import (
    CompositeScore,
    Flag,
    GateDecision,
    GatePolicy,
    ProjectSpec,
    ScoredEvidence,
    Severity,
    Verdict,
)

_SAFETY_LABELS = [(0.9, "very likely safe"), (0.8, "likely safe"),
                  (0.5, "questionable — review flagged samples"),
                  (0.0, "high risk — do not train on this as-is")]
_FIT_LABELS = [(0.75, "strong fit"), (0.55, "moderate fit"),
               (0.35, "weak fit"), (0.0, "poor fit — unlikely to help")]

MAX_RECOMMENDATIONS = 8
FIT_BROKEN_CAP = 0.30
UNVERIFIED_FIT_CAP = 0.70
MIN_STRONG_CONFIDENCE = 0.60
NOTABLE_BELOW = 0.50   # a check this low is named even when the gate passes

REFERENCE_HINT = ("Add a sample of your own task data (ideally your evaluation set) to "
                  "unlock gap coverage, directional value, per-row data value and "
                  "marginal-gain estimates.")


@dataclass
class Aggregate:
    safety: CompositeScore
    fit: CompositeScore
    decision: GateDecision
    quarantine: list[Flag]                 # one per row, worst first, uncapped
    quarantine_by_reason: dict[str, int]
    recommendations: list[str]
    fit_breakdown: dict[str, float] = field(default_factory=dict)
    safety_breakdown: dict[str, float] = field(default_factory=dict)


def aggregate(evidence: list[ScoredEvidence], policy: GatePolicy | None = None,
              spec: ProjectSpec | None = None) -> Aggregate:
    policy = policy or GatePolicy()
    security = [e for e in evidence if e.family == "security"]
    fit_ev = [e for e in evidence if e.family == "fit"]
    blockers = [b for e in security for b in e.blockers]

    safety = _safety(security, blockers, policy)
    fit = _fit(fit_ev)
    quarantine, by_reason = _quarantine(security)
    decision = _decide(safety, fit, blockers, security, fit_ev, policy, spec)

    return Aggregate(
        safety=safety, fit=fit, decision=decision,
        quarantine=quarantine, quarantine_by_reason=by_reason,
        recommendations=_recommendations(evidence),
        fit_breakdown={e.dimension or e.analyzer: round(e.score, 3) for e in _active(fit_ev)},
        safety_breakdown={e.dimension or e.analyzer: round(e.score, 3)
                          for e in _active(security)},
    )


def _active(evidence: list[ScoredEvidence]) -> list[ScoredEvidence]:
    return [e for e in evidence if not e.skipped and e.confidence > 0]


def _label(score: float, labels: list[tuple[float, str]]) -> str:
    return next(text for threshold, text in labels if score >= threshold)


def _confidence(active: list[ScoredEvidence], total: int) -> float:
    mean_conf = sum(e.confidence for e in active) / len(active)
    breadth = len(active) / total if total else 1.0
    return round(mean_conf * (0.5 + 0.5 * breadth), 3)


def _safety(evidence: list[ScoredEvidence], blockers: list[str],
            policy: GatePolicy) -> CompositeScore:
    active = _active(evidence)
    if not active:
        return CompositeScore(score=0.5, confidence=0.0,
                              label="no analyzers ran — no verdict")
    total = sum(e.confidence for e in active)
    mean = sum(e.score * e.confidence for e in active) / total
    # A low-confidence analyzer can only pull the floor down in proportion.
    worst = min(1.0 - e.confidence * (1.0 - e.score) for e in active)
    score = 0.6 * mean + 0.4 * worst
    if blockers:
        score = min(score, policy.block_safety - 0.01)
        label = "critical findings — blocked"
    else:
        label = _label(score, _SAFETY_LABELS)
    return CompositeScore(score=round(score, 3),
                          confidence=_confidence(active, len(evidence)), label=label)


def _fit(evidence: list[ScoredEvidence]) -> CompositeScore:
    active = _active(evidence)
    if not active:
        return CompositeScore(score=0.5, confidence=0.0,
                              label="no analyzers ran — no verdict")
    total = sum(e.confidence for e in active)
    score = sum(e.score * e.confidence for e in active) / total
    # A fit-breaking finding (no usable target, labels that never match, a
    # learning direction opposed to the task) cannot be averaged away by the
    # dimensions that happen to look fine.
    if any(e.blockers for e in active):
        score = min(score, FIT_BROKEN_CAP)
    label = _label(score, _FIT_LABELS)
    confidence = _confidence(active, len(evidence))
    # "Strong" is a claim; it needs evidence behind it.
    if label == "strong fit" and confidence < MIN_STRONG_CONFIDENCE:
        label = "likely fit — low confidence"
    # Never a bare "strong" above a failing check on the same side.
    weak = sum(1 for e in active if e.score < NOTABLE_BELOW)
    if weak and label == "strong fit":
        label = f"strong overall — {'one weak check' if weak == 1 else f'{weak} weak checks'}"
    # Without the user's own task data we can say the dataset is well-formed and
    # learnable — not that it fits *their* task. Don't let that read as "strong".
    if any(e.needs_reference for e in evidence) and score > UNVERIFIED_FIT_CAP:
        score = UNVERIFIED_FIT_CAP
        label = "plausible fit — not verified against your task data"
    return CompositeScore(score=round(score, 3), confidence=confidence, label=label)


def _quarantine(security: list[ScoredEvidence]) -> tuple[list[Flag], dict[str, int]]:
    """Union of security flags at warning level or above, merged to one flag per row."""
    per_row: dict[int, list[Flag]] = {}
    for ev in security:
        for flag in ev.flags:
            if flag.severity.rank >= Severity.WARNING.rank:
                per_row.setdefault(flag.row_index, []).append(flag)

    merged: list[Flag] = []
    by_reason: Counter[str] = Counter()
    for row, flags in per_row.items():
        flags.sort(key=lambda f: -f.severity.rank)
        reasons = list(dict.fromkeys(f.reason for f in flags))
        for reason in {r.split(":")[0] for r in reasons}:
            by_reason[reason] += 1
        detail = next((f.detail for f in flags if f.detail), None)
        merged.append(Flag(row_index=row, reason="; ".join(reasons),
                           severity=flags[0].severity, detail=detail))
    merged.sort(key=lambda f: (-f.severity.rank, f.row_index))
    return merged, dict(by_reason.most_common())


def _decide(safety: CompositeScore, fit: CompositeScore, blockers: list[str],
            security: list[ScoredEvidence], fit_ev: list[ScoredEvidence],
            policy: GatePolicy, spec: ProjectSpec | None) -> GateDecision:
    def worst(evidence: list[ScoredEvidence], k: int) -> list[str]:
        ranked = sorted(_active(evidence), key=lambda e: e.score)
        return [f"{e.dimension or e.analyzer}: {e.summary}" for e in ranked[:k]
                if e.score < 0.8]

    if blockers:
        return GateDecision(verdict=Verdict.BLOCK,
                            headline="Blocked — do not train on this dataset as-is",
                            reasons=blockers)
    if safety.confidence > 0 and safety.score < policy.block_safety:
        return GateDecision(
            verdict=Verdict.BLOCK,
            headline="Blocked — do not train on this dataset as-is",
            reasons=[f"Safety score {safety.score:.2f} is below the block threshold "
                     f"({policy.block_safety:.2f})."] + worst(security, 3))

    reasons: list[str] = []
    if safety.confidence < policy.min_confidence:
        reasons.append("Too few safety checks could run to support a verdict.")
    elif safety.score < policy.pass_safety:
        reasons.append(f"Safety score {safety.score:.2f} is below the pass threshold "
                       f"({policy.pass_safety:.2f}).")
        reasons.extend(worst(security, 2))
    if fit.confidence > 0 and fit.score < policy.min_fit:
        reasons.append(f"Fit score {fit.score:.2f} — a weak match for the declared task.")
        reasons.extend(worst(fit_ev, 2))
    if spec is not None and spec.is_ambiguous:
        reasons.append("The task spec is ambiguous; fit was judged against a guess.")

    if reasons:
        return GateDecision(verdict=Verdict.REVIEW,
                            headline="Needs review before training", reasons=reasons)
    # A pass should still say what is worth a look, not just repeat the scores.
    notable = [e for e in _active(security + fit_ev) if e.score < NOTABLE_BELOW]
    notable.sort(key=lambda e: e.score)
    return GateDecision(
        verdict=Verdict.PASS,
        headline="Passed the gate — no blocking issues found",
        reasons=[f"Safety {safety.score:.2f} ({safety.label}), "
                 f"fit {fit.score:.2f} ({fit.label})."]
        + [f"Worth a look, not blocking — {e.dimension or e.analyzer}: {e.summary}"
           for e in notable[:3]])


def _recommendations(evidence: list[ScoredEvidence]) -> list[str]:
    ranked = sorted(_active(evidence), key=lambda e: (not e.blockers, e.score))
    out: list[str] = []
    for ev in ranked:
        for rec in ev.recommendations:
            if rec not in out:
                out.append(rec)
    out = out[:MAX_RECOMMENDATIONS]
    if any(e.needs_reference for e in evidence):
        out.append(REFERENCE_HINT)
    return out
