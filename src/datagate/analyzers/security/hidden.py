"""Hidden-content scan: text that reads differently to a model than to a human.

Catches the cheap, effective ways to smuggle behaviour into training data:
- bidirectional-override and Unicode *tag* characters (render as nothing, or
  reorder what a reviewer sees — "Trojan Source" / ASCII smuggling),
- zero-width and control characters that split or hide tokens,
- prompt-injection phrases and chat-template control tokens,
- active content (script tags, javascript: URLs),
- long opaque encoded blobs.

Zero-width joiner/non-joiner (U+200C/U+200D) are deliberately *not* flagged:
they are legitimate in emoji sequences and in Arabic and Indic scripts.
"""

from __future__ import annotations

import polars as pl

from datagate.analyzers.base import Analyzer
from datagate.context import AnalysisContext
from datagate.scoring import pct, risk_score
from datagate.types import Flag, ScoredEvidence, Severity

PATTERNS: dict[str, tuple[str, Severity]] = {
    "bidi_control": (r"[\x{202A}-\x{202E}\x{2066}-\x{2069}]", Severity.CRITICAL),
    "unicode_tag": (r"[\x{E0000}-\x{E007F}]", Severity.CRITICAL),
    "zero_width": (r"[\x{200B}\x{2060}\x{FEFF}\x{180E}]", Severity.WARNING),
    "control_char": (r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", Severity.WARNING),
    "prompt_injection": (
        r"(?i)(ignore|disregard|forget)\s+(all\s+|any\s+)?(of\s+)?(the\s+|your\s+)?"
        r"(previous|prior|above|earlier)\s+(instructions?|prompts?|messages?|context)"
        r"|reveal\s+(your|the)\s+system\s+prompt"
        r"|you\s+are\s+now\s+(dan\b|in\s+developer\s+mode)"
        r"|<\|im_start\|>|<\|endoftext\|>|<<SYS>>|\[/?INST\]",
        Severity.WARNING),
    "active_content": (
        r"(?i)<script\b|javascript:|<iframe\b|\bonerror\s*=|\bonload\s*=", Severity.WARNING),
    "encoded_blob": (r"[A-Za-z0-9+/]{240,}={0,2}", Severity.INFO),
}

_LABELS = {
    "bidi_control": "bidirectional-override characters",
    "unicode_tag": "invisible Unicode tag characters",
    "zero_width": "zero-width characters",
    "control_char": "control characters",
    "prompt_injection": "prompt-injection phrases or chat control tokens",
    "active_content": "active web content (script/iframe/event handlers)",
    "encoded_blob": "long opaque encoded blobs",
}


class HiddenContentAnalyzer(Analyzer):
    name = "hidden_content"
    family = "security"
    dimension = "Hidden content"

    def run(self, ctx: AnalysisContext) -> ScoredEvidence:
        df = ctx.dataset.df
        cols = ctx.dataset.string_columns()
        if not cols or df.height == 0:
            return self.skip("no text columns to scan")

        kinds = list(PATTERNS)
        hits = (
            df.with_row_index("_idx")
            .select(["_idx", *[
                pl.any_horizontal([
                    pl.col(c).str.contains(PATTERNS[k][0]).fill_null(False) for c in cols
                ]).alias(k)
                for k in kinds
            ]])
            .filter(pl.any_horizontal([pl.col(k) for k in kinds]))
        )

        counts = {k: int(hits[k].sum()) for k in kinds}
        flags: list[Flag] = []
        n_critical = n_warning = 0
        for row in hits.iter_rows(named=True):
            found = [k for k in kinds if row[k]]
            worst = max((PATTERNS[k][1] for k in found), key=lambda s: s.rank)
            if worst is Severity.CRITICAL:
                n_critical += 1
            elif worst is Severity.WARNING:
                n_warning += 1
            flags.append(Flag(row_index=int(row["_idx"]),
                              reason="hidden:" + "+".join(found), severity=worst))
        flags.sort(key=lambda f: (-f.severity.rank, f.row_index))

        n = df.height
        score = min(risk_score(n_critical / n, ok=0.0, bad=0.005),
                    risk_score(n_warning / n, ok=0.001, bad=0.05))

        blockers, recommendations = [], []
        critical_kinds = [k for k in ("bidi_control", "unicode_tag") if counts[k]]
        if critical_kinds:
            what = " and ".join(_LABELS[k] for k in critical_kinds)
            blockers.append(
                f"{n_critical:,} rows contain {what} — text a reviewer sees differently "
                "from what a model reads.")
        if n_critical or n_warning:
            recommendations.append(
                "Strip invisible/control characters and review rows with injection "
                "phrases or active content; the cleaned export drops the affected rows.")

        found_text = ", ".join(f"{_LABELS[k]} ({counts[k]:,})" for k in kinds if counts[k])
        return self.evidence(
            score=score, confidence=0.85,
            summary=(f"hidden or adversarial content in {n_critical + n_warning:,} of "
                     f"{n:,} rows ({pct((n_critical + n_warning) / n)}): {found_text}"
                     if found_text else f"no hidden or adversarial content in {n:,} rows"),
            flags=flags,
            metrics={"counts": counts, "scanned_rows": n,
                     "critical_rows": n_critical, "warning_rows": n_warning},
            blockers=blockers, recommendations=recommendations,
        )
