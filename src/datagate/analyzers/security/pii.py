"""PII & secrets scan over every string column.

Matching runs inside polars (Rust regex), so it scales to the full dataset
instead of a sample. Credit-card candidates are confirmed with the Luhn
checksum to keep order numbers and IDs from being reported as cards.

Matched values are never echoed into the report — only kinds and row indices.
Presidio can replace the pattern tier later behind the same analyzer.
"""

from __future__ import annotations

import re

import polars as pl

from datagate.analyzers.base import Analyzer
from datagate.context import AnalysisContext
from datagate.scoring import pct, risk_score
from datagate.types import Flag, ScoredEvidence, Severity

# Patterns are written to be valid for both Rust regex (polars) and Python re.
PATTERNS: dict[str, tuple[str, Severity, str]] = {
    "email": (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
              Severity.WARNING, "[EMAIL]"),
    "phone": (r"(?:\+?\d{1,3}[-.\s])?\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b",
              Severity.WARNING, "[PHONE]"),
    "ssn": (r"\b\d{3}-\d{2}-\d{4}\b", Severity.CRITICAL, "[SSN]"),
    # Card-shaped only: a plausible issuer digit, then 4-4-4-N or Amex 4-6-5
    # grouping (or unbroken digits). A ZIP code next to a phone number is not a card.
    "credit_card": (r"\b[2-6]\d{3}[ -]?\d{4}[ -]?\d{4}[ -]?\d{1,7}\b"
                    r"|\b3[47]\d{2}[ -]?\d{6}[ -]?\d{5}\b",
                    Severity.CRITICAL, "[CARD]"),
    "ipv4": (r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b",
             Severity.INFO, "[IP]"),
    "aws_access_key": (r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b", Severity.CRITICAL, "[SECRET]"),
    "private_key": (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", Severity.CRITICAL, "[SECRET]"),
    "api_token": (r"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}"
                  r"|xox[baprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{30,})",
                  Severity.CRITICAL, "[SECRET]"),
    "jwt": (r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}",
            Severity.CRITICAL, "[SECRET]"),
    "secret_assignment": (
        r"(?i)\b(?:api[_-]?key|secret|token|passwd|password)\b\s*[:=]\s*['\"]?[^\s'\"]{8,}",
        Severity.CRITICAL, "[SECRET]"),
}

_COMPILED = {kind: re.compile(pattern) for kind, (pattern, _, _) in PATTERNS.items()}

_PII_COLUMN_NAME = re.compile(
    r"(?i)(e-?mail|phone|mobile|ssn|social_?sec|passport|date_?of_?birth|\bdob\b|"
    r"address|first_?name|last_?name|full_?name|surname|credit_?card|card_?number|"
    r"iban|password)")

_LABELS = {
    "email": "email addresses", "phone": "phone numbers", "ssn": "SSN-shaped IDs",
    "credit_card": "payment card numbers", "ipv4": "IP addresses",
    "aws_access_key": "AWS access keys", "private_key": "private keys",
    "api_token": "API tokens", "jwt": "JWTs", "secret_assignment": "inline secrets",
}


def luhn_valid(number: str) -> bool:
    digits = [int(ch) for ch in number if ch.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d = d * 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def redact_text(text: str) -> str:
    """Replace every detected PII/secret span with a typed placeholder."""
    for kind, (_, _, placeholder) in PATTERNS.items():
        if kind == "credit_card":
            text = _COMPILED[kind].sub(
                lambda m: placeholder if luhn_valid(m.group(0)) else m.group(0), text)
        else:
            text = _COMPILED[kind].sub(placeholder, text)
    return text


class PIIAnalyzer(Analyzer):
    name = "pii"
    family = "security"
    dimension = "PII & secrets"

    def run(self, ctx: AnalysisContext) -> ScoredEvidence:
        ds = ctx.dataset
        df = ds.df
        cols = ds.string_columns()
        n = df.height
        if not cols or n == 0:
            return self.skip("no text columns to scan")

        kinds = list(PATTERNS)
        hits = (
            df.with_row_index("_idx")
            .select(["_idx", *cols, *[
                pl.any_horizontal([
                    pl.col(c).str.contains(PATTERNS[k][0]).fill_null(False) for c in cols
                ]).alias(f"_hit_{k}")
                for k in kinds
            ]])
            .filter(pl.any_horizontal([pl.col(f"_hit_{k}") for k in kinds]))
        )

        counts = {k: 0 for k in kinds}
        flags: list[Flag] = []
        n_critical = n_sensitive = 0
        card = _COMPILED["credit_card"]
        for row in hits.iter_rows(named=True):
            found = [k for k in kinds if row[f"_hit_{k}"]]
            if "credit_card" in found:
                # Confirm with the Luhn checksum; drop the kind if nothing validates.
                texts = [row[c] for c in cols if row[c]]
                if not any(luhn_valid(m.group(0)) for t in texts for m in card.finditer(t)):
                    found.remove("credit_card")
            if not found:
                continue
            for k in found:
                counts[k] += 1
            worst = max((PATTERNS[k][1] for k in found), key=lambda s: s.rank)
            n_critical += worst is Severity.CRITICAL
            n_sensitive += worst is Severity.WARNING
            flags.append(Flag(row_index=int(row["_idx"]), reason="pii:" + "+".join(found),
                              severity=worst))
        flags.sort(key=lambda f: (-f.severity.rank, f.row_index))

        named_columns = [c for c in df.columns if _PII_COLUMN_NAME.search(c)]
        critical_kinds = [k for k in kinds
                          if counts[k] and PATTERNS[k][1] is Severity.CRITICAL]

        score = min(risk_score(n_critical / n, ok=0.0, bad=0.005),
                    risk_score(n_sensitive / n, ok=0.001, bad=0.20))

        blockers, recommendations = [], []
        if critical_kinds:
            blockers.append(
                f"{n_critical:,} rows contain {', '.join(_LABELS[k] for k in critical_kinds)} "
                "— credentials and government/payment identifiers must not enter training.")
        if n_critical or n_sensitive:
            recommendations.append(
                f"Redact or drop the {n_critical + n_sensitive:,} rows containing personal "
                "data or secrets (the cleaned export can mask them in place instead).")
        if named_columns:
            recommendations.append(
                "Columns named like personal data ("
                + ", ".join(repr(c) for c in named_columns[:5])
                + ") — confirm you may train on them, or drop them.")

        found_text = ", ".join(f"{_LABELS[k]} ({counts[k]:,})" for k in kinds if counts[k])
        return self.evidence(
            score=score, confidence=0.8,
            summary=(f"personal data or secrets in {len(flags):,} of {n:,} rows "
                     f"({pct(len(flags) / n)}): {found_text}"
                     if flags else f"no personal data or secrets detected in {n:,} rows"),
            flags=flags,
            metrics={"counts": counts, "scanned_rows": n, "critical_rows": n_critical,
                     "sensitive_rows": n_sensitive, "pii_named_columns": named_columns},
            blockers=blockers, recommendations=recommendations,
        )
