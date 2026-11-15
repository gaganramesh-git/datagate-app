"""Duplication scan: exact repeats, same-content repeats, near-duplicates, and
contradictory supervision (the same text carrying different labels).

Content-level checks apply to text data only. In low-cardinality tabular data
identical feature rows — and identical features with different outcomes — are
normal, so there only byte-identical full rows count as duplicates.
"""

from __future__ import annotations

import polars as pl

from datagate.analyzers.base import Analyzer
from datagate.context import AnalysisContext
from datagate.scoring import pct, risk_score, sample_indices
from datagate.textsim import build_lsh, minhash, normalized_expr
from datagate.types import Flag, ScoredEvidence, Severity

MAX_NEAR_DUP_ROWS = 20_000


class DuplicateAnalyzer(Analyzer):
    name = "duplicates"
    family = "security"
    dimension = "Duplication"

    def run(self, ctx: AnalysisContext) -> ScoredEvidence:
        ds = ctx.dataset
        df = ds.df
        n = df.height
        if n == 0:
            return self.skip("empty dataset")
        target = ctx.target

        reasons: dict[int, tuple[str, str | None]] = {}

        # 1) Byte-identical rows: every occurrence after the first.
        exact = (df.with_row_index("_idx")
                 .filter(pl.struct(pl.exclude("_idx")).is_first_distinct().not_())
                 ["_idx"].to_list())
        for i in exact:
            reasons[int(i)] = ("exact_duplicate", None)

        text_cols = ds.text_columns(exclude=[target] if target else [])
        n_content = n_near = n_conflict = 0
        if text_cols:
            keyed = (df.with_row_index("_idx")
                     .with_columns(normalized_expr(text_cols).alias("_key"))
                     .filter(pl.col("_key").str.len_chars() > 0))

            # 2) Same content after normalizing case, punctuation and spacing.
            repeats = keyed.filter(pl.col("_key").is_first_distinct().not_())
            for i in repeats["_idx"].to_list():
                if int(i) not in reasons:
                    reasons[int(i)] = ("duplicate_content", None)
                    n_content += 1

            # 3) Same content, different label: contradictory supervision.
            if target and ctx.task_kind == "classification":
                conflicted = (keyed.filter(pl.col(target).is_not_null())
                              .filter(pl.col(target).n_unique().over("_key") > 1))
                for i in conflicted["_idx"].to_list():
                    reasons[int(i)] = ("conflicting_label",
                                       "same text appears with a different label")
                    n_conflict += 1

            # 4) Near-duplicates (MinHash/LSH) among rows not already explained.
            fresh = keyed.filter(pl.col("_key").is_first_distinct())
            if fresh.height > MAX_NEAR_DUP_ROWS:
                fresh = fresh[sample_indices(fresh.height, MAX_NEAR_DUP_ROWS, ctx.seed)]
            lsh, _ = build_lsh([])
            for i, key in zip(fresh["_idx"].to_list(), fresh["_key"].to_list()):
                mh = minhash(key)
                if mh is None:
                    continue
                matches = lsh.query(mh)
                if matches and int(i) not in reasons:
                    reasons[int(i)] = ("near_duplicate", f"similar to row {min(matches)}")
                    n_near += 1
                lsh.insert(int(i), mh)

        flags = [Flag(row_index=i, reason=reason, severity=Severity.WARNING, detail=detail)
                 for i, (reason, detail) in sorted(reasons.items())]

        n_dup = len(exact) + n_content + n_near
        dup_rate, conflict_rate = n_dup / n, n_conflict / n

        # Low-cardinality tabular data repeats naturally; don't punish it.
        natural = (not text_cols and dup_rate > 0.01 and self._low_cardinality(ctx))
        if natural:
            return self.evidence(
                score=1.0, confidence=0.3,
                summary=(f"{len(exact):,} repeated rows, expected for low-cardinality "
                         "tabular data — not treated as duplication"),
                metrics={"exact_duplicates": len(exact), "duplicate_rate": dup_rate,
                         "treated_as_natural": True},
            )

        recommendations = []
        if n_dup:
            recommendations.append(
                f"Remove the {n_dup:,} duplicate rows ({pct(dup_rate)}) — they inflate "
                "apparent dataset size and leak between train and validation splits.")
        if n_conflict:
            recommendations.append(
                f"Resolve {n_conflict:,} rows where identical text carries different "
                "labels; contradictory supervision caps achievable accuracy.")

        return self.evidence(
            score=min(risk_score(dup_rate, ok=0.01, bad=0.30),
                      risk_score(conflict_rate, ok=0.0, bad=0.05)),
            confidence=0.9,
            summary=(f"{len(exact):,} exact, {n_content:,} same-content and {n_near:,} "
                     f"near-duplicate rows ({pct(dup_rate)} of {n:,})"
                     + (f"; {n_conflict:,} rows with conflicting labels" if n_conflict else "")),
            flags=flags,
            metrics={"exact_duplicates": len(exact), "content_duplicates": n_content,
                     "near_duplicates": n_near, "conflicting_label_rows": n_conflict,
                     "duplicate_rate": dup_rate},
            recommendations=recommendations,
        )

    @staticmethod
    def _low_cardinality(ctx: AnalysisContext) -> bool:
        df = ctx.dataset.df
        features = [c for c in df.columns if c != ctx.target]
        return len(features) <= 6 and all(df[c].n_unique() <= 20 for c in features)
