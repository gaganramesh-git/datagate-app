"""Eval contamination: do rows of the user's reference data appear in the candidate?

If the reference is a held-out evaluation set, training on overlapping rows
silently inflates every metric measured afterwards. Overlapping candidate rows
are quarantined, so the cleaned export is de-contaminated automatically.

Matching is on content (case-, punctuation- and whitespace-insensitive), with a
MinHash pass for near-identical text.
"""

from __future__ import annotations

import polars as pl

from datagate.analyzers.base import Analyzer
from datagate.context import AnalysisContext
from datagate.scoring import pct, risk_score, sample_indices
from datagate.textsim import build_lsh, minhash, normalized_expr
from datagate.types import Flag, ScoredEvidence, Severity

MAX_NEAR_REFERENCE = 5_000
MAX_NEAR_CANDIDATE = 20_000


class ContaminationAnalyzer(Analyzer):
    name = "contamination"
    family = "security"
    dimension = "Eval contamination"
    needs_reference = True

    def run(self, ctx: AnalysisContext) -> ScoredEvidence:
        ref = ctx.reference
        if ref is None:
            return self.skip_no_reference()
        cand = ctx.dataset
        target = ctx.target
        skip = [target] if target else []

        cand_text, ref_text = cand.text_columns(skip), ref.text_columns(skip)
        is_text = bool(cand_text and ref_text)
        if is_text:
            cand_cols, ref_cols = cand_text, ref_text
        else:
            shared = [c for c in cand.df.columns if c in ref.df.columns and c not in skip]
            if not shared:
                return self.skip("candidate and reference share no comparable columns")
            cand_cols = ref_cols = shared

        cand_keys = (cand.df.with_row_index("_idx")
                     .select("_idx", normalized_expr(cand_cols).alias("_key"))
                     .filter(pl.col("_key").str.len_chars() > 0))
        ref_keys = (ref.df.with_row_index("_ridx")
                    .select("_ridx", normalized_expr(ref_cols).alias("_key"))
                    .filter(pl.col("_key").str.len_chars() > 0))
        if ref_keys.height == 0:
            return self.skip("reference has no comparable content")

        exact = cand_keys.join(ref_keys.select("_key").unique(), on="_key", how="semi")
        ref_hit = ref_keys.join(cand_keys.select("_key").unique(), on="_key", how="semi")
        reasons: dict[int, str] = {int(i): "overlaps_reference"
                                   for i in exact["_idx"].to_list()}
        ref_overlap = set(ref_hit["_ridx"].to_list())

        n_near = 0
        if is_text:
            ref_rest = ref_keys.filter(~pl.col("_ridx").is_in(list(ref_overlap)))
            if ref_rest.height > MAX_NEAR_REFERENCE:
                ref_rest = ref_rest[sample_indices(ref_rest.height, MAX_NEAR_REFERENCE,
                                                   ctx.seed)]
            lsh, _ = build_lsh(zip(ref_rest["_ridx"].to_list(), ref_rest["_key"].to_list()))
            cand_rest = cand_keys.filter(~pl.col("_idx").is_in(list(reasons)))
            if cand_rest.height > MAX_NEAR_CANDIDATE:
                cand_rest = cand_rest[sample_indices(cand_rest.height, MAX_NEAR_CANDIDATE,
                                                     ctx.seed)]
            for i, key in zip(cand_rest["_idx"].to_list(), cand_rest["_key"].to_list()):
                mh = minhash(key)
                if mh is None:
                    continue
                matches = lsh.query(mh)
                if matches:
                    reasons[int(i)] = "near_overlaps_reference"
                    ref_overlap.update(matches)
                    n_near += 1

        flags = [Flag(row_index=i, reason=reason, severity=Severity.WARNING,
                      detail="also present in your reference data")
                 for i, reason in sorted(reasons.items())]

        ref_fraction = len(ref_overlap) / ref_keys.height
        recommendations = []
        if reasons:
            recommendations.append(
                f"{len(reasons):,} candidate rows also appear in your reference data. If "
                "that data is your evaluation set, training on them inflates your scores "
                "— the cleaned export removes them.")

        return self.evidence(
            score=risk_score(ref_fraction, ok=0.0, bad=0.25),
            confidence=0.9,
            summary=(f"{len(ref_overlap):,} of {ref_keys.height:,} reference rows "
                     f"({pct(ref_fraction)}) also appear in the candidate "
                     f"({len(reasons):,} candidate rows, {n_near:,} near-identical)"
                     if reasons else
                     f"no overlap between the candidate and {ref_keys.height:,} reference rows"),
            flags=flags,
            metrics={"reference_rows": ref_keys.height,
                     "reference_rows_overlapping": len(ref_overlap),
                     "reference_overlap_fraction": ref_fraction,
                     "candidate_rows_overlapping": len(reasons),
                     "near_matches": n_near},
            recommendations=recommendations,
        )
