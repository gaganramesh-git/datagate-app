"""Fit: does the dataset structurally match the declared task?

Modality, target sanity, class balance and support, volume, and — when the
user supplies their own task data — whether the label vocabulary matches.
Cheap, deterministic, runs on every dataset.
"""

from __future__ import annotations

import math

import polars as pl

from datagate.analyzers.base import Analyzer
from datagate.context import AnalysisContext
from datagate.types import Modality, ScoredEvidence, TaskType

# Rows that are "usually enough" to start learning, per task type.
_VOLUME_TARGETS = {
    TaskType.CLASSIFICATION: 1_000,
    TaskType.REGRESSION: 1_000,
    TaskType.GENERATION: 5_000,
    TaskType.RETRIEVAL: 2_000,
    TaskType.UNKNOWN: 1_000,
}


class StructureAnalyzer(Analyzer):
    name = "structure"
    family = "fit"
    dimension = "Task structure"

    def run(self, ctx: AnalysisContext) -> ScoredEvidence:
        ds, spec = ctx.dataset, ctx.spec
        df = ds.df
        n = df.height
        if n == 0:
            return self.skip("empty dataset")

        target = ctx.target
        exclude = [target] if target else []
        has_text = bool(ds.text_columns(exclude))
        has_struct = bool(ds.numeric_columns(exclude) or ds.categorical_columns(exclude))

        parts: dict[str, float] = {}
        notes: list[str] = []
        recs: list[str] = []
        fatal = False

        # Modality.
        if spec.modality is Modality.TEXT:
            parts["modality"] = 1.0 if has_text else 0.1
        elif spec.modality is Modality.TABULAR:
            parts["modality"] = 1.0 if has_struct else 0.2
        elif spec.modality is Modality.MIXED:
            parts["modality"] = 1.0 if (has_text and has_struct) else 0.6
        elif spec.modality is Modality.IMAGE:
            parts["modality"] = 0.3
            notes.append("image content is not analyzed yet — only the tabular/text "
                         "columns were checked")
        else:
            parts["modality"] = 0.7
        if parts["modality"] < 0.5 and spec.modality is not Modality.IMAGE:
            notes.append(f"the data does not look like {spec.modality.value} data")

        # Target.
        task = spec.task_type
        if task in (TaskType.CLASSIFICATION, TaskType.REGRESSION):
            if target is None:
                parts["target"] = 0.0
                fatal = True
                notes.append("no target column was found for a prediction task")
                recs.append("Pick the target/label column in the task spec.")
            elif task is TaskType.CLASSIFICATION:
                fatal = self._classification(df, target, n, parts, notes, recs)
            else:
                series = df[target]
                if not series.dtype.is_numeric():
                    parts["target"] = 0.1
                    fatal = True
                    notes.append(f"regression target {target!r} is not numeric")
                else:
                    parts["target"] = 1.0
                    if series.n_unique() <= 10:
                        notes.append(f"target {target!r} has only {series.n_unique()} "
                                     "distinct values — is this really regression?")
        elif task in (TaskType.GENERATION, TaskType.RETRIEVAL):
            text_cols = ds.text_columns()
            if not text_cols:
                parts["text_content"] = 0.1
                fatal = True
                notes.append(f"{task.value} needs free-text columns; none were found")
            else:
                mean_len = max(float(df[c].str.len_chars().mean() or 0) for c in text_cols)
                parts["text_content"] = 1.0 if mean_len >= 40 else 0.5
                if mean_len < 40:
                    notes.append(f"texts are very short (≈{mean_len:.0f} characters)")

        # Volume, log-scaled toward a "usually enough" row count.
        want = _VOLUME_TARGETS[task]
        parts["volume"] = min(1.0, math.log1p(n) / math.log1p(want))
        if n < want * 0.1:
            notes.append(f"only {n:,} rows; {task.value} tasks typically want ~{want:,}+")

        # Label vocabulary vs the user's own task data.
        ref = ctx.reference
        if (ref is not None and target and target in ref.df.columns
                and ctx.task_kind == "classification"):
            ref_labels = set(ref.df[target].drop_nulls().cast(pl.String).unique().to_list())
            cand_labels = set(df[target].drop_nulls().cast(pl.String).unique().to_list())
            if ref_labels:
                shared = ref_labels & cand_labels
                parts["label_match"] = len(shared) / len(ref_labels)
                missing = sorted(ref_labels - cand_labels)
                if not shared:
                    fatal = True
                    notes.append("the candidate's labels never match your task's labels")
                    recs.append("Map the candidate's label names onto your label set "
                                "before using it.")
                elif missing:
                    notes.append("labels in your task data that the candidate lacks: "
                                 + ", ".join(repr(m) for m in missing[:5]))

        score = sum(parts.values()) / len(parts)
        if fatal:
            score = min(score, 0.25)
        return self.evidence(
            score=score, confidence=0.85,
            summary="; ".join(notes) if notes else "structure matches the declared task",
            metrics={"components": {k: round(v, 3) for k, v in parts.items()}},
            recommendations=recs,
            blockers=["; ".join(notes)] if fatal else [],
        )

    @staticmethod
    def _classification(df: pl.DataFrame, target: str, n: int, parts: dict[str, float],
                        notes: list[str], recs: list[str]) -> bool:
        counts = df[target].drop_nulls().cast(pl.String).value_counts()["count"].to_list()
        k = len(counts)
        if k < 2:
            parts["target"] = 0.0
            notes.append(f"target {target!r} has a single class — nothing to learn")
            return True
        if k > max(100, 0.5 * n):
            parts["target"] = 0.2
            notes.append(f"target {target!r} has {k:,} distinct values for {n:,} rows — "
                         "this does not look like a classification label")
            return True

        parts["target"] = 1.0
        total = sum(counts)
        freqs = [c / total for c in counts]
        balance = -sum(f * math.log(f) for f in freqs if f > 0) / math.log(k)
        parts["class_balance"] = 0.4 + 0.6 * balance
        if balance < 0.5:
            notes.append(f"classes are highly imbalanced (balance {balance:.2f}; "
                         f"smallest class {min(counts):,} rows)")
            recs.append("Rebalance or reweight classes; the rarest class is badly "
                        "under-represented.")
        thin = sum(1 for c in counts if c < 10)
        parts["class_support"] = 1.0 - thin / k
        if thin:
            notes.append(f"{thin} of {k} classes have fewer than 10 examples")
        return False
