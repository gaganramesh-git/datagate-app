"""Task-Spec Compiler: project brief + training objective -> ProjectSpec.

Design doc calls this the highest-risk component, so the contract is defensive:
- Two inputs. ``objective`` ("what do you want to train?") drives task/target;
  ``project_brief`` (the idea, the area of use) mainly sharpens domain,
  deployment context, and success metric.
- Heuristic extraction is always the baseline, and the data itself is a
  witness: a categorical target implies classification even when the
  description never says so.
- An optional LLM backend (see ``spec/llm.py``) can fill fields the heuristics
  miss — it is boxed to parsing intent and never overrides a target the user
  set explicitly.
- When intent is still ambiguous, we attach clarifying questions instead of
  guessing silently; the UI surfaces the parsed spec for user correction.
"""

from __future__ import annotations

import re
from typing import Optional

from datagate.ingest.loader import KIND_CATEGORICAL, KIND_NUMERIC, Dataset
from datagate.spec import llm as llm_backend
from datagate.types import Modality, ProjectSpec, TaskType

_TASK_KEYWORDS: list[tuple[TaskType, list[str]]] = [
    (TaskType.CLASSIFICATION, ["classif", "detect", "categor", "label", "predict whether",
                               "sentiment", "spam", "fraud", "identify", "churn"]),
    (TaskType.REGRESSION, ["regress", "forecast", "predict the price", "predict price",
                           "estimate", "predict how much", "predict the value"]),
    (TaskType.GENERATION, ["generat", "chatbot", "summariz", "translat", "llm",
                           "fine-tune", "finetune", "instruction"]),
    (TaskType.RETRIEVAL, ["retriev", "search", "rank", "recommend", "rag"]),
]

_MODALITY_KEYWORDS: list[tuple[Modality, list[str]]] = [
    (Modality.IMAGE, ["image", "photo", "picture", "vision", "visual"]),
    (Modality.TEXT, ["text", "review", "comment", "document", "sentence", "nlp",
                     "language", "email", "tweet", "article"]),
    (Modality.TABULAR, ["tabular", "csv", "spreadsheet", "table", "features", "columns"]),
]

# Domain is drawn mostly from the project brief — the wider area of use.
_DOMAIN_KEYWORDS: list[tuple[str, list[str]]] = [
    ("healthcare", ["medical", "health", "clinical", "patient", "diagnos", "disease"]),
    ("finance", ["financ", "bank", "credit", "loan", "invest", "trading", "fraud"]),
    ("legal", ["legal", "law", "contract", "litigation", "compliance"]),
    ("e-commerce", ["e-commerce", "ecommerce", "retail", "product review", "shopping",
                    "checkout", "marketplace"]),
    ("social media", ["social media", "tweet", "post", "comment moderation", "toxic"]),
    ("cybersecurity", ["malware", "intrusion", "phishing", "security log", "threat"]),
    ("education", ["education", "student", "learning", "grading", "exam"]),
    ("customer support", ["support ticket", "helpdesk", "customer service", "chatbot"]),
]

_DEPLOYMENT_KEYWORDS: list[tuple[str, list[str]]] = [
    ("real-time production", ["real-time", "real time", "low latency", "online serving",
                              "production api", "live"]),
    ("batch production", ["batch", "offline scoring", "nightly", "pipeline"]),
    ("edge / mobile", ["edge", "on-device", "mobile", "embedded"]),
    ("research / prototype", ["research", "prototype", "experiment", "proof of concept",
                              "poc"]),
]

# Ordered so multi-word metrics win over their substrings.
_METRIC_PATTERNS: list[tuple[str, str]] = [
    (r"\bmacro[- ]?f1\b", "macro F1"),
    (r"\bf1[- ]?score\b|\bf1\b", "F1"),
    (r"\broc[- ]?auc\b|\bau[cr]oc\b|\bauc\b", "AUC"),
    (r"\bprecision\b", "precision"),
    (r"\brecall\b", "recall"),
    (r"\baccuracy\b", "accuracy"),
    (r"\brmse\b", "RMSE"),
    (r"\bmae\b", "MAE"),
    (r"\br2\b|\br\^2\b", "R2"),
    (r"\bbleu\b", "BLEU"),
    (r"\brouge\b", "ROUGE"),
    (r"\blatency\b", "latency"),
]

_TARGET_NAMES = ("label", "target", "class", "category", "y", "sentiment", "outcome")


def compile_spec(
    objective: str,
    dataset: Optional[Dataset] = None,
    target_column: Optional[str] = None,
    project_brief: Optional[str] = None,
    use_llm: Optional[bool] = None,
) -> ProjectSpec:
    """Compile the two free-text inputs into a structured ProjectSpec."""
    combined = " ".join(filter(None, [objective, project_brief])).lower()
    obj_text = objective.lower()
    user_set_target = target_column is not None

    task_type = _match(combined, _TASK_KEYWORDS, TaskType.UNKNOWN)
    modality = _match(combined, _MODALITY_KEYWORDS, Modality.UNKNOWN)

    # Target: explicit flag > "predict <column>" in the objective > a column the
    # objective names outright > a conventionally named label column.
    if target_column is None and dataset is not None:
        columns = dataset.df.columns
        lowered = {c.lower(): c for c in columns}
        m = re.search(r"predict(?:ing)?\s+(?:the\s+)?[\"'`]?(\w+)[\"'`]?", obj_text)
        named = _column_named_in(obj_text, columns,
                                 dataset.profile.get("label_candidates") or [])
        if m and m.group(1) in lowered:
            target_column = lowered[m.group(1)]
        elif named:
            target_column = named
        else:
            candidates = dataset.profile.get("label_candidates") or []
            cand_lower = {c.lower(): c for c in candidates}
            for guess in _TARGET_NAMES:
                if guess in cand_lower:
                    target_column = cand_lower[guess]
                    break
            if target_column is None:
                for guess in _TARGET_NAMES:
                    if guess in lowered:
                        target_column = lowered[guess]
                        break

    if dataset is not None:
        exclude = [target_column] if target_column else []
        has_text = bool(dataset.text_columns(exclude))
        has_struct = bool(dataset.numeric_columns(exclude)
                          or dataset.categorical_columns(exclude))
        if modality is Modality.UNKNOWN:
            if has_text and has_struct:
                modality = Modality.MIXED
            elif has_text:
                modality = Modality.TEXT
            elif has_struct:
                modality = Modality.TABULAR

        # Let the target column testify when the description is silent.
        if task_type is TaskType.UNKNOWN and target_column in dataset.df.columns:
            kind = dataset.kinds().get(target_column)
            series = dataset.df[target_column]
            if kind == KIND_CATEGORICAL or (series.dtype.is_integer()
                                           and series.n_unique() <= 20):
                task_type = TaskType.CLASSIFICATION
            elif kind == KIND_NUMERIC:
                task_type = TaskType.REGRESSION

    # Domain / deployment / metric lean on the fuller brief when present.
    brief_text = (project_brief or "").lower()
    spec = ProjectSpec(
        raw_description=objective,
        project_brief=project_brief,
        task_type=task_type,
        modality=modality,
        target_column=target_column,
        domain=_match_str(brief_text or combined, _DOMAIN_KEYWORDS),
        deployment_context=_match_str(brief_text or combined, _DEPLOYMENT_KEYWORDS),
        success_metric=_match_metric(combined),
    )

    should_use_llm = llm_backend.llm_enabled() if use_llm is None else use_llm
    if should_use_llm:
        spec = _apply_llm(spec, objective, project_brief, dataset, user_set_target)

    spec.clarifying_questions = _clarifications(spec)
    return spec


def _apply_llm(spec: ProjectSpec, objective: str, brief: Optional[str],
               dataset: Optional[Dataset], user_set_target: bool) -> ProjectSpec:
    backend = llm_backend.get_backend()
    if backend is None:
        return spec
    columns = list(dataset.df.columns) if dataset is not None else []
    prompt = llm_backend.build_prompt(objective, brief, columns)
    suggestions = llm_backend.sanitize(backend.suggest_fields(prompt), columns)

    filled: list[str] = []
    for field, value in suggestions.items():
        if field == "target_column" and user_set_target:
            continue  # never override an explicit user target
        current = getattr(spec, field)
        if current not in (None, TaskType.UNKNOWN, Modality.UNKNOWN):
            continue  # the LLM only fills gaps
        try:
            if field == "task_type":
                value = TaskType(value)
            elif field == "modality":
                value = Modality(value)
            setattr(spec, field, value)
            filled.append(field)
        except ValueError:
            continue  # out-of-enum value; keep the heuristic result
    spec.llm_assisted_fields = filled
    return spec


def _column_named_in(objective: str, columns: list[str],
                     label_candidates: list[str]) -> Optional[str]:
    """A column the objective refers to by name or by stem ("churn" -> ``churned``).

    Only label-like columns are considered, so an objective that mentions a
    feature ("classify reviews by sentiment" with a ``review`` text column) does
    not make that feature the target.
    """
    words = {w for w in re.findall(r"[a-z]+", objective) if len(w) >= 4}
    best: Optional[str] = None
    for column in columns:
        if column not in label_candidates:
            continue
        for part in re.findall(r"[a-z]+", column.lower()):
            if len(part) >= 4 and any(part.startswith(w) or w.startswith(part)
                                      for w in words):
                if best is not None and best != column:
                    return None  # two columns fit: ambiguous, let the user choose
                best = column
    return best


def _clarifications(spec: ProjectSpec) -> list[str]:
    questions: list[str] = []
    if spec.task_type is TaskType.UNKNOWN:
        questions.append("What kind of task is this? (classification, regression, "
                         "generation, retrieval)")
    if spec.task_type in (TaskType.CLASSIFICATION, TaskType.REGRESSION) \
            and spec.target_column is None:
        questions.append("Which column is the prediction target/label?")
    return questions


def _match(text: str, table: list[tuple], default):
    for value, keywords in table:
        if any(k in text for k in keywords):
            return value
    return default


def _match_str(text: str, table: list[tuple[str, list[str]]]) -> Optional[str]:
    for value, keywords in table:
        if any(k in text for k in keywords):
            return value
    return None


def _match_metric(text: str) -> Optional[str]:
    for pattern, name in _METRIC_PATTERNS:
        if re.search(pattern, text):
            return name
    return None
