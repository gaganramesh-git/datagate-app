"""Optional LLM backend for the Task-Spec Compiler.

Boxed exactly as the design doc requires: the LLM only *parses intent* into
structured spec fields. It never sees analyzer outputs and never influences a
score. It is inert unless explicitly enabled, so the heuristic path is always
the default and nothing here adds a dependency or a network call by accident.

Enable with:
    DATAGATE_USE_LLM=1
    ANTHROPIC_API_KEY=sk-...        (or DATAGATE_LLM_API_KEY)
    DATAGATE_LLM_MODEL=claude-...   (optional; sensible default)

The provider is abstracted behind ``LLMBackend`` — an Anthropic implementation
ships here; swapping providers means one new subclass, not a compiler change.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional, Protocol

log = logging.getLogger("datagate.spec.llm")

# The compiler owns these enums; the LLM is only allowed to fill these keys.
_ALLOWED_FIELDS = {
    "task_type",
    "modality",
    "target_column",
    "domain",
    "deployment_context",
    "success_metric",
}

_SYSTEM_PROMPT = (
    "You extract a structured machine-learning task spec from a user's project "
    "brief and training objective. Respond with a single JSON object and nothing "
    "else. Keys (all optional, omit what you cannot determine): "
    "task_type (one of classification, regression, generation, retrieval), "
    "modality (one of tabular, text, image, mixed), "
    "target_column (must be an exact name from the provided column list, or omit), "
    "domain (short phrase, e.g. 'healthcare', 'e-commerce reviews'), "
    "deployment_context (short phrase, e.g. 'real-time production API'), "
    "success_metric (e.g. 'macro F1', 'RMSE'). "
    "Never invent a target_column that is not in the list. Do not add commentary."
)


class LLMBackend(Protocol):
    def suggest_fields(self, prompt: str) -> dict[str, Any]: ...


def llm_enabled() -> bool:
    """True only when the operator has explicitly turned the backend on."""
    if os.environ.get("DATAGATE_USE_LLM", "").lower() not in ("1", "true", "yes"):
        return False
    return bool(_api_key())


def _api_key() -> Optional[str]:
    return os.environ.get("DATAGATE_LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")


class AnthropicBackend:
    """Anthropic implementation. Imports the SDK lazily so it is never a hard dep."""

    def __init__(self) -> None:
        self.model = os.environ.get("DATAGATE_LLM_MODEL", "claude-sonnet-5")

    def suggest_fields(self, prompt: str) -> dict[str, Any]:
        try:
            import anthropic
        except ImportError:
            log.warning("DATAGATE_USE_LLM set but the anthropic SDK is not installed")
            return {}
        try:
            client = anthropic.Anthropic(api_key=_api_key())
            message = client.messages.create(
                model=self.model,
                max_tokens=512,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                block.text for block in message.content if block.type == "text"
            )
            return json.loads(text)
        except Exception as exc:
            log.warning("LLM spec parsing failed, falling back to heuristics: %s", exc)
            return {}


def get_backend() -> Optional[LLMBackend]:
    return AnthropicBackend() if llm_enabled() else None


def build_prompt(
    objective: str, brief: Optional[str], columns: list[str]
) -> str:
    parts = [f"Training objective: {objective}"]
    if brief:
        parts.append(f"Project brief:\n{brief}")
    if columns:
        parts.append("Dataset columns: " + ", ".join(columns))
    return "\n\n".join(parts)


def sanitize(raw: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    """Keep only allowed keys; drop a hallucinated target_column."""
    clean: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in _ALLOWED_FIELDS or value in (None, ""):
            continue
        if key == "target_column" and columns and value not in columns:
            log.warning("LLM proposed unknown target_column %r; ignoring", value)
            continue
        clean[key] = value
    return clean
