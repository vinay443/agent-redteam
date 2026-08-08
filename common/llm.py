"""Thin helpers over the Anthropic SDK.

Two things live here:

* :func:`make_client` — a client whose ``base_url`` has been checked against the
  egress allowlist before construction.
* :func:`complete_json` — one-shot structured-output call used by the attacker's
  variant generator and by the LLM judge. It handles the ``refusal`` stop reason
  explicitly rather than indexing blindly into ``response.content``.
"""

from __future__ import annotations

import json
from typing import Any

import anthropic

from common.config import Settings
from target_agent.net import assert_url_allowed

__all__ = ["make_client", "complete_json", "LLMError", "LLMRefusal"]


class LLMError(RuntimeError):
    """A model call failed in a way the caller should surface, not retry blindly."""


class LLMRefusal(LLMError):
    """The model declined the request (``stop_reason == "refusal"``)."""

    def __init__(self, category: str | None, explanation: str | None) -> None:
        self.category = category
        self.explanation = explanation
        super().__init__(f"model refused (category={category}): {explanation}")


def make_client(settings: Settings) -> anthropic.Anthropic:
    """Construct an Anthropic client, enforcing the egress allowlist on ``base_url``."""
    assert_url_allowed(settings.base_url, settings.egress_allowlist)
    return anthropic.Anthropic(
        api_key=settings.require_api_key(),
        base_url=settings.base_url,
    )


def _first_text(response: Any) -> str:
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


def complete_json(
    client: anthropic.Anthropic,
    *,
    model: str,
    system: str,
    user: str,
    schema: dict[str, Any],
    max_tokens: int = 4000,
    effort: str = "medium",
) -> dict[str, Any]:
    """Call the model and return a dict validated by the API against ``schema``.

    Uses structured outputs (``output_config.format``), so the first text block
    is guaranteed to be JSON matching the schema — no regex extraction, no
    retry-on-parse loop.
    """
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={
                "effort": effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        )
    except anthropic.APIError as exc:
        raise LLMError(f"{type(exc).__name__}: {exc}") from exc

    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        raise LLMRefusal(
            getattr(details, "category", None),
            getattr(details, "explanation", None),
        )

    text = _first_text(response)
    if not text:
        raise LLMError(f"no text block in response (stop_reason={response.stop_reason})")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:  # pragma: no cover - schema makes this unreachable
        raise LLMError(f"structured output was not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LLMError("structured output was not a JSON object")
    return payload
