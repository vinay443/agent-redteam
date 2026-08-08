"""Backward-compatible facade over :mod:`common.llm_client`.

The LLM layer was refactored to support pluggable backends (Anthropic, Ollama).
The real implementation now lives in :mod:`common.llm_client`; this module keeps
the historical import surface working:

* ``make_client(settings)`` — now returns an :class:`~common.llm_client.LLMClient`
  chosen by ``settings.llm_backend`` instead of a raw Anthropic client.
* ``complete_json(client, *, ...)`` — a free-function shim that delegates to
  ``client.complete_json(...)``. New code should call the method on the client
  directly.

Prefer importing from :mod:`common.llm_client` in new code.
"""

from __future__ import annotations

from typing import Any

from common.llm_client import (
    AnthropicClient,
    LLMClient,
    LLMError,
    LLMRefusal,
    LLMResponse,
    OllamaClient,
    ToolCall,
    ToolResult,
    make_client,
)

__all__ = [
    "make_client",
    "complete_json",
    "LLMError",
    "LLMRefusal",
    "LLMClient",
    "LLMResponse",
    "AnthropicClient",
    "OllamaClient",
    "ToolCall",
    "ToolResult",
]


def complete_json(
    client: LLMClient,
    *,
    model: str,
    system: str,
    user: str,
    schema: dict[str, Any],
    max_tokens: int = 4000,
    effort: str = "medium",
) -> dict[str, Any]:
    """Deprecated free-function form of :meth:`LLMClient.complete_json`.

    Kept so older call sites keep working; delegates to the client's method.
    """
    return client.complete_json(
        system=system,
        user=user,
        schema=schema,
        model=model,
        max_tokens=max_tokens,
        effort=effort,
    )
