"""Provider-neutral LLM client interface, plus Anthropic and Ollama backends.

Every component that talks to a model does so through :class:`LLMClient`. Two
implementations satisfy it:

* :class:`AnthropicClient` — the original hosted path, unchanged in behaviour
  (adaptive thinking, effort, native structured outputs, native tool use).
* :class:`OllamaClient` — a local backend with ``qwen3:4b`` (the project
  default), hitting ``http://localhost:11434`` on the host and
  ``http://host.docker.internal:11434`` from inside the target container.

:func:`make_client` reads :class:`~common.config.Settings` and returns the
backend selected by ``LLM_BACKEND`` (``ollama`` by default, ``anthropic`` opt-in).

## Why an abstraction, and what stays provider-specific

Two conversation shapes exist in this project:

* the **agent's multi-turn tool loop** (`target_agent/agent.py`), and
* **one-shot structured JSON** for the attacker generator and the LLM judge.

Both go through this interface. The agent never builds a provider-shaped message
dict itself — it uses ``build_user`` / ``build_assistant_echo`` /
``build_tool_results`` and reads a normalised :class:`LLMResponse`. Each client
owns its own wire dialect, so the Anthropic path is byte-for-byte what it was and
the Ollama path can differ (e.g. one ``role:"tool"`` message per result vs
Anthropic's single ``user`` message of ``tool_result`` blocks) without the loop
knowing.

## Tool-calling reliability — READ THIS

``qwen3:4b`` on Ollama supports **native function/tool calling**: `/api/chat`
returns structured ``message.tool_calls`` and accepts ``role:"tool"`` results,
so this backend uses that native path — no brittle prompt-based "emit JSON in
your reply" parsing. Verified empirically before this was written.

That said, tool-use reliability is still **materially lower than a frontier
hosted model's**, and the difference is the model, not the plumbing:

* a 4B local model follows tool schemas and multi-step instructions less
  consistently — it may skip a tool, hallucinate an argument, or answer in prose
  when it should have called a tool;
* ``qwen3`` is a *reasoning* model: it emits a separate ``thinking`` stream and
  is slower per turn than a hosted model;
* there is **no ``refusal`` stop reason** — Ollama refusals arrive as ordinary
  text, so ``refusal``-based signals never fire on this backend;
* disabling thinking (``think:false``) is rejected by Ollama when ``tools`` are
  present, so the agent path always runs with thinking on.

Net effect on the lab: attack-success and defence-attribution numbers from the
Ollama backend measure *this local model*, and are not comparable to numbers
from the Anthropic backend. Treat the two as different subjects, not two runs of
the same experiment.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from common.config import Settings

__all__ = [
    "LLMError",
    "LLMRefusal",
    "ToolCall",
    "ToolResult",
    "LLMResponse",
    "LLMClient",
    "AnthropicClient",
    "OllamaClient",
    "make_client",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LLMError(RuntimeError):
    """A model call failed in a way the caller should surface, not retry blindly."""


class LLMRefusal(LLMError):
    """The model declined the request (Anthropic ``stop_reason == "refusal"``).

    Ollama has no equivalent; this is never raised on that backend.
    """

    def __init__(self, category: str | None, explanation: str | None) -> None:
        self.category = category
        self.explanation = explanation
        super().__init__(f"model refused (category={category}): {explanation}")


# ---------------------------------------------------------------------------
# Normalised request/response types
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """A provider-agnostic tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """The outcome of running one tool, fed back to the model next turn."""

    tool_use_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass
class LLMResponse:
    """One model turn, normalised across providers.

    ``display_blocks`` is the transcript-shaped view (``text`` / ``thinking`` /
    ``tool_use`` blocks) every downstream consumer (judge, signals) already
    understands. ``assistant_message`` is the provider-native message, opaque to
    the loop, used only to replay history via :meth:`LLMClient.build_assistant_echo`.
    """

    provider: str
    model: str
    text: str
    stop_reason: str  # "end_turn" | "tool_use" | "max_tokens" | "refusal"
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    display_blocks: list[dict[str, Any]] = field(default_factory=list)
    assistant_message: Any = None
    refusal: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class LLMClient(ABC):
    """The single surface every caller uses instead of a provider SDK."""

    provider: str = "abstract"

    def __init__(self, model: str) -> None:
        self.model = model

    # -- message construction (each returns messages to append to history) --

    @abstractmethod
    def build_user(self, text: str) -> list[dict[str, Any]]:
        """A user turn carrying plain text."""

    @abstractmethod
    def build_assistant_echo(self, response: LLMResponse) -> list[dict[str, Any]]:
        """Echo a prior model turn back into history (for the next request)."""

    @abstractmethod
    def build_tool_results(self, results: list[ToolResult]) -> list[dict[str, Any]]:
        """Feed tool outcomes back to the model. May be one message or many."""

    # -- the core call ------------------------------------------------------

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 8000,
        effort: str = "medium",
    ) -> LLMResponse:
        """Run one model turn over ``messages`` and return a normalised response."""

    # -- one-shot structured output ----------------------------------------

    @abstractmethod
    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str | None = None,
        max_tokens: int = 4000,
        effort: str = "medium",
    ) -> dict[str, Any]:
        """Return a dict constrained to ``schema`` (no regex extraction)."""


# ---------------------------------------------------------------------------
# Anthropic backend — behaviourally identical to the original code
# ---------------------------------------------------------------------------


class AnthropicClient(LLMClient):
    """Hosted Anthropic backend. One interchangeable implementation of the
    interface; the original code path, preserved."""

    provider = "anthropic"

    def __init__(self, *, api_key: str, base_url: str, model: str) -> None:
        super().__init__(model)
        import anthropic  # local import: the container/Ollama path needn't load it

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key, base_url=base_url)

    # -- message construction ----------------------------------------------

    def build_user(self, text: str) -> list[dict[str, Any]]:
        return [{"role": "user", "content": text}]

    def build_assistant_echo(self, response: LLMResponse) -> list[dict[str, Any]]:
        # Echo the raw SDK content so thinking blocks round-trip unmodified.
        return [{"role": "assistant", "content": response.assistant_message}]

    def build_tool_results(self, results: list[ToolResult]) -> list[dict[str, Any]]:
        # Anthropic: ALL tool results for a turn go back in ONE user message.
        blocks = [
            {
                "type": "tool_result",
                "tool_use_id": r.tool_use_id,
                "content": r.content,
                "is_error": r.is_error,
            }
            for r in results
        ]
        return [{"role": "user", "content": blocks}]

    # -- core call ----------------------------------------------------------

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 8000,
        effort: str = "medium",
    ) -> LLMResponse:
        try:
            response = self._client.messages.create(
                model=model or self.model,
                max_tokens=max_tokens,
                system=system,
                tools=tools or [],
                messages=messages,
                thinking={"type": "adaptive"},
                output_config={"effort": effort},
            )
        except self._anthropic.APIError as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc

        stop_reason = response.stop_reason or "end_turn"
        refusal = None
        if stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            refusal = {
                "category": getattr(details, "category", None),
                "explanation": getattr(details, "explanation", None),
            }

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        display_blocks: list[dict[str, Any]] = []
        for block in response.content:
            btype = getattr(block, "type", None)
            display_blocks.append(_anthropic_block_to_dict(block))
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {}))
                )

        return LLMResponse(
            provider=self.provider,
            model=model or self.model,
            text="\n".join(text_parts).strip(),
            stop_reason=stop_reason,
            tool_calls=tool_calls,
            usage=_anthropic_usage(response),
            display_blocks=display_blocks,
            assistant_message=response.content,
            refusal=refusal,
        )

    # -- structured JSON ----------------------------------------------------

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str | None = None,
        max_tokens: int = 4000,
        effort: str = "medium",
    ) -> dict[str, Any]:
        try:
            response = self._client.messages.create(
                model=model or self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={
                    "effort": effort,
                    "format": {"type": "json_schema", "schema": schema},
                },
            )
        except self._anthropic.APIError as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMRefusal(
                getattr(details, "category", None),
                getattr(details, "explanation", None),
            )

        text = ""
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text = block.text
                break
        return _parse_json_object(text, stop_reason=response.stop_reason)


def _anthropic_block_to_dict(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        return block
    dump = getattr(block, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json", exclude_none=True)
        except Exception:  # noqa: BLE001, S110 - fall back to a repr below
            pass
    return {"type": getattr(block, "type", "unknown"), "repr": repr(block)}


def _anthropic_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    out: dict[str, int] = {}
    if usage is None:
        return out
    for field_name in (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ):
        value = getattr(usage, field_name, None)
        if isinstance(value, int):
            out[field_name] = value
    return out


# ---------------------------------------------------------------------------
# Ollama backend — local, native tool calling
# ---------------------------------------------------------------------------

# Where the Ollama endpoint may point.
#
# Two execution contexts, two answers:
#
# * **on the host** (in-process executor) the daemon is on loopback and the
#   traffic never leaves the machine — always permitted, no allowlist needed;
# * **inside the container** loopback is the container itself, so the host is
#   reached via ``host.docker.internal``. That is real egress, so it is gated by
#   the same allowlist the Anthropic path uses (``target_agent/net.py``), which
#   permits exactly ``host.docker.internal:11434`` and nothing else.
#
# A remote inference host is still refused: a red-team lab has no reason to ship
# prompts and transcripts off-box, and forbidding it removes an exfiltration
# surface. Point OLLAMA_BASE_URL at anything else and this raises rather than
# silently phoning home.
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _in_container() -> bool:
    """True when this process is running inside the target container."""
    return Path("/.dockerenv").exists()


def _assert_endpoint_allowed(base_url: str, allowlist: tuple[str, ...] | list[str]) -> None:
    """Permit loopback (host-side), or an allowlisted local model endpoint (container)."""
    parts = urlsplit(base_url)
    if parts.scheme not in ("http", "https"):
        raise LLMError(f"OLLAMA_BASE_URL scheme {parts.scheme!r} is not http/https: {base_url!r}")

    host = (parts.hostname or "").lower()
    if host in _LOOPBACK_HOSTS:
        if _in_container():
            # Loud, not silent: inside the container loopback resolves to the
            # container itself, so this would fail with a connection error that
            # looks like "Ollama isn't running" rather than a misconfiguration.
            raise LLMError(
                f"OLLAMA_BASE_URL is {base_url!r}, but this process is inside the "
                f"container, where loopback is the container — not the host running "
                f"Ollama. docker-compose.yml sets OLLAMA_BASE_URL to "
                f"http://host.docker.internal:11434 for the container; override it "
                f"with OLLAMA_CONTAINER_BASE_URL, not OLLAMA_BASE_URL."
            )
        return

    from target_agent.net import EgressViolation, assert_url_allowed

    try:
        assert_url_allowed(base_url, allowlist)
    except EgressViolation as exc:
        raise LLMError(
            f"OLLAMA_BASE_URL {base_url!r} is neither loopback nor permitted by the "
            f"egress allowlist {tuple(allowlist)!r}: {exc}"
        ) from exc


class OllamaClient(LLMClient):
    """Local Ollama backend using the native ``/api/chat`` tool-calling API."""

    provider = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout: int = 300,
        num_ctx: int = 8192,
        egress_allowlist: tuple[str, ...] | list[str] = (),
    ) -> None:
        super().__init__(model)
        _assert_endpoint_allowed(base_url, egress_allowlist)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.num_ctx = num_ctx

    # -- message construction ----------------------------------------------

    def build_user(self, text: str) -> list[dict[str, Any]]:
        return [{"role": "user", "content": text}]

    def build_assistant_echo(self, response: LLMResponse) -> list[dict[str, Any]]:
        # assistant_message is the native Ollama message dict (content + tool_calls).
        return [response.assistant_message]

    def build_tool_results(self, results: list[ToolResult]) -> list[dict[str, Any]]:
        # Ollama expects ONE role:"tool" message per result, keyed by tool_name.
        return [
            {"role": "tool", "content": r.content, "tool_name": r.name} for r in results
        ]

    # -- core call ----------------------------------------------------------

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 8000,
        effort: str = "medium",  # noqa: ARG002 - no effort concept on Ollama
    ) -> LLMResponse:
        wire_messages = [{"role": "system", "content": system}, *messages]
        body: dict[str, Any] = {
            "model": model or self.model,
            "messages": wire_messages,
            "stream": False,
            "options": {"temperature": 0, "num_ctx": self.num_ctx, "num_predict": max_tokens},
        }
        if tools:
            body["tools"] = _to_ollama_tools(tools)
            # NB: do NOT send think:false alongside tools — Ollama 500s on that
            # combination for qwen3. Thinking stays on for the agent path.

        data = self._post("/api/chat", body)
        message = data.get("message", {}) or {}

        raw_tool_calls = message.get("tool_calls") or []
        tool_calls: list[ToolCall] = []
        for index, call in enumerate(raw_tool_calls):
            fn = call.get("function", {}) or {}
            arguments = fn.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"_raw": arguments}
            tool_calls.append(
                ToolCall(
                    id=call.get("id") or f"ollama-call-{index}",
                    name=fn.get("name", ""),
                    arguments=arguments if isinstance(arguments, dict) else {},
                )
            )

        text = message.get("content") or ""
        thinking = message.get("thinking") or ""
        done_reason = data.get("done_reason")
        if tool_calls:
            stop_reason = "tool_use"
        elif done_reason == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"

        display_blocks: list[dict[str, Any]] = []
        if thinking:
            display_blocks.append({"type": "thinking", "thinking": thinking})
        if text:
            display_blocks.append({"type": "text", "text": text})
        for call in tool_calls:
            display_blocks.append(
                {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
            )

        return LLMResponse(
            provider=self.provider,
            model=model or self.model,
            text=text.strip(),
            stop_reason=stop_reason,
            tool_calls=tool_calls,
            thinking=thinking,
            usage=_ollama_usage(data),
            display_blocks=display_blocks,
            # Native assistant message for history replay. Content must be a
            # string; tool_calls carried through so the model sees its own call.
            assistant_message={
                "role": "assistant",
                "content": text,
                **({"tool_calls": raw_tool_calls} if raw_tool_calls else {}),
            },
            refusal=None,  # Ollama has no structured refusal signal
        )

    # -- structured JSON ----------------------------------------------------

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str | None = None,
        max_tokens: int = 4000,
        effort: str = "medium",  # noqa: ARG002 - no effort concept on Ollama
    ) -> dict[str, Any]:
        body = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": schema,  # Ollama structured outputs: constrain to this schema
            "stream": False,
            # No tools here, so think:false is safe and gives a clean JSON body.
            "think": False,
            "options": {"temperature": 0, "num_ctx": self.num_ctx, "num_predict": max_tokens},
        }
        data = self._post("/api/chat", body)
        text = (data.get("message", {}) or {}).get("content", "") or ""
        return _parse_json_object(text, stop_reason=data.get("done_reason"))

    # -- transport ----------------------------------------------------------

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(  # noqa: S310 - loopback URL, validated in __init__
            url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            raise LLMError(
                f"Ollama HTTP {exc.code} from {path}: {detail}. Is Ollama running "
                f"and is the model pulled (`ollama pull {body.get('model')}`)?"
            ) from exc
        except urllib.error.URLError as exc:
            raise LLMError(
                f"Cannot reach Ollama at {self.base_url} ({exc.reason}). "
                f"Start it with `ollama serve`."
            ) from exc
        except (TimeoutError, json.JSONDecodeError) as exc:
            raise LLMError(f"Ollama request to {path} failed: {exc}") from exc


def _to_ollama_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the project's neutral tool schemas to Ollama's function format.

    The neutral schema (from ``target_agent.tools.tool_schemas``) is Anthropic-
    shaped: ``{name, description, input_schema, strict}``. Ollama wants
    ``{type:"function", function:{name, description, parameters}}``; ``strict``
    has no Ollama equivalent and is dropped.
    """
    converted: list[dict[str, Any]] = []
    for tool in tools:
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
        )
    return converted


def _ollama_usage(data: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    if isinstance(data.get("prompt_eval_count"), int):
        out["input_tokens"] = data["prompt_eval_count"]
    if isinstance(data.get("eval_count"), int):
        out["output_tokens"] = data["eval_count"]
    return out


# ---------------------------------------------------------------------------
# Shared helpers + factory
# ---------------------------------------------------------------------------


def _parse_json_object(text: str, *, stop_reason: Any = None) -> dict[str, Any]:
    if not text.strip():
        raise LLMError(f"model returned no text (stop_reason={stop_reason})")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"structured output was not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LLMError("structured output was not a JSON object")
    return payload


def make_client(settings: Settings) -> LLMClient:
    """Construct the LLM client selected by ``settings.llm_backend``.

    * ``ollama`` (default) → :class:`OllamaClient` (no API key required).
    * ``anthropic``        → :class:`AnthropicClient`, with the egress allowlist
      enforced on ``base_url`` exactly as before.
    """
    backend = settings.llm_backend
    if backend == "ollama":
        return OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.target_model,  # per-role model is passed on each call
            timeout=settings.ollama_timeout,
            num_ctx=settings.ollama_num_ctx,
            # Only consulted for a non-loopback endpoint, i.e. the container
            # reaching host.docker.internal. Host-side runs never hit it.
            egress_allowlist=settings.egress_allowlist,
        )
    if backend == "anthropic":
        # Enforce the egress allowlist on the API base URL, unchanged from the
        # original make_client (target_agent.net is not modified by this refactor).
        from target_agent.net import assert_url_allowed

        assert_url_allowed(settings.base_url, settings.egress_allowlist)
        return AnthropicClient(
            api_key=settings.require_api_key(),
            base_url=settings.base_url,
            model=settings.target_model,
        )
    raise LLMError(f"unknown LLM_BACKEND {backend!r}; expected 'ollama' or 'anthropic'")
