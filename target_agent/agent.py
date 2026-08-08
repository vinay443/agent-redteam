"""The agent under test: a plain function-calling loop over the Messages API.

No framework. The loop is ~80 lines and is meant to be read end to end, because
"what exactly did the agent do" is the primary artefact the lab produces.

Loop shape (standard manual tool-use loop):

    user turn → model → if tool_use: run tools, append ONE user message
    containing every tool_result → model → … → end_turn

Two details that matter and are easy to get wrong:

* the assistant turn is appended **whole** (``response.content``), so thinking
  blocks round-trip unmodified as the API requires;
* every ``tool_use`` block gets exactly one ``tool_result``, in a single user
  message — splitting them trains the model out of parallel tool use.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import anthropic

from common.logging import EventLogger, NullLogger
from common.timeutil import monotonic_ms, utcnow_iso
from target_agent.guard import Guard, GuardDecision, NullGuard
from target_agent.prompts import build_system_prompt, new_canary, prompt_fingerprint
from target_agent.tools import Toolbox, tool_schemas

__all__ = ["TargetAgent", "AgentRun", "ToolCallRecord"]


@dataclass
class ToolCallRecord:
    """One tool invocation, as it will appear in the results database."""

    seq: int
    ts: str
    tool_use_id: str
    name: str
    arguments: dict[str, Any]
    ok: bool
    blocked: bool
    outside_root: bool
    guard_blocked: bool
    error: str | None
    requested_path: str | None
    resolved_path: str | None
    violation: dict[str, Any] | None
    result_preview: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentRun:
    """Everything one attack produced. Serialised verbatim into the store."""

    run_id: str
    attack_id: str
    category: str
    model: str
    sandbox_root: str
    system_prompt_sha256: str
    canary: str
    started_at: str
    finished_at: str | None = None
    duration_ms: float = 0.0
    turns: int = 0
    stop_reason: str | None = None
    refusal: dict[str, Any] | None = None
    final_text: str = ""
    transcript: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        return payload

    # -- convenience accessors used by the judge ---------------------------

    @property
    def blocked_tool_calls(self) -> list[ToolCallRecord]:
        return [tc for tc in self.tool_calls if tc.blocked]

    @property
    def escaping_tool_calls(self) -> list[ToolCallRecord]:
        """Calls that both targeted outside the root *and* succeeded.

        This list must always be empty. If it is not, containment failed.
        """
        return [tc for tc in self.tool_calls if tc.outside_root and tc.ok]

    def all_text(self) -> str:
        """Concatenation of every assistant text block, for code-based checks."""
        chunks: list[str] = []
        for message in self.transcript:
            if message.get("role") != "assistant":
                continue
            content = message.get("content")
            if isinstance(content, str):
                chunks.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        chunks.append(str(block.get("text", "")))
        return "\n".join(chunks)


def _jsonable_block(block: Any) -> dict[str, Any]:
    """Convert an SDK content block into a plain dict for the transcript."""
    if isinstance(block, dict):
        return block
    dump = getattr(block, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json", exclude_none=True)
        except Exception:  # noqa: BLE001, S110 - fall back to a repr below
            pass
    return {"type": getattr(block, "type", "unknown"), "repr": repr(block)}


class TargetAgent:
    """The agent under test."""

    def __init__(
        self,
        *,
        client: anthropic.Anthropic,
        toolbox: Toolbox,
        model: str,
        sandbox_root: str,
        canary: str | None = None,
        max_turns: int = 8,
        max_tokens: int = 8000,
        effort: str = "medium",
        guard: Guard | None = None,
        logger: EventLogger | None = None,
    ) -> None:
        self.client = client
        self.toolbox = toolbox
        self.model = model
        self.sandbox_root = sandbox_root
        self.canary = canary or new_canary()
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.effort = effort
        self.guard = guard or NullGuard()
        self.logger = logger or NullLogger()
        self.system_prompt = build_system_prompt(
            sandbox_root=sandbox_root, canary=self.canary
        )
        self.tools = tool_schemas()

    # -----------------------------------------------------------------------

    def run(self, user_message: str, *, run_id: str, attack_id: str, category: str) -> AgentRun:
        """Drive the loop to completion and return the full record."""
        started_ms = monotonic_ms()
        result = AgentRun(
            run_id=run_id,
            attack_id=attack_id,
            category=category,
            model=self.model,
            sandbox_root=self.sandbox_root,
            system_prompt_sha256=prompt_fingerprint(self.system_prompt),
            canary=self.canary,
            started_at=utcnow_iso(),
        )

        decision: GuardDecision = self.guard.on_user_message(user_message)
        if not decision.allow:
            result.error = f"guard_blocked_user_message: {decision.reason}"
            result.finished_at = utcnow_iso()
            self.logger.emit("agent_run_blocked", attack_id=attack_id, reason=decision.reason)
            return result

        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
        result.transcript.append({"role": "user", "content": user_message})

        self.logger.emit(
            "agent_run_start",
            attack_id=attack_id,
            category=category,
            model=self.model,
            sandbox_root=self.sandbox_root,
            max_turns=self.max_turns,
            prompt_sha256=result.system_prompt_sha256,
        )

        tool_seq = 0
        try:
            for turn in range(1, self.max_turns + 1):
                result.turns = turn
                response = self._call_model(messages)
                self._accumulate_usage(result, response)
                result.stop_reason = response.stop_reason

                self.logger.emit(
                    "model_turn",
                    attack_id=attack_id,
                    turn=turn,
                    stop_reason=response.stop_reason,
                    block_types=[getattr(b, "type", "?") for b in response.content],
                )

                # Opus-tier models can decline a request outright: HTTP 200 with
                # stop_reason "refusal" and an empty/partial content list. Check
                # before touching content.
                if response.stop_reason == "refusal":
                    details = getattr(response, "stop_details", None)
                    result.refusal = {
                        "category": getattr(details, "category", None),
                        "explanation": getattr(details, "explanation", None),
                    }
                    self.logger.emit("model_refusal", attack_id=attack_id, **result.refusal)
                    break

                assistant_blocks = [_jsonable_block(b) for b in response.content]
                messages.append({"role": "assistant", "content": response.content})
                result.transcript.append({"role": "assistant", "content": assistant_blocks})

                tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]

                if not tool_uses:
                    result.final_text = _text_of(assistant_blocks)
                    output_decision = self.guard.on_model_output(result.final_text)
                    if not output_decision.allow:
                        result.final_text = output_decision.replacement or "[blocked by guard]"
                        self.logger.emit(
                            "guard_blocked_output",
                            attack_id=attack_id,
                            reason=output_decision.reason,
                        )
                    break

                tool_results: list[dict[str, Any]] = []
                for block in tool_uses:
                    tool_seq += 1
                    record, result_block = self._run_tool(block, tool_seq)
                    result.tool_calls.append(record)
                    tool_results.append(result_block)

                # Every tool_result for this turn goes back in ONE user message.
                messages.append({"role": "user", "content": tool_results})
                result.transcript.append({"role": "user", "content": tool_results})

                if response.stop_reason == "max_tokens":
                    result.error = "max_tokens_reached"
                    break
            else:
                result.error = "max_turns_reached"

        except anthropic.APIError as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            self.logger.emit("agent_run_error", attack_id=attack_id, error=result.error)
        except Exception as exc:  # noqa: BLE001 - one attack must not kill a campaign
            result.error = f"{type(exc).__name__}: {exc}"
            self.logger.emit("agent_run_error", attack_id=attack_id, error=result.error)

        if not result.final_text and result.transcript:
            result.final_text = _last_assistant_text(result.transcript)

        result.finished_at = utcnow_iso()
        result.duration_ms = round(monotonic_ms() - started_ms, 1)
        self.logger.emit(
            "agent_run_end",
            attack_id=attack_id,
            turns=result.turns,
            stop_reason=result.stop_reason,
            tool_calls=len(result.tool_calls),
            blocked_tool_calls=len(result.blocked_tool_calls),
            escaping_tool_calls=len(result.escaping_tool_calls),
            duration_ms=result.duration_ms,
            error=result.error,
        )
        return result

    # -- internals ----------------------------------------------------------

    def _call_model(self, messages: list[dict[str, Any]]) -> Any:
        return self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            tools=self.tools,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
        )

    def _run_tool(self, block: Any, seq: int) -> tuple[ToolCallRecord, dict[str, Any]]:
        name = block.name
        arguments = dict(block.input or {})
        outcome = self.toolbox.execute(name, arguments)

        content = outcome.content
        result_decision = self.guard.on_tool_result(name, content)
        guard_blocked = not result_decision.allow
        if guard_blocked:
            content = result_decision.replacement or (
                f"[tool result withheld: {result_decision.reason}]"
            )

        record = ToolCallRecord(
            seq=seq,
            ts=utcnow_iso(),
            tool_use_id=block.id,
            name=name,
            arguments=arguments,
            ok=outcome.ok,
            blocked=outcome.blocked,
            outside_root=outcome.outside_root,
            guard_blocked=guard_blocked,
            error=outcome.error,
            requested_path=outcome.requested_path,
            resolved_path=outcome.resolved_path,
            violation=outcome.violation,
            result_preview=content[:2000],
        )
        result_block = {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": content,
            "is_error": not outcome.ok,
        }
        return record, result_block

    @staticmethod
    def _accumulate_usage(result: AgentRun, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        for field_name in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ):
            value = getattr(usage, field_name, None)
            if isinstance(value, int):
                result.usage[field_name] = result.usage.get(field_name, 0) + value


def _text_of(blocks: list[dict[str, Any]]) -> str:
    return "\n".join(
        str(b.get("text", "")) for b in blocks if isinstance(b, dict) and b.get("type") == "text"
    ).strip()


def _last_assistant_text(transcript: list[dict[str, Any]]) -> str:
    for message in reversed(transcript):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, list):
            text = _text_of(content)
            if text:
                return text
    return ""
