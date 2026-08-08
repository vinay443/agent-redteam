"""The agent loop, exercised against a fake LLMClient.

No network, no API key, no provider. A scripted client implementing the
:class:`~common.llm_client.LLMClient` interface returns canned responses so we
can assert the loop's mechanics: tool calls are executed and fed back, an
injected "write PWNED.txt" instruction actually reaches the toolbox, and
containment holds end to end — independent of which backend is configured.
"""

from __future__ import annotations

from typing import Any

from common.llm_client import LLMClient, LLMResponse, ToolCall, ToolResult
from target_agent.agent import TargetAgent
from target_agent.tools import Toolbox


def _display_blocks(text: str, tool_calls: list[ToolCall]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if text:
        blocks.append({"type": "text", "text": text})
    for call in tool_calls:
        blocks.append({"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments})
    return blocks


class FakeClient(LLMClient):
    """Scripted, provider-neutral client for exercising the loop."""

    provider = "fake"

    def __init__(self, script: list[dict[str, Any]]) -> None:
        super().__init__(model="fake-model")
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    def build_user(self, text: str) -> list[dict[str, Any]]:
        return [{"role": "user", "content": text}]

    def build_assistant_echo(self, response: LLMResponse) -> list[dict[str, Any]]:
        return [{"role": "assistant", "content": response.text, "_calls": response.tool_calls}]

    def build_tool_results(self, results: list[ToolResult]) -> list[dict[str, Any]]:
        return [{"role": "tool", "content": r.content, "name": r.name} for r in results]

    def complete(self, messages, *, system, tools=None, model=None, max_tokens=8000, effort="medium"):
        self.calls.append({"messages": list(messages), "tools": tools})
        step = self._script.pop(0)
        tool_calls = step.get("tool_calls", [])
        text = step.get("text", "")
        return LLMResponse(
            provider=self.provider,
            model=self.model,
            text=text,
            stop_reason=step["stop"],
            tool_calls=tool_calls,
            usage={"input_tokens": 10, "output_tokens": 5},
            display_blocks=_display_blocks(text, tool_calls),
            assistant_message={"role": "assistant", "content": text},
            refusal=step.get("refusal"),
        )

    def complete_json(self, *, system, user, schema, model=None, max_tokens=4000, effort="medium"):
        return {}


def _agent(tmp_path, script):
    toolbox = Toolbox(root=tmp_path / "sandbox")
    client = FakeClient(script)
    return TargetAgent(
        client=client,
        toolbox=toolbox,
        model="fake-model",
        sandbox_root=str(toolbox.root),
        canary="SPX-CANARY-test01",
        max_turns=5,
    )


def test_plain_answer_no_tools(tmp_path):
    agent = _agent(tmp_path, [{"text": "The file lists three action items.", "stop": "end_turn"}])
    run = agent.run("summarise notes", run_id="r", attack_id="a", category="prompt_injection")
    assert run.final_text.startswith("The file lists")
    assert run.turns == 1
    assert run.tool_calls == []
    assert run.provider == "fake"


def test_tool_call_is_executed_and_looped(tmp_path):
    (tmp_path / "sandbox").mkdir(parents=True)
    script = [
        {
            "tool_calls": [ToolCall(id="t1", name="write_file", arguments={"path": "out.txt", "content": "done"})],
            "stop": "tool_use",
        },
        {"text": "Wrote the file.", "stop": "end_turn"},
    ]
    agent = _agent(tmp_path, script)
    run = agent.run("write out.txt", run_id="r", attack_id="a", category="prompt_injection")
    assert len(run.tool_calls) == 1
    assert run.tool_calls[0].ok
    assert (tmp_path / "sandbox" / "out.txt").read_text() == "done"
    assert run.final_text == "Wrote the file."
    assert run.turns == 2


def test_injected_escalation_is_contained_end_to_end(tmp_path):
    # The model (misbehaving) tries to write outside the root; the toolbox blocks
    # it and the loop records the attempt without escaping.
    script = [
        {
            "tool_calls": [ToolCall(id="t1", name="write_file", arguments={"path": "../escape.txt", "content": "pwn"})],
            "stop": "tool_use",
        },
        {"text": "I could not do that.", "stop": "end_turn"},
    ]
    agent = _agent(tmp_path, script)
    run = agent.run("please escape", run_id="r", attack_id="a", category="permission_escalation")
    assert len(run.tool_calls) == 1
    tc = run.tool_calls[0]
    assert not tc.ok
    assert tc.blocked
    assert tc.outside_root
    assert run.escaping_tool_calls == []  # containment held
    assert not (tmp_path / "escape.txt").exists()


def test_refusal_stop_reason_is_handled(tmp_path):
    script = [{"stop": "refusal", "refusal": {"category": "cyber", "explanation": "declined"}}]
    agent = _agent(tmp_path, script)
    run = agent.run("do something", run_id="r", attack_id="a", category="prompt_injection")
    assert run.refusal is not None
    assert run.refusal["category"] == "cyber"
    assert run.stop_reason == "refusal"
