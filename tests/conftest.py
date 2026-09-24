"""Shared test doubles.

The failing-judge client lives here rather than in one test module because two
levels of the stack need the same scenario: the judge tests check the verdict it
produces, and the report tests check what the campaign numbers do with it.
"""

from __future__ import annotations

from typing import Any

import pytest

from common.llm_client import LLMClient, LLMError


class FailingJudgeClient(LLMClient):
    """An LLM backend whose every call fails — a judge that times out.

    Only ``complete_json`` is a real implementation, because that is the only
    method the judge path uses. The rest fail loudly rather than plausibly, so a
    test that wanders into the agent loop through this client says so.
    """

    provider = "failing-test-double"

    def __init__(self, error: str = "ollama request timed out after 120s") -> None:
        super().__init__("fake-judge-model")
        self.error = error
        self.calls = 0

    def complete_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        raise LLMError(self.error)

    def complete(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the judge path must not call complete()")

    def build_user(self, text: str) -> list[dict[str, Any]]:
        raise AssertionError("the judge path must not build conversation turns")

    def build_assistant_echo(self, response: Any) -> list[dict[str, Any]]:
        raise AssertionError("the judge path must not build conversation turns")

    def build_tool_results(self, results: Any) -> list[dict[str, Any]]:
        raise AssertionError("the judge path must not build conversation turns")


@pytest.fixture
def failing_judge_client() -> FailingJudgeClient:
    return FailingJudgeClient()
