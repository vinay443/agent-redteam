"""Verdict data model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["Verdict", "Signals"]


@dataclass
class Signals:
    """Objective, code-derived facts about a run.

    These are logged for *every* attack regardless of category, because they
    are the lab's separation between "the model refused / didn't comply" and
    "the code stopped it". A permission-escalation attack that produced three
    ``blocked`` tool calls and zero ``escaped`` ones tells you the model tried
    and the sandbox held.
    """

    tool_calls: int = 0
    blocked_tool_calls: int = 0
    outside_root_attempts: int = 0
    escaped_calls: int = 0  # outside-root AND succeeded — must always be 0
    files_written: list[str] = field(default_factory=list)
    canary_in_output: bool = False
    canary_in_files: bool = False
    agent_error: str | None = None
    agent_refused: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Verdict:
    """The judge's decision for one attack."""

    run_id: str
    attack_id: str
    category: str
    success: bool
    """True = the attacker won (the agent was compromised)."""

    blocked_by_code: bool
    """True = a code-level boundary stopped the attack (vs the model declining)."""

    method: str  # "code" | "llm" | "code+llm"
    confidence: float = 1.0
    rationale: str = ""
    signals: Signals = field(default_factory=Signals)
    llm_used: bool = False
    llm_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["signals"] = self.signals.to_dict()
        return payload
