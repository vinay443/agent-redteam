"""Defence hook — the seam Milestone 5 slots into.

Today the only implementation is :class:`NullGuard`, which allows everything.
It exists now so that adding a real guard later is a *registration* change, not
a rewrite of the agent loop: the loop and the toolbox already call every hook.

Baseline runs must use ``NullGuard`` — the whole point of the lab is to measure
an undefended agent first, then measure the delta a guard buys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

__all__ = ["GuardDecision", "Guard", "NullGuard", "load_guard", "GUARDS"]


@dataclass(frozen=True)
class GuardDecision:
    """Outcome of a guard check.

    Attributes:
        allow: whether the action may proceed.
        reason: human-readable justification, logged when ``allow`` is False.
        replacement: optional sanitised text to substitute for the original.
    """

    allow: bool = True
    reason: str | None = None
    replacement: str | None = None

    @classmethod
    def allowed(cls) -> GuardDecision:
        return cls(allow=True)

    @classmethod
    def blocked(cls, reason: str) -> GuardDecision:
        return cls(allow=False, reason=reason)


@runtime_checkable
class Guard(Protocol):
    """Interception points around the agent loop.

    A guard sees the untrusted user turn, every proposed tool call, every tool
    result before it re-enters the context, and the model's final text.
    """

    name: str

    def on_user_message(self, text: str) -> GuardDecision: ...

    def on_tool_call(self, name: str, arguments: dict[str, Any]) -> GuardDecision: ...

    def on_tool_result(self, name: str, content: str) -> GuardDecision: ...

    def on_model_output(self, text: str) -> GuardDecision: ...


class NullGuard:
    """Pass-through guard. The undefended baseline."""

    name = "none"

    def on_user_message(self, text: str) -> GuardDecision:  # noqa: ARG002
        return GuardDecision.allowed()

    def on_tool_call(self, name: str, arguments: dict[str, Any]) -> GuardDecision:  # noqa: ARG002
        return GuardDecision.allowed()

    def on_tool_result(self, name: str, content: str) -> GuardDecision:  # noqa: ARG002
        return GuardDecision.allowed()

    def on_model_output(self, text: str) -> GuardDecision:  # noqa: ARG002
        return GuardDecision.allowed()


GUARDS: dict[str, type] = {"none": NullGuard}


def load_guard(name: str) -> Guard:
    """Instantiate a registered guard by name.

    Milestone 5 registers additional implementations in :data:`GUARDS`; nothing
    else in the codebase needs to change.
    """
    try:
        factory = GUARDS[name]
    except KeyError:
        known = ", ".join(sorted(GUARDS))
        raise ValueError(f"unknown guard {name!r}; known guards: {known}") from None
    return factory()
