"""The agent under test.

Kept deliberately separate from the attacker, judge and runner so that the
container image needs only this package plus :mod:`common`, and so that a
defence layer (Milestone 5) can be inserted at :mod:`target_agent.guard`
without touching the loop.
"""

from target_agent.agent import AgentRun, TargetAgent, ToolCallRecord
from target_agent.guard import Guard, GuardDecision, NullGuard, load_guard
from target_agent.net import EgressViolation, assert_url_allowed
from target_agent.sandbox import SandboxViolation, resolve_in_sandbox
from target_agent.tools import Toolbox, ToolOutcome, tool_schemas

__all__ = [
    "AgentRun",
    "EgressViolation",
    "Guard",
    "GuardDecision",
    "NullGuard",
    "SandboxViolation",
    "TargetAgent",
    "ToolCallRecord",
    "ToolOutcome",
    "Toolbox",
    "assert_url_allowed",
    "load_guard",
    "resolve_in_sandbox",
    "tool_schemas",
]
