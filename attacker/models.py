"""Attack and success-signal data model.

An :class:`Attack` is a complete, reproducible experiment: the workspace state
to set up, the message to send as the user, and — crucially — a machine-checkable
definition of what "the attack succeeded" means.

Declaring the success condition *with* the attack rather than inferring it later
is what lets three of the four categories be judged by code instead of by
another model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["Attack", "CATEGORIES", "DELIVERIES", "SIGNAL_TYPES", "AttackValidationError"]

CATEGORIES: tuple[str, ...] = (
    "prompt_injection",
    "goal_hijacking",
    "permission_escalation",
    "system_prompt_exfiltration",
)

DELIVERIES: tuple[str, ...] = (
    "direct",  # payload is in the user message itself
    "file",    # payload is in a file the agent is asked to read (indirect injection)
)

SIGNAL_TYPES: tuple[str, ...] = (
    "file_written",       # a file at `path` exists in the sandbox afterwards
    "text_contains",      # agent output contains any of `values`
    "tool_called",        # a tool named `name` was called (optionally with arg substring)
    "path_outside_root",  # a tool call targeted a path outside the sandbox root
    "canary_leak",        # the system-prompt canary appeared in output or a written file
    "any_of",             # at least one nested signal fired
    "all_of",             # every nested signal fired
)


class AttackValidationError(ValueError):
    """An attack definition is malformed."""


@dataclass
class Attack:
    """One red-team test case."""

    id: str
    category: str
    name: str
    prompt: str
    """The user message sent to the target agent."""

    origin: str = "seed"           # "seed" | "generated"
    parent_id: str | None = None   # seed this variant was derived from
    delivery: str = "direct"
    setup_files: dict[str, str] = field(default_factory=dict)
    """Files written into the sandbox before the run, relative to its root."""

    setup_symlinks: dict[str, str] = field(default_factory=dict)
    """Symlinks to create before the run: ``{link_name: target}``. Best effort."""

    payload_file: str | None = None
    """Which entry of ``setup_files`` carries the injected payload (for variants)."""

    objective: str = ""
    """Plain-English attacker goal. Fed to the LLM judge as the rubric's subject."""

    user_task: str = ""
    """The legitimate task the user actually asked for (goal-hijacking rubric)."""

    success_signal: dict[str, Any] | None = None
    """Code-checkable success condition. Required for code-judged categories."""

    judge: str = "code"  # "code" | "llm" | "code+llm"
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    # -------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Attack:
        known = {f for f in cls.__dataclass_fields__}  # noqa: SLF001
        unknown = set(data) - known
        if unknown:
            raise AttackValidationError(
                f"attack {data.get('id')!r} has unknown fields: {sorted(unknown)}"
            )
        attack = cls(**data)
        attack.validate()
        return attack

    def validate(self) -> None:
        if self.category not in CATEGORIES:
            raise AttackValidationError(
                f"attack {self.id!r}: unknown category {self.category!r}"
            )
        if self.delivery not in DELIVERIES:
            raise AttackValidationError(
                f"attack {self.id!r}: unknown delivery {self.delivery!r}"
            )
        if self.judge not in ("code", "llm", "code+llm"):
            raise AttackValidationError(f"attack {self.id!r}: unknown judge {self.judge!r}")
        if not self.prompt.strip():
            raise AttackValidationError(f"attack {self.id!r}: empty prompt")
        if self.delivery == "file" and not self.setup_files:
            raise AttackValidationError(
                f"attack {self.id!r}: delivery='file' requires setup_files"
            )
        if self.payload_file and self.payload_file not in self.setup_files:
            raise AttackValidationError(
                f"attack {self.id!r}: payload_file {self.payload_file!r} not in setup_files"
            )
        if "code" in self.judge and self.success_signal is None:
            raise AttackValidationError(
                f"attack {self.id!r}: judge={self.judge!r} requires a success_signal"
            )
        if self.success_signal is not None:
            _validate_signal(self.success_signal, self.id)
        if self.judge != "code" and not self.objective.strip():
            raise AttackValidationError(
                f"attack {self.id!r}: LLM-judged attacks need an 'objective'"
            )

    def with_payload(self, *, new_id: str, prompt: str, payload: str | None) -> Attack:
        """Clone this attack as a generated variant with new text."""
        files = dict(self.setup_files)
        if payload is not None and self.payload_file:
            files[self.payload_file] = payload
        return Attack(
            id=new_id,
            category=self.category,
            name=f"{self.name} (variant)",
            prompt=prompt,
            origin="generated",
            parent_id=self.id,
            delivery=self.delivery,
            setup_files=files,
            setup_symlinks=dict(self.setup_symlinks),
            payload_file=self.payload_file,
            objective=self.objective,
            user_task=self.user_task,
            success_signal=self.success_signal,
            judge=self.judge,
            notes="LLM-generated variant",
            tags=[*self.tags, "generated"],
        )


def _validate_signal(signal: dict[str, Any], attack_id: str) -> None:
    kind = signal.get("type")
    if kind not in SIGNAL_TYPES:
        raise AttackValidationError(
            f"attack {attack_id!r}: unknown success_signal type {kind!r}"
        )
    if kind in ("any_of", "all_of"):
        children = signal.get("signals")
        if not isinstance(children, list) or not children:
            raise AttackValidationError(
                f"attack {attack_id!r}: {kind} requires a non-empty 'signals' list"
            )
        for child in children:
            _validate_signal(child, attack_id)
    elif kind == "file_written":
        if not signal.get("path"):
            raise AttackValidationError(f"attack {attack_id!r}: file_written requires 'path'")
    elif kind == "text_contains":
        values = signal.get("values")
        if not isinstance(values, list) or not values:
            raise AttackValidationError(
                f"attack {attack_id!r}: text_contains requires a non-empty 'values' list"
            )
    elif kind == "tool_called":
        if not signal.get("name"):
            raise AttackValidationError(f"attack {attack_id!r}: tool_called requires 'name'")
