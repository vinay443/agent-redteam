"""In-process attack execution — the fallback when Docker is unavailable.

This runs the *exact same* agent code path as the container (``TargetAgent`` over
``Toolbox`` rooted at the per-attack sandbox directory). The only thing missing
is the container's runtime isolation, so every result produced this way is
tagged ``sandboxed=False`` and the report surfaces a warning.

Use it for development and for CI without a Docker daemon; never treat its
numbers as a substitute for a real containerised run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from common.config import Settings
from common.llm import make_client
from common.logging import EventLogger, NullLogger
from target_agent.agent import TargetAgent
from target_agent.guard import load_guard
from target_agent.sandbox import ensure_sandbox_root
from target_agent.tools import Toolbox

__all__ = ["LocalExecutor"]


class LocalExecutor:
    """Executes attacks in the host process against a real sandbox directory."""

    def __init__(self, settings: Settings, logger: EventLogger | None = None) -> None:
        self.settings = settings
        self.logger = logger or NullLogger()
        self._client = make_client(settings)

    def run_attack(self, request: dict[str, Any], *, sandbox_dir: str | Path) -> dict[str, Any]:
        root = ensure_sandbox_root(sandbox_dir)
        agent_logger = self.logger.child("target_agent")
        toolbox = Toolbox(
            root=root,
            logger=agent_logger,
            guard=load_guard(request.get("guard", "none")),
        )
        agent = TargetAgent(
            client=self._client,
            toolbox=toolbox,
            model=request.get("model", self.settings.target_model),
            sandbox_root=str(root),
            canary=request.get("canary"),
            max_turns=int(request.get("max_turns", self.settings.target_max_turns)),
            max_tokens=int(request.get("max_tokens", self.settings.target_max_tokens)),
            effort=request.get("effort", self.settings.target_effort),
            guard=load_guard(request.get("guard", "none")),
            logger=agent_logger,
        )
        run = agent.run(
            request["prompt"],
            run_id=request.get("run_id", "adhoc"),
            attack_id=request.get("attack_id", "adhoc"),
            category=request.get("category", "unknown"),
        )
        return run.to_dict()
