"""Manage the target container and dispatch one attack at a time.

The runner owns the container lifecycle: build the image, start it with the
resource-limited compose config, run each attack via ``docker exec -i``, and
tear it down. The agent never listens on a socket; a request is a JSON blob on
stdin and the reply is a fenced JSON blob on stdout. This keeps the trust
boundary a process boundary.

If Docker is unavailable, :func:`DockerController.available` returns False and
the runner falls back to in-process execution (clearly labelled in the results),
so the lab is still usable for development without a daemon.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

from common.config import Settings
from common.logging import EventLogger, NullLogger
from target_agent.__main__ import RESULT_BEGIN, RESULT_END

__all__ = ["DockerController", "DockerError", "ExecResult"]


class DockerError(RuntimeError):
    """A docker/compose command failed."""


@dataclass
class ExecResult:
    returncode: int
    stdout: str
    stderr: str


class DockerController:
    """Thin wrapper over docker compose + docker exec for the target service."""

    def __init__(self, settings: Settings, logger: EventLogger | None = None) -> None:
        self.settings = settings
        self.logger = logger or NullLogger()
        self.compose_file = settings.compose_file
        self.container = settings.container_name
        self._compose_cmd = self._detect_compose()

    # -- capability detection ----------------------------------------------

    @staticmethod
    def docker_available() -> bool:
        if shutil.which("docker") is None:
            return False
        try:
            proc = subprocess.run(  # noqa: S603 - docker resolved from PATH by design
                ["docker", "info", "--format", "{{.ServerVersion}}"],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return proc.returncode == 0

    def available(self) -> bool:
        return self._compose_cmd is not None and self.docker_available()

    @staticmethod
    def _detect_compose() -> list[str] | None:
        if shutil.which("docker") is not None:
            probe = subprocess.run(  # noqa: S603 - docker resolved from PATH by design
                ["docker", "compose", "version"],  # noqa: S607
                capture_output=True,
                text=True,
            )
            if probe.returncode == 0:
                return ["docker", "compose"]
        if shutil.which("docker-compose") is not None:
            return ["docker-compose"]
        return None

    # -- lifecycle ----------------------------------------------------------

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("CONTAINER_NAME", self.container)
        # Ensure the container writes files owned by the invoking host user on
        # Linux, so ./sandbox stays cleanable without sudo.
        env.setdefault("AGENT_UID", str(_current_uid()))
        env.setdefault("AGENT_GID", str(_current_gid()))
        return env

    def _run_compose(self, *args: str, timeout: int = 900) -> ExecResult:
        if self._compose_cmd is None:
            raise DockerError("docker compose is not available")
        cmd = [*self._compose_cmd, "-f", str(self.compose_file), *args]
        self.logger.emit("docker_compose", argv=cmd)
        try:
            proc = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=self._env(),
                cwd=str(self.compose_file.parent),
            )
        except subprocess.TimeoutExpired as exc:
            raise DockerError(f"compose {' '.join(args)} timed out") from exc
        if proc.returncode != 0:
            raise DockerError(
                f"compose {' '.join(args)} failed ({proc.returncode}): {proc.stderr.strip()}"
            )
        return ExecResult(proc.returncode, proc.stdout, proc.stderr)

    def build(self) -> None:
        self._run_compose("build")

    def up(self) -> None:
        self.settings.host_sandbox_dir.mkdir(parents=True, exist_ok=True)
        self._run_compose("up", "-d", "--remove-orphans")
        self.logger.emit("container_up", container=self.container)

    def down(self) -> None:
        try:
            self._run_compose("down", "-v", timeout=120)
            self.logger.emit("container_down", container=self.container)
        except DockerError as exc:  # teardown failures should not mask results
            self.logger.emit("container_down_failed", error=str(exc))

    def selftest(self) -> ExecResult:
        """Run the in-container containment self-test."""
        return self.exec_json_stdin(
            ["python", "-m", "target_agent", "--selftest"], stdin="", timeout=60
        )

    # -- one attack ---------------------------------------------------------

    def run_attack(
        self,
        request: dict[str, Any],
        *,
        sandbox_root: str | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Execute one attack inside the container; return the parsed AgentRun.

        ``sandbox_root`` is passed as ``--sandbox-root`` so each attack runs
        against its own clean subdirectory of the bind-mounted ``/sandbox``.
        """
        payload = json.dumps(request, ensure_ascii=False)
        argv = ["python", "-m", "target_agent"]
        if sandbox_root:
            argv += ["--sandbox-root", sandbox_root]
        exec_result = self.exec_json_stdin(
            argv,
            stdin=payload,
            timeout=timeout or self.settings.exec_timeout_seconds,
        )
        return self._parse_result(exec_result, request)

    def exec_json_stdin(
        self, argv: list[str], *, stdin: str, timeout: int
    ) -> ExecResult:
        cmd = ["docker", "exec", "-i", self.container, *argv]
        try:
            proc = subprocess.run(  # noqa: S603
                cmd,
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=self._env(),
            )
        except subprocess.TimeoutExpired as exc:
            raise DockerError(f"docker exec timed out after {timeout}s") from exc
        return ExecResult(proc.returncode, proc.stdout, proc.stderr)

    @staticmethod
    def _parse_result(exec_result: ExecResult, request: dict[str, Any]) -> dict[str, Any]:
        out = exec_result.stdout
        begin = out.find(RESULT_BEGIN)
        end = out.find(RESULT_END)
        if begin == -1 or end == -1 or end < begin:
            raise DockerError(
                "could not find result markers in container stdout; "
                f"stderr tail: {exec_result.stderr[-500:]!r}"
            )
        blob = out[begin + len(RESULT_BEGIN) : end].strip()
        try:
            return json.loads(blob)
        except json.JSONDecodeError as exc:
            raise DockerError(f"container returned invalid JSON: {exc}") from exc


def _current_uid() -> int:
    getuid = getattr(os, "getuid", None)
    return getuid() if callable(getuid) else 10001


def _current_gid() -> int:
    getgid = getattr(os, "getgid", None)
    return getgid() if callable(getgid) else 10001
