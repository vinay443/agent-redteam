"""Environment-driven configuration.

Everything the lab needs is read once, here, into a frozen :class:`Settings`.
The target agent reads the same class inside the container (where only a subset
of the fields is populated), so there is a single definition of "what knobs
exist".
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Settings", "load_settings", "repo_root", "DEFAULT_EGRESS_ALLOWLIST"]

DEFAULT_EGRESS_ALLOWLIST: tuple[str, ...] = ("api.anthropic.com",)

# Container-side path of the bind mount. Must match docker-compose.yml.
CONTAINER_SANDBOX_ROOT = "/sandbox"


def repo_root() -> Path:
    """Directory containing this repository (the parent of ``common/``)."""
    return Path(__file__).resolve().parent.parent


def _load_dotenv_if_present() -> None:
    """Load ``<repo>/.env`` if python-dotenv is installed and the file exists.

    Deliberately optional: inside the container there is no ``.env`` and no
    python-dotenv, and the process must still start.
    """
    env_path = repo_root() / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_path, override=False)


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return tuple(part.strip().lower() for part in raw.split(",") if part.strip())


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the lab's configuration."""

    # --- credentials -------------------------------------------------------
    api_key: str | None
    base_url: str

    # --- models ------------------------------------------------------------
    target_model: str
    attacker_model: str
    judge_model: str

    # --- target agent behaviour -------------------------------------------
    target_max_turns: int
    target_max_tokens: int
    target_effort: str

    # --- safety ------------------------------------------------------------
    egress_allowlist: tuple[str, ...]
    web_fetch_enabled: bool

    # --- filesystem --------------------------------------------------------
    host_sandbox_dir: Path
    container_sandbox_root: str
    results_dir: Path

    # --- container ---------------------------------------------------------
    container_name: str
    compose_file: Path
    exec_timeout_seconds: int

    extra: dict[str, str] = field(default_factory=dict)

    # -- derived paths ------------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.results_dir / "results.sqlite3"

    def run_dir(self, run_id: str) -> Path:
        return self.results_dir / run_id

    def events_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "events.jsonl"

    def host_attack_dir(self, run_id: str, attack_id: str) -> Path:
        return self.host_sandbox_dir / "runs" / run_id / attack_id

    def container_attack_root(self, run_id: str, attack_id: str) -> str:
        # POSIX path inside the container; do not use os.path.join on Windows.
        return f"{self.container_sandbox_root}/runs/{run_id}/{attack_id}"

    def require_api_key(self) -> str:
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        return self.api_key


def load_settings(*, load_dotenv: bool = True) -> Settings:
    """Build :class:`Settings` from the process environment."""
    if load_dotenv:
        _load_dotenv_if_present()

    root = repo_root()
    return Settings(
        api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
        base_url=_env("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        target_model=_env("TARGET_MODEL", "claude-opus-5"),
        attacker_model=_env("ATTACKER_MODEL", "claude-opus-5"),
        judge_model=_env("JUDGE_MODEL", "claude-opus-5"),
        target_max_turns=_env_int("TARGET_MAX_TURNS", 8),
        target_max_tokens=_env_int("TARGET_MAX_TOKENS", 8000),
        target_effort=_env("TARGET_EFFORT", "medium"),
        egress_allowlist=_env_list("EGRESS_ALLOWLIST", DEFAULT_EGRESS_ALLOWLIST),
        # Deliberately off. See SAFETY.md — enabling a network tool for the
        # target agent is a capability decision, not a config tweak.
        web_fetch_enabled=_env_bool("WEB_FETCH_ENABLED", False),
        host_sandbox_dir=Path(_env("HOST_SANDBOX_DIR", str(root / "sandbox"))).resolve(),
        container_sandbox_root=_env("CONTAINER_SANDBOX_ROOT", CONTAINER_SANDBOX_ROOT),
        results_dir=Path(_env("RESULTS_DIR", str(root / "results"))).resolve(),
        container_name=_env("CONTAINER_NAME", "agent-redteam-target"),
        compose_file=Path(
            _env("COMPOSE_FILE", str(root / "docker" / "docker-compose.yml"))
        ).resolve(),
        exec_timeout_seconds=_env_int("EXEC_TIMEOUT_SECONDS", 300),
    )
