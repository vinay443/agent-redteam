"""Container entrypoint for one attack execution.

Protocol (kept dead simple so the runner needs no port, no server, no shared
state — just ``docker exec -i``):

* **stdin**  — one JSON object: ``{"run_id", "attack_id", "category", "prompt", ...}``
* **stdout** — the :class:`~target_agent.agent.AgentRun` as JSON, fenced by
  ``<<<AGENT_RESULT_BEGIN>>>`` / ``<<<AGENT_RESULT_END>>>`` markers so stray
  output can never corrupt parsing.
* **stderr** — JSONL event log (one line per tool call, model turn, etc.).

``--selftest`` runs the containment assertions *inside the container* with no
API key and no network. That is the in-container half of the verification
procedure documented in SAFETY.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common.config import load_settings
from common.llm import make_client
from common.logging import EventLogger
from target_agent.agent import TargetAgent
from target_agent.guard import load_guard
from target_agent.prompts import new_canary
from target_agent.sandbox import SandboxViolation, ensure_sandbox_root, resolve_in_sandbox
from target_agent.tools import Toolbox

RESULT_BEGIN = "<<<AGENT_RESULT_BEGIN>>>"
RESULT_END = "<<<AGENT_RESULT_END>>>"


def _selftest(sandbox_root: str) -> int:
    """Prove containment holds in this container. No API key required."""
    root = ensure_sandbox_root(sandbox_root)
    checks: list[tuple[str, bool, str]] = []

    def expect_violation(label: str, path: str) -> None:
        try:
            resolved = resolve_in_sandbox(root, path)
        except SandboxViolation:
            checks.append((label, True, "blocked"))
        except FileNotFoundError:
            checks.append((label, True, "blocked (missing)"))
        else:
            checks.append((label, False, f"ESCAPED to {resolved}"))

    def expect_ok(label: str, path: str) -> None:
        try:
            resolved = resolve_in_sandbox(root, path)
        except Exception as exc:  # noqa: BLE001
            checks.append((label, False, f"unexpectedly rejected: {exc}"))
        else:
            checks.append((label, True, str(resolved)))

    expect_violation("parent traversal", "../escape.txt")
    expect_violation("deep traversal", "../../../../etc/passwd")
    expect_violation("absolute outside", "/etc/passwd")
    expect_violation("absolute tmp", "/tmp/pwn.txt")  # noqa: S108 - asserting this is BLOCKED
    expect_violation("mixed traversal", "notes/../../escape.txt")
    expect_violation("root sibling prefix", str(root) + "-evil/x.txt")
    expect_violation("empty path", "")
    expect_ok("nested write path", "reports/summary.md")
    expect_ok("plain path", "notes.txt")

    # Symlink escape: only meaningful where symlinks can be created.
    link = root / "escape-link"
    try:
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(Path("/etc"))
        expect_violation("symlink escape", "escape-link/passwd")
        link.unlink()
    except (OSError, NotImplementedError) as exc:
        checks.append(("symlink escape", True, f"skipped ({type(exc).__name__}: {exc})"))

    failures = [c for c in checks if not c[1]]
    for label, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}: {detail}", file=sys.stderr)
    print(
        f"selftest: {len(checks) - len(failures)}/{len(checks)} passed",
        file=sys.stderr,
    )
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="target_agent",
        description="Run one attack against the sandboxed target agent.",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Verify sandbox containment inside this container and exit.",
    )
    parser.add_argument(
        "--sandbox-root",
        default=None,
        help="Override AGENT_SANDBOX_ROOT.",
    )
    args = parser.parse_args(argv)

    settings = load_settings(load_dotenv=False)
    import os

    sandbox_root = (
        args.sandbox_root
        or os.environ.get("AGENT_SANDBOX_ROOT")
        or settings.container_sandbox_root
    )

    if args.selftest:
        return _selftest(sandbox_root)

    raw = sys.stdin.read()
    if not raw.strip():
        print("no request on stdin", file=sys.stderr)
        return 2
    request = json.loads(raw)

    run_id = request.get("run_id", "adhoc")
    attack_id = request.get("attack_id", "adhoc")
    category = request.get("category", "unknown")
    prompt = request["prompt"]

    logger = EventLogger(path=None, stream=sys.stderr, run_id=run_id, component="target_agent")

    try:
        client = make_client(settings)
        toolbox = Toolbox(
            root=ensure_sandbox_root(sandbox_root),
            logger=logger,
            guard=load_guard(request.get("guard", "none")),
        )
        agent = TargetAgent(
            client=client,
            toolbox=toolbox,
            model=request.get("model", settings.target_model),
            sandbox_root=sandbox_root,
            canary=request.get("canary") or new_canary(),
            max_turns=int(request.get("max_turns", settings.target_max_turns)),
            max_tokens=int(request.get("max_tokens", settings.target_max_tokens)),
            effort=request.get("effort", settings.target_effort),
            guard=load_guard(request.get("guard", "none")),
            logger=logger,
        )
        run = agent.run(prompt, run_id=run_id, attack_id=attack_id, category=category)
        payload = run.to_dict()
    except Exception as exc:  # noqa: BLE001 - always emit a parseable result
        logger.emit("fatal", error=f"{type(exc).__name__}: {exc}")
        payload = {
            "run_id": run_id,
            "attack_id": attack_id,
            "category": category,
            "error": f"{type(exc).__name__}: {exc}",
            "tool_calls": [],
            "transcript": [],
            "final_text": "",
        }

    sys.stdout.write(RESULT_BEGIN + "\n")
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    sys.stdout.write(RESULT_END + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
