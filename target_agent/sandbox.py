"""Filesystem containment for the target agent's tools.

This module is the innermost of the lab's three safety layers (container →
tool boundary → system prompt) and the only one that is *load-bearing*: the
container limits blast radius and the prompt states intent, but this is what
actually decides whether a path is reachable.

Design rules, all of which the test suite pins:

1. **Resolve, then compare.** Containment is decided on ``os.path.realpath`` of
   both the root and the candidate — never on string prefixes. String matching
   is defeated by ``..`` segments, by symlinks, and by a sibling directory whose
   name merely starts with the root's name (``/sandbox-evil`` vs ``/sandbox``).
2. **Reject, don't clamp.** A path outside the root raises
   :class:`SandboxViolation`. Silently rewriting it to something safe would hide
   the attack from the metrics, which is the whole point of the lab.
3. **Symlinks resolve before the check.** ``realpath`` follows every link in the
   path, so a symlink planted inside the sandbox that points at ``/etc`` is
   caught, not followed.
4. **The check is re-run after a write.** Defence in depth against a race
   between resolution and open (see :func:`assert_within`).
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "SandboxViolation",
    "resolve_in_sandbox",
    "assert_within",
    "ensure_sandbox_root",
    "relative_to_root",
]


class SandboxViolation(Exception):
    """Raised when a requested path resolves outside the sandbox root.

    Carries the requested and resolved paths so that the tool layer can log a
    precise, attacker-visible-free record of what was attempted.
    """

    def __init__(
        self,
        message: str,
        *,
        requested: str | None = None,
        resolved: str | None = None,
        root: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.requested = requested
        self.resolved = resolved
        self.root = root

    def as_dict(self) -> dict[str, str | None]:
        return {
            "message": self.message,
            "requested": self.requested,
            "resolved": self.resolved,
            "root": self.root,
        }


def ensure_sandbox_root(root: str | os.PathLike[str]) -> Path:
    """Return the fully resolved sandbox root, creating it if needed.

    Raises :class:`SandboxViolation` if the path exists but is not a directory —
    a mis-provisioned root must fail loudly rather than degrade into "no
    containment".
    """
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    real = Path(os.path.realpath(root_path))
    if not real.is_dir():
        raise SandboxViolation(
            "sandbox root is not a directory",
            requested=str(root),
            resolved=str(real),
        )
    return real


def _is_within(root_real: Path, candidate_real: Path) -> bool:
    """True when ``candidate_real`` is ``root_real`` or lives beneath it.

    Uses path-component comparison (``Path.parents``) rather than
    ``str.startswith``: ``/sandbox-evil`` is not inside ``/sandbox``, and a
    Windows path on another drive simply never matches.
    """
    if candidate_real == root_real:
        return True
    return root_real in candidate_real.parents


def resolve_in_sandbox(
    root: str | os.PathLike[str],
    candidate: str,
    *,
    must_exist: bool = False,
) -> Path:
    """Resolve ``candidate`` against ``root`` and prove it stays inside.

    ``candidate`` may be relative (joined to the root) or absolute (accepted
    only if it already resolves inside the root). Returns the resolved
    :class:`~pathlib.Path`.

    Raises:
        SandboxViolation: if the path is malformed or escapes the root.
        FileNotFoundError: if ``must_exist`` and the resolved path is absent.
    """
    root_real = ensure_sandbox_root(root)

    if not isinstance(candidate, str):
        raise SandboxViolation("path must be a string", requested=repr(candidate))
    if candidate.strip() == "":
        raise SandboxViolation("path must not be empty", requested=candidate, root=str(root_real))
    if "\x00" in candidate:
        raise SandboxViolation(
            "path must not contain a NUL byte",
            requested=candidate.replace("\x00", "\\x00"),
            root=str(root_real),
        )

    raw = Path(candidate)
    joined = raw if raw.is_absolute() else root_real / raw

    # realpath resolves symlinks across the whole path and normalises ".." even
    # when trailing components do not yet exist (needed for write_file).
    resolved = Path(os.path.realpath(joined))

    if not _is_within(root_real, resolved):
        raise SandboxViolation(
            "path resolves outside the sandbox root",
            requested=candidate,
            resolved=str(resolved),
            root=str(root_real),
        )

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"no such file inside the sandbox: {candidate}")

    return resolved


def assert_within(root: str | os.PathLike[str], path: str | os.PathLike[str]) -> Path:
    """Re-verify that an already-resolved ``path`` is inside ``root``.

    Called *after* a write completes. Between resolution and ``open()`` an
    attacker with concurrent access to the sandbox could in principle swap a
    directory component for a symlink; this second check turns that race into a
    detected violation instead of a silent escape.
    """
    root_real = Path(os.path.realpath(root))
    resolved = Path(os.path.realpath(path))
    if not _is_within(root_real, resolved):
        raise SandboxViolation(
            "post-write verification failed: path is outside the sandbox root",
            requested=str(path),
            resolved=str(resolved),
            root=str(root_real),
        )
    return resolved


def relative_to_root(root: str | os.PathLike[str], path: str | os.PathLike[str]) -> str:
    """POSIX-style path of ``path`` relative to ``root`` (for logs and reports)."""
    root_real = Path(os.path.realpath(root))
    resolved = Path(os.path.realpath(path))
    try:
        return resolved.relative_to(root_real).as_posix()
    except ValueError:
        return resolved.as_posix()
