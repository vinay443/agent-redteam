"""The mandated safety test: path traversal (and friends) must be blocked.

These are the assertions SAFETY.md points at. They exercise
:func:`target_agent.sandbox.resolve_in_sandbox` directly — the one function that
actually decides containment — with the traversal, absolute-path, symlink and
prefix-confusion vectors an attacker would use.
"""

from __future__ import annotations

import os
import sys

import pytest

from target_agent.sandbox import (
    SandboxViolation,
    assert_within,
    relative_to_root,
    resolve_in_sandbox,
)


@pytest.fixture()
def sandbox(tmp_path):
    root = tmp_path / "sandbox"
    root.mkdir()
    (root / "notes.txt").write_text("hello", encoding="utf-8")
    return root


# --- the headline requirement ------------------------------------------------

def test_parent_traversal_is_blocked(sandbox):
    with pytest.raises(SandboxViolation):
        resolve_in_sandbox(sandbox, "../escape.txt")


def test_deep_traversal_is_blocked(sandbox):
    with pytest.raises(SandboxViolation):
        resolve_in_sandbox(sandbox, "../../../../etc/passwd")


def test_absolute_path_outside_is_blocked(sandbox):
    target = "C:\\Windows\\System32\\drivers\\etc\\hosts" if os.name == "nt" else "/etc/passwd"
    with pytest.raises(SandboxViolation):
        resolve_in_sandbox(sandbox, target)


def test_traversal_hidden_in_the_middle_is_blocked(sandbox):
    with pytest.raises(SandboxViolation):
        resolve_in_sandbox(sandbox, "notes/../../escape.txt")


def test_traversal_uses_realpath_not_string_prefix(sandbox):
    # A sibling directory whose name starts with the sandbox name must NOT be
    # considered "inside" — this is exactly what str.startswith() gets wrong.
    sibling = sandbox.parent / (sandbox.name + "-evil")
    sibling.mkdir()
    (sibling / "loot.txt").write_text("x", encoding="utf-8")
    with pytest.raises(SandboxViolation):
        resolve_in_sandbox(sandbox, str(sibling / "loot.txt"))


# --- symlink escape ----------------------------------------------------------

@pytest.mark.skipif(
    os.name == "nt" and sys.version_info < (3, 8),
    reason="symlink creation needs privilege on old Windows",
)
def test_symlink_escaping_root_is_blocked(sandbox, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("top secret", encoding="utf-8")
    link = sandbox / "escape-link"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlinks on this host")
    with pytest.raises(SandboxViolation):
        resolve_in_sandbox(sandbox, "escape-link/secret.txt")


def test_symlink_pointing_inside_root_is_allowed(sandbox):
    inner = sandbox / "inner"
    inner.mkdir()
    (inner / "ok.txt").write_text("fine", encoding="utf-8")
    link = sandbox / "inner-link"
    try:
        link.symlink_to(inner)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlinks on this host")
    resolved = resolve_in_sandbox(sandbox, "inner-link/ok.txt", must_exist=True)
    assert resolved.read_text(encoding="utf-8") == "fine"


# --- malformed input ---------------------------------------------------------

def test_empty_path_is_blocked(sandbox):
    with pytest.raises(SandboxViolation):
        resolve_in_sandbox(sandbox, "")


def test_nul_byte_is_blocked(sandbox):
    with pytest.raises(SandboxViolation):
        resolve_in_sandbox(sandbox, "notes\x00.txt")


def test_non_string_path_is_blocked(sandbox):
    with pytest.raises(SandboxViolation):
        resolve_in_sandbox(sandbox, None)  # type: ignore[arg-type]


# --- happy paths -------------------------------------------------------------

def test_relative_path_inside_root_is_allowed(sandbox):
    resolved = resolve_in_sandbox(sandbox, "notes.txt", must_exist=True)
    assert resolved.name == "notes.txt"
    assert resolve_in_sandbox.__module__  # sanity


def test_nested_write_path_is_allowed_even_if_missing(sandbox):
    resolved = resolve_in_sandbox(sandbox, "reports/2026/summary.md")
    assert resolved.name == "summary.md"
    assert not resolved.exists()  # must_exist defaults to False


def test_must_exist_raises_for_missing_file(sandbox):
    with pytest.raises(FileNotFoundError):
        resolve_in_sandbox(sandbox, "nope.txt", must_exist=True)


def test_root_itself_is_within_root(sandbox):
    resolved = resolve_in_sandbox(sandbox, ".")
    assert os.path.realpath(resolved) == os.path.realpath(sandbox)


def test_assert_within_catches_post_write_escape(sandbox, tmp_path):
    outside = tmp_path / "outside" / "x.txt"
    outside.parent.mkdir(parents=True)
    outside.write_text("x", encoding="utf-8")
    with pytest.raises(SandboxViolation):
        assert_within(sandbox, outside)


def test_relative_to_root_is_posix(sandbox):
    resolved = resolve_in_sandbox(sandbox, "a/b/c.txt")
    assert relative_to_root(sandbox, resolved) == "a/b/c.txt"
