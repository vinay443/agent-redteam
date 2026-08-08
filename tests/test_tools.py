"""The tool layer must enforce the boundary and record blocked attempts.

These tests prove the property the metrics depend on: an escape attempt comes
back as ``ok=False, blocked=True, outside_root=True`` (never a silent success),
and a legitimate write lands inside the root.
"""

from __future__ import annotations

import pytest

from target_agent.tools import MAX_WRITE_BYTES, Toolbox


@pytest.fixture()
def toolbox(tmp_path):
    return Toolbox(root=tmp_path / "sandbox")


def test_write_then_read_roundtrip(toolbox):
    write = toolbox.execute("write_file", {"path": "out/hello.txt", "content": "hi there"})
    assert write.ok and not write.blocked
    read = toolbox.execute("read_file", {"path": "out/hello.txt"})
    assert read.ok
    assert "hi there" in read.content


def test_write_outside_root_is_blocked_not_silent(toolbox):
    outcome = toolbox.execute("write_file", {"path": "../escape.txt", "content": "pwn"})
    assert not outcome.ok
    assert outcome.blocked
    assert outcome.outside_root
    assert outcome.violation is not None
    # And nothing was written outside.
    assert not (toolbox.root.parent / "escape.txt").exists()


def test_read_absolute_outside_is_blocked(toolbox):
    import os

    target = "C:\\Windows\\win.ini" if os.name == "nt" else "/etc/passwd"
    outcome = toolbox.execute("read_file", {"path": target})
    assert not outcome.ok
    assert outcome.outside_root


def test_read_missing_file_is_not_a_boundary_violation(toolbox):
    outcome = toolbox.execute("read_file", {"path": "nope.txt"})
    assert not outcome.ok
    assert not outcome.outside_root
    assert outcome.error == "not_found"


def test_write_over_size_limit_is_refused(toolbox):
    big = "a" * (MAX_WRITE_BYTES + 1)
    outcome = toolbox.execute("write_file", {"path": "big.txt", "content": big})
    assert not outcome.ok
    assert outcome.blocked
    assert outcome.error == "write_too_large"


def test_unknown_tool_is_rejected(toolbox):
    outcome = toolbox.execute("rm_rf", {"path": "/"})
    assert not outcome.ok
    assert outcome.error == "unknown_tool"


def test_write_quota_is_enforced(tmp_path):
    box = Toolbox(root=tmp_path / "sandbox", max_files=2)
    assert box.execute("write_file", {"path": "a.txt", "content": "1"}).ok
    assert box.execute("write_file", {"path": "b.txt", "content": "2"}).ok
    third = box.execute("write_file", {"path": "c.txt", "content": "3"})
    assert not third.ok
    assert third.error == "write_quota_exhausted"
