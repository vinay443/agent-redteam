"""Identifier generation.

Run ids are sortable (timestamp-prefixed) so that ``ls results/`` and
``ORDER BY run_id`` both give chronological order.
"""

from __future__ import annotations

import re
import secrets
import time

__all__ = ["new_run_id", "new_attack_id", "new_id", "slugify"]

_UNSAFE = re.compile(r"[^a-zA-Z0-9._-]+")


def new_run_id(prefix: str = "run") -> str:
    """e.g. ``run-20260808T111803Z-4f9a12``."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{prefix}-{stamp}-{secrets.token_hex(3)}"


def new_id(prefix: str) -> str:
    """e.g. ``atk-9f2c1b04ae77``."""
    return f"{prefix}-{secrets.token_hex(6)}"


def new_attack_id(seed_id: str, index: int) -> str:
    """Deterministic-looking id for a generated variant of ``seed_id``."""
    return f"{slugify(seed_id)}-v{index:02d}-{secrets.token_hex(2)}"


def slugify(value: str) -> str:
    """Reduce ``value`` to something safe for a filesystem path segment."""
    cleaned = _UNSAFE.sub("-", value).strip("-.")
    return cleaned[:80] or "unnamed"
