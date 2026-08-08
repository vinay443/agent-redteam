"""Timestamps.

Every record the lab writes carries a UTC ISO-8601 timestamp so that a
transcript, a tool call and a judge verdict can be interleaved after the fact.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

__all__ = ["utcnow_iso", "monotonic_ms", "iso_to_epoch"]


def utcnow_iso() -> str:
    """Current UTC time as ``2026-08-08T11:18:03.412Z``."""
    return (
        datetime.now(UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def monotonic_ms() -> float:
    """Monotonic clock in milliseconds, for durations (never for timestamps)."""
    return time.monotonic() * 1000.0


def iso_to_epoch(value: str) -> float:
    """Parse an ISO-8601 timestamp produced by :func:`utcnow_iso` to epoch seconds."""
    normalised = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalised).timestamp()
