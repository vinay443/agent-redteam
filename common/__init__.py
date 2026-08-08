"""Shared, dependency-light helpers used by every component of the lab.

Only :mod:`common.llm` imports the Anthropic SDK; the rest is stdlib-only so
that the container image and the test suite can import them cheaply.
"""

from common.ids import new_attack_id, new_id, new_run_id
from common.logging import EventLogger, NullLogger
from common.timeutil import utcnow_iso

__all__ = [
    "EventLogger",
    "NullLogger",
    "new_attack_id",
    "new_id",
    "new_run_id",
    "utcnow_iso",
]
