"""Structured JSON-lines event logging.

The lab's auditability requirement is simple: *nothing is a black box*. Every
attack selected, every agent turn, every tool call and every judge verdict is
emitted here as one JSON object per line, always with a timestamp and the
owning ``run_id``.

Two sinks are supported simultaneously:

* a file (``results/<run_id>/events.jsonl``) — the durable audit trail;
* a stream (stderr by default) — so that a container's logs are self-describing.

Inside the target container stdout is reserved for the agent result payload, so
the logger there writes to stderr only.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any, TextIO

from common.timeutil import utcnow_iso

__all__ = ["EventLogger", "NullLogger"]


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of ``value`` into something ``json.dumps`` accepts."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    # Pydantic models (Anthropic SDK content blocks) and dataclass-likes.
    for attr, kwargs in (("model_dump", {"mode": "json"}), ("to_dict", {}), ("_asdict", {})):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                return _jsonable(method(**kwargs))
            except Exception:  # noqa: BLE001 - logging must never raise
                break
    if hasattr(value, "__dict__"):
        return {k: _jsonable(v) for k, v in vars(value).items() if not k.startswith("_")}
    return repr(value)


class EventLogger:
    """Append-only JSONL logger. Thread-safe; never raises on the caller."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        stream: TextIO | None = sys.stderr,
        run_id: str | None = None,
        component: str = "lab",
        echo: bool = True,
    ) -> None:
        self.run_id = run_id
        self.component = component
        self._stream = stream if echo else None
        self._lock = threading.Lock()
        self._fh: TextIO | None = None
        self.path: Path | None = None
        if path is not None:
            self.path = Path(path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        """Write one event record and return it."""
        record: dict[str, Any] = {
            "ts": utcnow_iso(),
            "run_id": self.run_id,
            "component": self.component,
            "event": event,
        }
        record.update({k: _jsonable(v) for k, v in fields.items()})
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            if self._fh is not None:
                self._fh.write(line + "\n")
                self._fh.flush()
            if self._stream is not None:
                try:
                    self._stream.write(line + "\n")
                    self._stream.flush()
                except Exception:  # noqa: BLE001, S110 - a broken pipe must not kill a run
                    pass
        return record

    def child(self, component: str) -> EventLogger:
        """A logger sharing this one's sinks but tagged with a new component name."""
        clone = EventLogger.__new__(EventLogger)
        clone.run_id = self.run_id
        clone.component = component
        clone._stream = self._stream
        clone._lock = self._lock
        clone._fh = self._fh
        clone.path = self.path
        return clone

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    def __enter__(self) -> EventLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class NullLogger(EventLogger):
    """Drops everything. Used in unit tests and library-style embedding."""

    def __init__(self, run_id: str | None = None, component: str = "null") -> None:
        super().__init__(path=None, stream=None, run_id=run_id, component=component, echo=False)

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:  # noqa: D102
        return {"event": event, **fields}
