"""Results persistence: SQLite (primary) plus a JSONL mirror.

Every campaign writes to two places:

* ``results/results.sqlite3`` — the queryable store the report reads;
* ``results/<run_id>/results.jsonl`` — a flat, greppable mirror, one line per
  attack, so a run is inspectable with ``cat`` and never locked behind the DB.

The schema keeps one row per attack with the full transcript and tool-call log
inline as JSON. That is intentional: an attack result is only ever read whole,
and keeping it in one row means a single ``SELECT *`` reconstructs everything the
report or a human needs.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.timeutil import utcnow_iso

__all__ = ["ResultStore", "AttackResult"]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,
    started_at        TEXT NOT NULL,
    finished_at       TEXT,
    target_model      TEXT,
    attacker_model    TEXT,
    judge_model       TEXT,
    guard             TEXT,
    categories        TEXT,
    n_per_category    INTEGER,
    prompt_sha256     TEXT,
    settings          TEXT
);

CREATE TABLE IF NOT EXISTS attacks (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT NOT NULL,
    attack_id         TEXT NOT NULL,
    category          TEXT NOT NULL,
    origin            TEXT,
    parent_id         TEXT,
    delivery          TEXT,
    name              TEXT,
    judge_method      TEXT,
    success           INTEGER NOT NULL,
    blocked_by_code   INTEGER NOT NULL,
    confidence        REAL,
    rationale         TEXT,
    escaped_calls     INTEGER,
    blocked_tool_calls INTEGER,
    outside_root_attempts INTEGER,
    tool_calls        INTEGER,
    canary_in_output  INTEGER,
    canary_in_files   INTEGER,
    agent_refused     INTEGER,
    agent_error       TEXT,
    llm_used          INTEGER,
    duration_ms       REAL,
    created_at        TEXT NOT NULL,
    attack_json       TEXT NOT NULL,
    run_json          TEXT NOT NULL,
    verdict_json      TEXT NOT NULL,
    UNIQUE(run_id, attack_id)
);

CREATE INDEX IF NOT EXISTS idx_attacks_run ON attacks(run_id);
CREATE INDEX IF NOT EXISTS idx_attacks_cat ON attacks(category);
CREATE INDEX IF NOT EXISTS idx_attacks_success ON attacks(success);
"""


@dataclass
class AttackResult:
    """One row of the ``attacks`` table, reassembled for the report."""

    run_id: str
    attack_id: str
    category: str
    success: bool
    blocked_by_code: bool
    judge_method: str
    confidence: float
    rationale: str
    origin: str
    escaped_calls: int
    blocked_tool_calls: int
    outside_root_attempts: int
    tool_calls: int
    canary_in_output: bool
    canary_in_files: bool
    agent_refused: bool
    agent_error: str | None
    duration_ms: float
    attack: dict[str, Any]
    run: dict[str, Any]
    verdict: dict[str, Any]


class ResultStore:
    """SQLite-backed result store with a JSONL mirror per run."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- writing ------------------------------------------------------------

    def record_run(self, run_id: str, meta: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO runs (
                run_id, started_at, finished_at, target_model, attacker_model,
                judge_model, guard, categories, n_per_category, prompt_sha256, settings
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                finished_at=excluded.finished_at,
                categories=excluded.categories,
                prompt_sha256=excluded.prompt_sha256,
                settings=excluded.settings
            """,
            (
                run_id,
                meta.get("started_at", utcnow_iso()),
                meta.get("finished_at"),
                meta.get("target_model"),
                meta.get("attacker_model"),
                meta.get("judge_model"),
                meta.get("guard"),
                json.dumps(meta.get("categories", [])),
                meta.get("n_per_category"),
                meta.get("prompt_sha256"),
                json.dumps(meta.get("settings", {})),
            ),
        )
        self._conn.commit()

    def finish_run(self, run_id: str, *, finished_at: str | None = None) -> None:
        self._conn.execute(
            "UPDATE runs SET finished_at = ? WHERE run_id = ?",
            (finished_at or utcnow_iso(), run_id),
        )
        self._conn.commit()

    def record_attack(
        self,
        *,
        run_id: str,
        attack: dict[str, Any],
        run: dict[str, Any],
        verdict: dict[str, Any],
        jsonl_path: str | Path | None = None,
    ) -> None:
        signals = verdict.get("signals", {})
        self._conn.execute(
            """
            INSERT INTO attacks (
                run_id, attack_id, category, origin, parent_id, delivery, name,
                judge_method, success, blocked_by_code, confidence, rationale,
                escaped_calls, blocked_tool_calls, outside_root_attempts, tool_calls,
                canary_in_output, canary_in_files, agent_refused, agent_error,
                llm_used, duration_ms, created_at, attack_json, run_json, verdict_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(run_id, attack_id) DO UPDATE SET
                success=excluded.success,
                blocked_by_code=excluded.blocked_by_code,
                verdict_json=excluded.verdict_json,
                run_json=excluded.run_json
            """,
            (
                run_id,
                attack.get("id"),
                attack.get("category"),
                attack.get("origin"),
                attack.get("parent_id"),
                attack.get("delivery"),
                attack.get("name"),
                verdict.get("method"),
                1 if verdict.get("success") else 0,
                1 if verdict.get("blocked_by_code") else 0,
                float(verdict.get("confidence", 0.0)),
                verdict.get("rationale", ""),
                int(signals.get("escaped_calls", 0)),
                int(signals.get("blocked_tool_calls", 0)),
                int(signals.get("outside_root_attempts", 0)),
                int(signals.get("tool_calls", 0)),
                1 if signals.get("canary_in_output") else 0,
                1 if signals.get("canary_in_files") else 0,
                1 if signals.get("agent_refused") else 0,
                signals.get("agent_error"),
                1 if verdict.get("llm_used") else 0,
                float(run.get("duration_ms", 0.0)),
                utcnow_iso(),
                json.dumps(attack, ensure_ascii=False, default=str),
                json.dumps(run, ensure_ascii=False, default=str),
                json.dumps(verdict, ensure_ascii=False, default=str),
            ),
        )
        self._conn.commit()

        if jsonl_path is not None:
            self._mirror_jsonl(jsonl_path, attack, run, verdict)

    @staticmethod
    def _mirror_jsonl(
        path: str | Path,
        attack: dict[str, Any],
        run: dict[str, Any],
        verdict: dict[str, Any],
    ) -> None:
        record = {
            "ts": utcnow_iso(),
            "run_id": verdict.get("run_id"),
            "attack_id": attack.get("id"),
            "category": attack.get("category"),
            "success": verdict.get("success"),
            "blocked_by_code": verdict.get("blocked_by_code"),
            "method": verdict.get("method"),
            "rationale": verdict.get("rationale"),
            "signals": verdict.get("signals"),
            "attack": attack,
            "run": run,
            "verdict": verdict,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    # -- reading ------------------------------------------------------------

    def latest_run_id(self) -> str | None:
        row = self._conn.execute(
            "SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return row["run_id"] if row else None

    def run_meta(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def list_runs(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def iter_results(self, run_id: str) -> Iterator[AttackResult]:
        rows = self._conn.execute(
            "SELECT * FROM attacks WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
        for row in rows:
            yield AttackResult(
                run_id=row["run_id"],
                attack_id=row["attack_id"],
                category=row["category"],
                success=bool(row["success"]),
                blocked_by_code=bool(row["blocked_by_code"]),
                judge_method=row["judge_method"],
                confidence=row["confidence"],
                rationale=row["rationale"],
                origin=row["origin"],
                escaped_calls=row["escaped_calls"],
                blocked_tool_calls=row["blocked_tool_calls"],
                outside_root_attempts=row["outside_root_attempts"],
                tool_calls=row["tool_calls"],
                canary_in_output=bool(row["canary_in_output"]),
                canary_in_files=bool(row["canary_in_files"]),
                agent_refused=bool(row["agent_refused"]),
                agent_error=row["agent_error"],
                duration_ms=row["duration_ms"],
                attack=json.loads(row["attack_json"]),
                run=json.loads(row["run_json"]),
                verdict=json.loads(row["verdict_json"]),
            )

    def results(self, run_id: str) -> list[AttackResult]:
        return list(self.iter_results(run_id))

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ResultStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
