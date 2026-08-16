"""Persistence round-trips, metric aggregation, and report rendering.

ResultStore commits after every write, and SQLite's default synchronous=FULL
turns each commit into an fsync — ~78ms per test against a real file. Tests that
are actually about metrics or rendering only need the store as scaffolding, so
they build it in memory: the same SQL, serialisation and query paths run, minus
the disk flush. Durability is a real guarantee, so the tests that genuinely
assert it — the persistence round-trip, and the CLI tests where report_main
opens the database file by path — keep using tmp_path.
"""

from __future__ import annotations

import csv
import io
import shutil

import pytest

from attacker.models import Attack
from judge.models import Signals, Verdict
from report.__main__ import main as report_main
from report.metrics import compute_metrics
from report.render import CSV_COLUMNS, render_csv, render_markdown, render_table
from runner.store import ResultStore

# SQLite's in-memory database, for tests that don't assert durability.
IN_MEMORY = ":memory:"


def _attack(attack_id="a1", category="prompt_injection"):
    return Attack(
        id=attack_id,
        category=category,
        name="test attack",
        prompt="do the thing",
        judge="code",
        success_signal={"type": "file_written", "path": "PWNED.txt"},
    )


def _verdict(
    attack,
    *,
    success,
    blocked_by_code=False,
    escaped=0,
    outside=0,
    method="code",
    agent_error=None,
):
    return Verdict(
        run_id="run-1",
        attack_id=attack.id,
        category=attack.category,
        success=success,
        blocked_by_code=blocked_by_code,
        method=method,
        signals=Signals(
            tool_calls=1,
            outside_root_attempts=outside,
            escaped_calls=escaped,
            blocked_tool_calls=1 if blocked_by_code else 0,
            agent_error=agent_error,
        ),
    )


def test_store_roundtrip(tmp_path):
    store = ResultStore(tmp_path / "results.sqlite3")
    store.record_run("run-1", {"target_model": "claude-opus-5", "categories": ["prompt_injection"]})

    attack = _attack()
    verdict = _verdict(attack, success=True)
    run = {"run_id": "run-1", "attack_id": attack.id, "duration_ms": 123.4, "tool_calls": []}
    store.record_attack(
        run_id="run-1", attack=attack.to_dict(), run=run, verdict=verdict.to_dict()
    )
    store.finish_run("run-1")

    assert store.latest_run_id() == "run-1"
    results = store.results("run-1")
    assert len(results) == 1
    assert results[0].success is True
    assert results[0].category == "prompt_injection"
    store.close()


def test_metrics_and_render():
    store = ResultStore(IN_MEMORY)
    store.record_run("run-1", {"target_model": "claude-opus-5"})

    # one injection win, one injection loss, one escalation blocked-by-code
    specs = [
        (_attack("pi-1", "prompt_injection"), _verdict(_attack("pi-1"), success=True)),
        (
            _attack("pi-2", "prompt_injection"),
            _verdict(_attack("pi-2", "prompt_injection"), success=False, blocked_by_code=True),
        ),
        (
            _attack("pe-1", "permission_escalation"),
            _verdict(
                _attack("pe-1", "permission_escalation"),
                success=False,
                blocked_by_code=True,
                outside=1,
            ),
        ),
    ]
    for attack, verdict in specs:
        store.record_attack(
            run_id="run-1",
            attack=attack.to_dict(),
            run={"run_id": "run-1", "attack_id": attack.id, "tool_calls": []},
            verdict=verdict.to_dict(),
        )

    meta = store.run_meta("run-1")
    results = store.results("run-1")
    metrics = compute_metrics("run-1", meta, results)

    pi = metrics.categories["prompt_injection"]
    assert pi.total == 2
    assert pi.succeeded == 1
    assert pi.success_rate == 0.5
    assert pi.blocked_by_code == 1

    pe = metrics.categories["permission_escalation"]
    assert pe.success_rate == 0.0
    assert pe.escaping_calls == 0

    assert metrics.containment_ok
    # renderers must not raise and must mention the run id
    assert "run-1" in render_table(metrics)
    assert "run-1" in render_markdown(metrics)
    store.close()


def test_errored_attacks_are_broken_out_and_leave_the_rate_denominator():
    store = ResultStore(IN_MEMORY)
    store.record_run("run-e", {})

    specs = [
        # one win, one genuine loss, one attack that never ran
        (_attack("pi-1", "prompt_injection"), _verdict(_attack("pi-1"), success=True)),
        (_attack("pi-2", "prompt_injection"), _verdict(_attack("pi-2"), success=False)),
        (
            _attack("pi-3", "prompt_injection"),
            _verdict(_attack("pi-3"), success=False, method="errored"),
        ),
        # a row written before the judge grew the "errored" verdict: only the
        # stored agent_error marks it, and it must still be excluded.
        (
            _attack("pi-4", "prompt_injection"),
            _verdict(
                _attack("pi-4"),
                success=False,
                method="llm",
                agent_error="container_error: docker exec timed out after 300s",
            ),
        ),
        # a category where every attack errored -> rate is undefined, not 0%
        (
            _attack("gh-1", "goal_hijacking"),
            _verdict(_attack("gh-1", "goal_hijacking"), success=False, method="errored"),
        ),
    ]
    for attack, verdict in specs:
        store.record_attack(
            run_id="run-e",
            attack=attack.to_dict(),
            run={"run_id": "run-e", "attack_id": attack.id, "tool_calls": []},
            verdict=verdict.to_dict(),
        )

    metrics = compute_metrics("run-e", store.run_meta("run-e"), store.results("run-e"))

    pi = metrics.categories["prompt_injection"]
    assert pi.total == 4
    assert pi.errored == 2
    assert pi.scored == 2
    assert pi.succeeded == 1
    assert pi.success_rate == 0.5  # 1/2 completed, NOT 1/4
    assert pi.rate_known
    assert pi.to_dict()["errored"] == 2

    gh = metrics.categories["goal_hijacking"]
    assert gh.total == 1
    assert gh.errored == 1
    assert not gh.rate_known
    assert gh.success_rate == 0.0  # never divides by zero
    assert gh.to_dict()["success_rate"] is None  # undefined, not a measured 0%

    # A method="errored" verdict never reached the LLM judge, so it is not
    # counted as one. The legacy row still counts: it really was sent to the
    # LLM judge — that is exactly the mis-scoring this fix prevents going forward.
    assert pi.llm_judged == 1

    assert metrics.errored == 3
    assert metrics.scored == 2
    assert metrics.overall_success_rate == 0.5

    table = render_table(metrics)
    assert "errored" in table
    assert "n/a" in table  # the all-errored category
    markdown = render_markdown(metrics)
    assert "Errored" in markdown
    assert "n/a" in markdown
    store.close()


def test_all_errored_run_renders_without_dividing_by_zero():
    store = ResultStore(IN_MEMORY)
    store.record_run("run-z", {})
    attack = _attack("pi-9", "prompt_injection")
    store.record_attack(
        run_id="run-z",
        attack=attack.to_dict(),
        run={"run_id": "run-z", "attack_id": attack.id, "tool_calls": []},
        verdict=_verdict(attack, success=False, method="errored").to_dict(),
    )
    metrics = compute_metrics("run-z", store.run_meta("run-z"), store.results("run-z"))
    assert not metrics.rate_known
    assert metrics.overall_success_rate == 0.0
    assert "n/a" in render_table(metrics)
    assert render_markdown(metrics)  # must not raise
    store.close()


# --- CSV export --------------------------------------------------------------
#
# NB: there is no test for embedded commas/quotes. Every CSV field is either an
# integer, a float, or a category name, and category names come from
# attacker.models.CATEGORIES — fixed identifiers like "prompt_injection" that
# cannot contain a comma or a quote. Rather than invent an unreachable case, the
# quoting guarantee is left to the stdlib csv module these renderers use.


def _csv_store(db_path, run_id="run-csv"):
    """A store with one win, one loss, and an all-errored second category.

    ``db_path`` is IN_MEMORY for render-only tests, or a real file for the CLI
    tests, which reopen the database by path.
    """
    store = ResultStore(db_path)
    store.record_run(run_id, {})
    specs = [
        (_attack("pi-1", "prompt_injection"), _verdict(_attack("pi-1"), success=True)),
        (_attack("pi-2", "prompt_injection"), _verdict(_attack("pi-2"), success=False)),
        (
            _attack("gh-1", "goal_hijacking"),
            _verdict(_attack("gh-1", "goal_hijacking"), success=False, method="errored"),
        ),
    ]
    for attack, verdict in specs:
        store.record_attack(
            run_id=run_id,
            attack=attack.to_dict(),
            run={"run_id": run_id, "attack_id": attack.id, "tool_calls": []},
            verdict=verdict.to_dict(),
        )
    return store


def _parse_csv(text):
    return list(csv.reader(io.StringIO(text)))


def test_render_csv_header_and_one_row_per_category_plus_overall():
    store = _csv_store(IN_MEMORY)
    metrics = compute_metrics("run-csv", store.run_meta("run-csv"), store.results("run-csv"))

    rows = _parse_csv(render_csv(metrics))

    assert rows[0] == list(CSV_COLUMNS)
    assert len(rows) == 1 + len(metrics.categories) + 1  # header + categories + OVERALL

    by_category = {row[0]: row for row in rows[1:]}
    assert set(by_category) == {"prompt_injection", "goal_hijacking", "OVERALL"}

    # category, attacks, wins, errored, scored, success_rate, canary_leaks
    assert by_category["prompt_injection"] == ["prompt_injection", "2", "1", "0", "2", "0.5", "0"]
    assert by_category["OVERALL"] == ["OVERALL", "3", "1", "1", "2", "0.5", "0"]
    store.close()


def test_render_csv_unknown_rate_is_an_empty_cell():
    store = _csv_store(IN_MEMORY)
    metrics = compute_metrics("run-csv", store.run_meta("run-csv"), store.results("run-csv"))

    rows = _parse_csv(render_csv(metrics))
    goal_hijacking = next(row for row in rows if row[0] == "goal_hijacking")

    rate = goal_hijacking[CSV_COLUMNS.index("success_rate")]
    assert rate == ""  # not "n/a" (breaks numeric columns), not "0.0" (a lie)
    assert goal_hijacking[CSV_COLUMNS.index("errored")] == "1"
    assert goal_hijacking[CSV_COLUMNS.index("scored")] == "0"
    store.close()


@pytest.fixture(scope="session")
def csv_db_template(tmp_path_factory):
    """The CLI store, built on disk once for the whole session.

    The three CLI tests below each need a real database file for report_main to
    open by path, and all three want identical contents. Building one costs
    ~76ms (a commit fsync per write); copying the finished file costs ~0.5ms.
    """
    path = tmp_path_factory.mktemp("csv-db") / "results.sqlite3"
    _csv_store(path).close()
    return path


def _install_db(template, tmp_path):
    """Give one test its own on-disk copy of the prebuilt database."""
    shutil.copyfile(template, tmp_path / "results.sqlite3")


def test_cli_format_csv_writes_report_csv(tmp_path, monkeypatch, capsys, csv_db_template):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    _install_db(csv_db_template, tmp_path)

    exit_code = report_main(["--run-id", "run-csv", "--format", "csv"])

    assert exit_code == 0
    out_path = tmp_path / "run-csv" / "report.csv"
    assert out_path.exists()

    rows = _parse_csv(out_path.read_text(encoding="utf-8"))
    assert rows[0] == list(CSV_COLUMNS)
    assert rows[-1][0] == "OVERALL"

    stdout = capsys.readouterr().out
    assert str(out_path) in stdout
    assert "Defence attribution" not in stdout  # the console table is suppressed
    assert not (tmp_path / "run-csv" / "report.md").exists()


def test_cli_format_csv_with_no_file_prints_to_stdout(
    tmp_path, monkeypatch, capsys, csv_db_template
):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    _install_db(csv_db_template, tmp_path)

    assert report_main(["--run-id", "run-csv", "--format", "csv", "--no-file"]) == 0

    rows = _parse_csv(capsys.readouterr().out)
    assert rows[0] == list(CSV_COLUMNS)
    assert not (tmp_path / "run-csv" / "report.csv").exists()


def test_cli_default_format_still_prints_the_table_and_writes_markdown(
    tmp_path, monkeypatch, capsys, csv_db_template
):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    _install_db(csv_db_template, tmp_path)

    assert report_main(["--run-id", "run-csv"]) == 0

    stdout = capsys.readouterr().out
    assert "Defence attribution" in stdout
    assert (tmp_path / "run-csv" / "report.md").exists()
    assert not (tmp_path / "run-csv" / "report.csv").exists()


def test_containment_failure_is_flagged():
    store = ResultStore(IN_MEMORY)
    store.record_run("run-x", {})
    attack = _attack("pe-2", "permission_escalation")
    verdict = _verdict(attack, success=True, escaped=1, outside=1)
    store.record_attack(
        run_id="run-x",
        attack=attack.to_dict(),
        run={"run_id": "run-x", "attack_id": attack.id, "tool_calls": []},
        verdict=verdict.to_dict(),
    )
    metrics = compute_metrics("run-x", store.run_meta("run-x"), store.results("run-x"))
    assert not metrics.containment_ok
    assert "CONTAINMENT FAILURE" in render_table(metrics)
    assert "Containment failure" in render_markdown(metrics)
    store.close()
