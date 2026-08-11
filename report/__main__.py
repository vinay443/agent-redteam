"""CLI: turn stored results into a table, a Markdown report, or a CSV export.

    python -m report                       # latest run, table to stdout + report file
    python -m report --run-id run-...       # a specific run
    python -m report --list                 # list available runs
    python -m report --json                 # emit the metrics as JSON
    python -m report --format csv           # write results/<run_id>/report.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common.config import load_settings
from report.metrics import compute_metrics
from report.render import render_csv, render_markdown, render_table
from runner.store import ResultStore

__all__ = ["main"]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m report",
        description="Summarise a red-team campaign's results.",
    )
    parser.add_argument("--run-id", default=None, help="Run to report on (default: latest).")
    parser.add_argument("--list", action="store_true", help="List available runs and exit.")
    parser.add_argument("--json", action="store_true", help="Print metrics as JSON to stdout.")
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("text", "markdown", "csv"),
        default="text",
        help=(
            "text (default): console table; markdown: the Markdown report to stdout; "
            "csv: write results/<run_id>/report.csv instead of printing the table."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output file path (default: results/<run_id>/report.md, "
            "or report.csv with --format csv)."
        ),
    )
    parser.add_argument(
        "--no-file",
        action="store_true",
        help="Do not write a report file (with --format csv, print the CSV to stdout).",
    )
    return parser.parse_args(argv)


def _list_runs(store: ResultStore) -> int:
    runs = store.list_runs()
    if not runs:
        print("no runs found. Run `python -m runner --category all --n 5` first.")
        return 0
    print(f"{'run_id':<40}{'started':<26}{'target model':<20}categories")
    print("-" * 100)
    for run in runs:
        cats = run.get("categories", "[]")
        try:
            cats = ", ".join(json.loads(cats))
        except (json.JSONDecodeError, TypeError):
            pass
        print(
            f"{run['run_id']:<40}{run.get('started_at', ''):<26}"
            f"{run.get('target_model', ''):<20}{cats}"
        )
    return 0


def _force_utf8_stdout() -> None:
    """Windows consoles default to cp1252; the report may contain non-ASCII."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    args = _parse_args(argv)
    settings = load_settings()

    if not settings.db_path.exists():
        print(
            f"no results database at {settings.db_path}. Run a campaign first:\n"
            "    python -m runner --category all --n 5",
            file=sys.stderr,
        )
        return 2

    store = ResultStore(settings.db_path)
    try:
        if args.list:
            return _list_runs(store)

        run_id = args.run_id or store.latest_run_id()
        if run_id is None:
            print("no runs in the database yet.", file=sys.stderr)
            return 2

        meta = store.run_meta(run_id)
        if meta is None:
            print(f"no such run: {run_id}", file=sys.stderr)
            return 2

        # Decode JSON columns for the renderer.
        for key in ("categories", "settings"):
            value = meta.get(key)
            if isinstance(value, str):
                try:
                    meta[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass

        results = store.results(run_id)
        metrics = compute_metrics(run_id, meta, results)

        if args.json:
            print(json.dumps(metrics.to_dict(), indent=2))
            return 0

        if args.output_format == "csv":
            # A CSV run is an export, not a read: the console table would just
            # bury the one line that says where the file went.
            csv_text = render_csv(metrics)
            if args.no_file:
                print(csv_text, end="")
            else:
                path = Path(args.out or settings.run_dir(run_id) / "report.csv")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(csv_text, encoding="utf-8")
                print(f"CSV report written to {path}")
        else:
            print(
                render_markdown(metrics)
                if args.output_format == "markdown"
                else render_table(metrics)
            )

            if not args.no_file:
                path = Path(args.out or settings.run_dir(run_id) / "report.md")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(render_markdown(metrics), encoding="utf-8")
                print(f"\nMarkdown report written to {path}")

        if not metrics.containment_ok:
            return 3  # signal a containment failure to CI
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
