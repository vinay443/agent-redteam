"""CLI: run a red-team campaign.

    python -m runner --category prompt_injection --n 10
    python -m runner --category all --n 5 --no-docker
    python -m runner --category permission_escalation --n 8 --docker

Exit status is 0 on a completed campaign regardless of how many attacks
succeeded — a high success rate is a finding, not a failure of the tool. A
non-zero exit means the campaign itself could not run.
"""

from __future__ import annotations

import argparse
import sys

from attacker.models import CATEGORIES
from common.config import load_settings
from runner.engine import CampaignConfig, CampaignEngine, CampaignSummary

__all__ = ["main"]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m runner",
        description="Run an attack campaign against the sandboxed target agent.",
    )
    parser.add_argument(
        "--category",
        default="all",
        help="Attack category, or 'all'. Choices: " + ", ".join(CATEGORIES) + ", all.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Attacks per category (seed + generated variants).",
    )
    parser.add_argument(
        "--guard",
        default="none",
        help="Defence layer to apply to the target agent (default: none / undefended baseline).",
    )
    parser.add_argument(
        "--no-variants",
        action="store_true",
        help="Do not call the attacker model; repeat seeds to fill --n instead.",
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Disable the LLM judge; LLM-only attacks are scored as undecided (failure).",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Exclude the original hand-written seed; run generated variants only.",
    )
    docker_group = parser.add_mutually_exclusive_group()
    docker_group.add_argument(
        "--docker",
        dest="use_docker",
        action="store_true",
        default=None,
        help="Require containerised execution; fail if Docker is unavailable.",
    )
    docker_group.add_argument(
        "--no-docker",
        dest="use_docker",
        action="store_false",
        help="Force the in-process executor (NOT sandboxed; for development).",
    )
    parser.add_argument("--run-id", default=None, help="Override the generated run id.")
    return parser.parse_args(argv)


def _resolve_categories(choice: str) -> list[str]:
    if choice == "all":
        return list(CATEGORIES)
    if choice not in CATEGORIES:
        raise SystemExit(
            f"unknown category {choice!r}. Choices: {', '.join(CATEGORIES)}, all."
        )
    return [choice]


def _print_summary(summary: CampaignSummary) -> None:
    print(f"\nCampaign {summary.run_id} complete ({summary.executor}).")
    if not summary.sandboxed:
        print(
            "  WARNING: run was NOT containerised (in-process fallback). "
            "Results are for development only.",
            file=sys.stderr,
        )
    print(
        f"  attacks: {summary.total}   attacker wins: {summary.succeeded}   "
        f"errored: {summary.errored}"
    )
    for category, stats in summary.per_category.items():
        total = stats["total"] or 1
        rate = 100.0 * stats["succeeded"] / total
        print(
            f"    {category:<28} {stats['succeeded']:>3}/{stats['total']:<3} "
            f"({rate:5.1f}% success)  blocked_by_code={stats['blocked_by_code']}"
        )
    print(f"\nGenerate the report with:\n    python -m report --run-id {summary.run_id}")


def _force_utf8_stdout() -> None:
    """Windows consoles default to cp1252; campaign output may contain non-ASCII."""
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

    try:
        settings.require_api_key()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    config = CampaignConfig(
        categories=_resolve_categories(args.category),
        n_per_category=args.n,
        guard=args.guard,
        include_seed=not args.no_seed,
        generate_variants=not args.no_variants,
        use_docker=args.use_docker,
        enable_llm_judge=not args.no_llm_judge,
        run_id=args.run_id,
    )

    engine = CampaignEngine(settings)
    try:
        summary = engine.run(config)
    except Exception as exc:  # noqa: BLE001 - surface a clean message, not a traceback
        print(f"campaign failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
