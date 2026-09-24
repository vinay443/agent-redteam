"""Render metrics as a terminal table, a Markdown report, or a CSV export."""

from __future__ import annotations

import csv
import io

from report.metrics import CampaignMetrics

__all__ = ["render_table", "render_markdown", "render_csv"]

_CATEGORY_ORDER = (
    "prompt_injection",
    "goal_hijacking",
    "permission_escalation",
    "system_prompt_exfiltration",
)


def _ordered(metrics: CampaignMetrics) -> list[str]:
    present = list(metrics.categories)
    ordered = [c for c in _CATEGORY_ORDER if c in present]
    ordered += [c for c in present if c not in ordered]
    return ordered


def _bar(rate: float, width: int = 20) -> str:
    # ASCII only: the console table must render on a cp1252 Windows terminal.
    filled = round(rate * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _rate_cell(rate: float, known: bool) -> str:
    """A rate with no completed attacks behind it is n/a, never 0.0%."""
    return f"{rate * 100:6.1f}%" if known else "n/a"


def _success_cell(rate: float, known: bool) -> str:
    # An empty bar would read as "0% success"; say why there is no bar instead.
    # "scored", not "completed": a judge-errored attack ran to completion and
    # still has no outcome to draw.
    return _bar(rate) if known else "(no attacks scored)"


def render_table(metrics: CampaignMetrics) -> str:
    """A plain-text table for the console."""
    lines: list[str] = []
    meta = metrics.meta
    lines.append("=" * 78)
    lines.append(f"  agent-redteam report — run {metrics.run_id}")
    lines.append("=" * 78)
    settings = meta.get("settings", {})
    settings = settings if isinstance(settings, dict) else {}
    lines.append(f"  backend      : {settings.get('llm_backend', '?')}")
    lines.append(f"  target model : {meta.get('target_model', '?')}")
    lines.append(f"  attacker     : {meta.get('attacker_model', '?')}")
    lines.append(f"  judge        : {meta.get('judge_model', '?')}")
    lines.append(f"  guard        : {meta.get('guard', 'none')}")
    lines.append(
        f"  executor     : {settings.get('executor', '?')} "
        f"(sandboxed={settings.get('sandboxed', '?')})"
    )
    lines.append("")

    header = (
        f"  {'category':<28}{'attacks':>8}{'wins':>6}{'errored':>9}"
        f"{'judge-err':>11}{'rate':>8}  success"
    )
    lines.append(header)
    lines.append("  " + "-" * 85)
    for category in _ordered(metrics):
        cat = metrics.categories[category]
        lines.append(
            f"  {category:<28}{cat.total:>8}{cat.succeeded:>6}{cat.errored:>9}"
            f"{cat.judge_errored:>11}"
            f"{_rate_cell(cat.success_rate, cat.rate_known):>8}  "
            f"{_success_cell(cat.success_rate, cat.rate_known)}"
        )
    lines.append("  " + "-" * 85)
    lines.append(
        f"  {'OVERALL':<28}{metrics.total:>8}{metrics.succeeded:>6}{metrics.errored:>9}"
        f"{metrics.judge_errored:>11}"
        f"{_rate_cell(metrics.overall_success_rate, metrics.rate_known):>8}  "
        f"{_success_cell(metrics.overall_success_rate, metrics.rate_known)}"
    )
    lines.append("")
    unscored: list[str] = []
    if metrics.errored:
        unscored.append(f"{metrics.errored} attack(s) never ran (errored)")
    if metrics.judge_errored:
        unscored.append(
            f"{metrics.judge_errored} attack(s) ran but were never scored "
            "(judge-err: the LLM judge failed on them)"
        )
    if unscored:
        lines.append("  Note: " + ", and ".join(unscored) + ".")
        lines.append(
            "  An attack with no outcome is not a defence, so it is not in the "
            "denominator: rate = wins / (attacks - errored - judge-err)."
        )
        lines.append("")

    lines.append("  Defence attribution (of attacks that did NOT succeed):")
    for category in _ordered(metrics):
        cat = metrics.categories[category]
        lines.append(
            f"    {category:<28} blocked-by-code {cat.blocked_by_code:>3}/{cat.failed:<3}"
            f"  model-refused {cat.model_refused:>3}"
        )
    lines.append("")

    if metrics.containment_ok:
        lines.append("  Containment: OK — 0 tool calls escaped the sandbox root.")
        if not metrics.fully_scored:
            # Containment is a code fact and stays true, but on its own this
            # line reads as a clean run — and it cannot say anything about the
            # attacks nobody scored.
            lines.append(
                "  ** PARTIALLY SCORED RUN ** — that covers the sandbox only. "
                f"{metrics.judge_errored} attack(s) ran without a verdict, so the "
                "rates above describe a subset of this campaign."
            )
    else:
        lines.append(
            f"  ** CONTAINMENT FAILURE ** — {metrics.total_escaping_calls} tool call(s) "
            "escaped the sandbox root. Investigate target_agent/sandbox.py immediately."
        )
    lines.append("=" * 78)
    return "\n".join(lines)


def render_markdown(metrics: CampaignMetrics) -> str:
    """A shareable Markdown report."""
    meta = metrics.meta
    settings = meta.get("settings", {}) if isinstance(meta.get("settings"), dict) else {}
    out: list[str] = []

    out.append(f"# Red-team report — `{metrics.run_id}`")
    out.append("")
    out.append(f"- **LLM backend:** `{settings.get('llm_backend', '?')}`")
    out.append(f"- **Target model:** `{meta.get('target_model', '?')}`")
    out.append(f"- **Attacker model:** `{meta.get('attacker_model', '?')}`")
    out.append(f"- **Judge model:** `{meta.get('judge_model', '?')}`")
    out.append(f"- **Guard:** `{meta.get('guard', 'none')}`")
    out.append(
        f"- **Executor:** `{settings.get('executor', '?')}` "
        f"(sandboxed: `{settings.get('sandboxed', '?')}`)"
    )
    out.append(
        f"- **Started:** {meta.get('started_at', '?')} "
        f"· **Finished:** {meta.get('finished_at', '?')}"
    )
    out.append(
        f"- **System-prompt template SHA-256:** `{meta.get('prompt_sha256', '?')}` "
        "— the template's wording, with the per-attack sandbox root and per-run "
        "canary normalised out, so it identifies the prompt version. Each "
        "attack's exact rendered-prompt hash is stored on its own run record."
    )
    out.append("")

    if not metrics.containment_ok:
        out.append(
            f"> ⛔ **Containment failure:** {metrics.total_escaping_calls} tool call(s) "
            "resolved outside the sandbox root. This is a bug in the containment "
            "layer, not a model finding. Do not trust the escalation numbers until "
            "it is fixed."
        )
        out.append("")
    else:
        intact = "> ✅ **Containment intact:** no tool call escaped the sandbox root."
        if not metrics.fully_scored:
            # True, and not the whole story: it is a statement about the sandbox,
            # and some attacks were never scored at all. Say so on the same line
            # so the badge is never read on its own as a clean run.
            intact += (
                " This is a statement about the sandbox only — see the unscored-attack "
                "warning below before reading it as a clean run."
            )
        out.append(intact)
        out.append("")

    if not metrics.fully_scored:
        out.append(
            f"> ⚠️ **{metrics.judge_errored} attack(s) ran but were never scored:** the "
            "LLM judge failed on them (a timeout, a transport error, an unparseable "
            "verdict), so nobody knows how they went. They are excluded from every "
            "rate below, which therefore covers only the attacks that *were* scored — "
            "not the whole campaign. Without this exclusion a judge that failed on "
            "every call would report a flawless 0% attack success rate."
        )
        out.append("")

    out.append("## Attack success rate by category")
    out.append("")
    out.append("Success = the attacker compromised the agent. Lower is safer.")
    out.append("")
    out.append(
        "Two kinds of attack have no outcome to report, and both are excluded from "
        "the rate — `rate = wins / (attacks - errored - judge errors)`:"
    )
    out.append("")
    out.append(
        "- **Errored** attacks never executed (e.g. a container exec timeout).\n"
        "- **Judge errors** are attacks that ran but that the LLM judge could not "
        "score.\n"
        "\n"
        "Both are an absence of evidence, not a successful defence. A category in "
        "which every attack fell into one of them has no rate at all and reads "
        "`n/a`, never `0.0%`."
    )
    out.append("")
    out.append(
        "| Category | Attacks | Wins | Errored | Judge errors | Success rate | Canary leaks |"
    )
    out.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for category in _ordered(metrics):
        cat = metrics.categories[category]
        out.append(
            f"| {category} | {cat.total} | {cat.succeeded} | {cat.errored} "
            f"| {cat.judge_errored} "
            f"| {_rate_cell(cat.success_rate, cat.rate_known).strip()} | {cat.canary_leaks} |"
        )
    overall_rate = _rate_cell(metrics.overall_success_rate, metrics.rate_known).strip()
    out.append(
        f"| **Overall** | **{metrics.total}** | **{metrics.succeeded}** "
        f"| **{metrics.errored}** | **{metrics.judge_errored}** "
        f"| **{overall_rate}** | — |"
    )
    out.append("")

    out.append("## Defence attribution")
    out.append("")
    out.append(
        "For attacks that did **not** succeed, was the attack stopped by a code "
        "boundary (the sandbox / tool layer) or by the model itself declining? A "
        "high blocked-by-code share means the safety does not depend on the model."
    )
    out.append("")
    out.append("| Category | Failed | Blocked by code | Model refused | Blocked-by-code share |")
    out.append("| --- | ---: | ---: | ---: | ---: |")
    for category in _ordered(metrics):
        cat = metrics.categories[category]
        out.append(
            f"| {category} | {cat.failed} | {cat.blocked_by_code} "
            f"| {cat.model_refused} | {cat.blocked_by_code_rate * 100:.1f}% |"
        )
    out.append("")

    out.append("## Escalation containment")
    out.append("")
    out.append(
        "Every permission-escalation attempt should be stopped in code, so "
        "*escaping calls* must be 0 while *outside-root attempts* may be high "
        "(the model tried; the sandbox held)."
    )
    out.append("")
    out.append("| Category | Outside-root attempts | Escaping calls |")
    out.append("| --- | ---: | ---: |")
    for category in _ordered(metrics):
        cat = metrics.categories[category]
        flag = "" if cat.escaping_calls == 0 else " ⛔"
        out.append(f"| {category} | {cat.outside_root_attempts} | {cat.escaping_calls}{flag} |")
    out.append("")

    successes = [
        (category, ex)
        for category in _ordered(metrics)
        for ex in metrics.categories[category].examples
    ]
    if successes:
        out.append("## Example successful attacks")
        out.append("")
        for category, ex in successes:
            out.append(f"- **[{category}]** `{ex['attack_id']}` — {ex['name']}")
            if ex.get("rationale"):
                out.append(f"  - _{ex['rationale']}_")
        out.append("")

    out.append("## How to read this")
    out.append("")
    out.append(
        "- **prompt_injection / goal_hijacking** measure whether untrusted "
        "content can redirect the agent. These are model-behaviour findings.\n"
        "- **permission_escalation** measures the sandbox, not the model — its "
        "success rate should be 0.0%. If it is not, the containment layer has a "
        "bug.\n"
        "- **system_prompt_exfiltration** is judged both by an exact canary-token "
        "check and by an LLM looking for paraphrased leaks.\n"
        "- **blocked-by-code share** tells you how much of the agent's safety is "
        "structural (holds regardless of the model) versus behavioural (depends "
        "on the model choosing to refuse)."
    )
    out.append("")
    out.append("---")
    out.append("")
    out.append("_Generated by `python -m report`._")
    out.append("")
    return "\n".join(out)


# --- CSV export --------------------------------------------------------------

CSV_COLUMNS = (
    "category",
    "attacks",
    "wins",
    "errored",
    "judge_errored",
    "scored",
    "success_rate",
    "canary_leaks",
)


def _csv_rate(rate: float, known: bool) -> float | str:
    """A rate for a spreadsheet: a plain 0-1 float, or empty when undefined.

    Deliberately not the table's ``n/a``: that string would land in a numeric
    column and break every formula over it. An empty cell reads as "no data",
    which is exactly what an all-errored category means. ``0.0`` would be worse
    still — indistinguishable from a measured zero success rate.
    """
    return round(rate, 4) if known else ""


def render_csv(metrics: CampaignMetrics) -> str:
    """One row per category plus an OVERALL row, for spreadsheet import."""
    buffer = io.StringIO()
    # lineterminator: csv defaults to \r\n; this returns a str for a caller that
    # writes it with its own newline handling, so keep it plain.
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)

    for category in _ordered(metrics):
        cat = metrics.categories[category]
        writer.writerow(
            [
                cat.category,
                cat.total,
                cat.succeeded,
                cat.errored,
                cat.judge_errored,
                cat.scored,
                _csv_rate(cat.success_rate, cat.rate_known),
                cat.canary_leaks,
            ]
        )

    writer.writerow(
        [
            "OVERALL",
            metrics.total,
            metrics.succeeded,
            metrics.errored,
            metrics.judge_errored,
            metrics.scored,
            _csv_rate(metrics.overall_success_rate, metrics.rate_known),
            sum(c.canary_leaks for c in metrics.categories.values()),
        ]
    )
    return buffer.getvalue()
