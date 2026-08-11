"""CLI-level prechecks and console output in ``runner.__main__``.

The prechecks assert the *fail-fast* contract, which is distinct from the
client-layer behaviour covered in ``tests/test_llm_client.py``: an Anthropic
campaign with no key must be refused before anything is built, started or
dispatched — not surfaced as a 401 partway through, after the image is built
and N attacks have already been provisioned.

The summary tests assert that the console's success rate uses the same
denominator as the generated report — attacks that actually ran — so the two
never disagree about the same run.
"""

from __future__ import annotations

import pytest

import common.config
from runner import __main__ as cli
from runner.engine import CampaignEngine, CampaignSummary


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    """The CLI loads <repo>/.env; a developer's real key must not leak in here."""
    monkeypatch.setattr(common.config, "_load_dotenv_if_present", lambda: None)


@pytest.fixture
def campaign_never_runs(monkeypatch):
    """Trip if the campaign starts; a precheck that runs too late is a failure."""

    def _boom(self, config):  # noqa: ARG001
        raise AssertionError("campaign started despite a failed precheck")

    monkeypatch.setattr(CampaignEngine, "run", _boom)


def test_anthropic_without_key_fails_fast(monkeypatch, capsys, campaign_never_runs):
    monkeypatch.setenv("LLM_BACKEND", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    exit_code = cli.main(["--category", "prompt_injection", "--n", "5", "--docker"])

    assert exit_code == 2
    stderr = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in stderr  # names the variable
    assert ".env" in stderr  # and how to fix it


def test_anthropic_with_empty_key_also_fails_fast(monkeypatch, campaign_never_runs):
    # An exported-but-empty key is the common CI shape and must not pass.
    monkeypatch.setenv("LLM_BACKEND", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    assert cli.main(["--category", "prompt_injection", "--n", "1"]) == 2


def test_ollama_backend_needs_no_key(monkeypatch):
    # The default backend is keyless: the precheck must not block it.
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    seen: dict[str, object] = {}

    def _fake_run(self, config):  # noqa: ARG001
        seen["config"] = config
        return CampaignSummary(run_id="run-test", executor="docker", sandboxed=True)

    monkeypatch.setattr(CampaignEngine, "run", _fake_run)

    assert cli.main(["--category", "prompt_injection", "--n", "5", "--docker"]) == 0
    assert seen["config"].use_docker is True  # --docker survived the precheck


# --- console summary ---------------------------------------------------------


def _summary(**per_category):
    """A CampaignSummary whose totals are the sum of the given categories."""
    stats = {
        name: {
            "total": total,
            "succeeded": succeeded,
            "blocked_by_code": blocked,
            "errored": errored,
        }
        for name, (total, succeeded, errored, blocked) in per_category.items()
    }
    return CampaignSummary(
        run_id="run-test",
        executor="docker",
        sandboxed=True,
        total=sum(s["total"] for s in stats.values()),
        succeeded=sum(s["succeeded"] for s in stats.values()),
        errored=sum(s["errored"] for s in stats.values()),
        per_category=stats,
    )


def test_console_rate_excludes_errored_from_the_denominator(capsys):
    # 20 attacks, 4 wins, 3 never ran. The rate is over the 17 that did:
    # 4/17 = 23.5%, not the raw 4/20 = 20.0%. The report says 23.5% for this
    # same run, and a console that disagrees with it is a reporting bug.
    cli._print_summary(_summary(prompt_injection=(20, 4, 3, 0)))

    out = capsys.readouterr().out
    assert "23.5%" in out
    assert "20.0%" not in out


def test_console_rate_matches_report_across_a_mixed_campaign(capsys):
    cli._print_summary(
        _summary(
            prompt_injection=(20, 4, 3, 0),          # 4/17  = 23.5%
            goal_hijacking=(20, 15, 1, 0),           # 15/19 = 78.9%
            permission_escalation=(20, 0, 4, 7),     # 0/16  =  0.0%
            system_prompt_exfiltration=(20, 4, 5, 0),  # 4/15 = 26.7%
        )
    )

    out = capsys.readouterr().out
    for rate in ("23.5%", "78.9%", "0.0%", "26.7%"):
        assert rate in out, f"missing {rate}"
    # The raw wins/total rates for these categories must not appear.
    for raw in ("75.0%", "20.0%"):
        assert raw not in out, f"raw rate {raw} leaked into the summary"
    # Headline counts are untouched by the denominator change.
    assert "attacks: 80" in out
    assert "attacker wins: 23" in out
    assert "errored: 13" in out


def test_console_rate_is_na_when_every_attack_errored(capsys):
    # Nothing ran, so there is no rate. 0.0% would read as "the model defended
    # all of them" — the exact conflation the errored bucket exists to prevent.
    cli._print_summary(_summary(permission_escalation=(6, 0, 6, 0)))

    out = capsys.readouterr().out
    assert "n/a" in out
    assert "%" not in out  # no percentage at all, not even 0.0%
