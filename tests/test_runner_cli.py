"""CLI-level prechecks in ``runner.__main__``.

These assert the *fail-fast* contract, which is distinct from the client-layer
behaviour covered in ``tests/test_llm_client.py``: an Anthropic campaign with no
key must be refused before anything is built, started or dispatched — not
surfaced as a 401 partway through, after the image is built and N attacks have
already been provisioned.
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
