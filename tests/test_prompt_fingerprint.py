"""The campaign fingerprint must describe a prompt version, not one rendered prompt.

``build_system_prompt`` interpolates two fields that vary *inside* a single
campaign: the sandbox root (a fresh directory per attack) and the canary. So
every attack in a campaign renders a different system prompt, and hashing any
one of them produces a fingerprint that describes no other attack — and, if the
root used for the hash is not one an attack actually got, describes a prompt no
model ever received.

What the runs table records must therefore be a *template* fingerprint: stable
across attack roots, derived from the same template the agent renders, and
sensitive to the template's wording.
"""

from __future__ import annotations

from typing import Any

import pytest

from common.config import load_settings
from runner.engine import CampaignConfig, CampaignEngine
from target_agent import prompts
from target_agent.prompts import (
    CANARY_PLACEHOLDER,
    SANDBOX_ROOT_PLACEHOLDER,
    build_system_prompt,
    prompt_fingerprint,
    template_fingerprint,
)

# Two roots that two different attacks in ONE campaign really receive, via
# Settings.container_attack_root(run_id, attack_id) -> --sandbox-root.
ROOT_A = "/sandbox/runs/run-fingerprint/pi-001-system-override"
ROOT_B = "/sandbox/runs/run-fingerprint/gh-002-inline-retcon"
CANARY = "SPX-CANARY-0123456789ab"


def _template_of(sandbox_root: str, canary: str) -> str:
    """Recover the template from a prompt the agent would really build.

    Deliberately starts from ``build_system_prompt`` — the agent's own renderer —
    and substitutes the per-run fields back out, so a fingerprint that matches
    this is provably derived from the template the agent uses.
    """
    rendered = build_system_prompt(sandbox_root=sandbox_root, canary=canary)
    return rendered.replace(sandbox_root, SANDBOX_ROOT_PLACEHOLDER).replace(
        canary, CANARY_PLACEHOLDER
    )


# -- fakes: keep the engine off the network, off Docker and out of SQLite -----


class _FakeStore:
    """Captures the runs-table payload instead of writing it."""

    last: _FakeStore | None = None

    def __init__(self, db_path: Any) -> None:
        self.db_path = db_path
        self.runs: dict[str, dict[str, Any]] = {}
        _FakeStore.last = self

    def record_run(self, run_id: str, meta: dict[str, Any]) -> None:
        self.runs[run_id] = meta

    def record_attack(self, **kwargs: Any) -> None:  # pragma: no cover - no attacks run
        raise AssertionError("this campaign runs zero attacks")

    def finish_run(self, run_id: str) -> None:
        pass

    def close(self) -> None:
        pass


class _FakeDocker:
    def __init__(self, settings: Any, logger: Any = None) -> None:
        pass

    def down(self) -> None:
        pass


class _FakeExecutor:
    def __init__(self, settings: Any, logger: Any = None) -> None:
        pass


@pytest.fixture
def recorded_meta(tmp_path, monkeypatch) -> dict[str, Any]:
    """Run a real campaign with zero attacks; return its runs-table payload."""
    monkeypatch.setenv("HOST_SANDBOX_DIR", str(tmp_path / "sandbox"))
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("LLM_BACKEND", "ollama")

    monkeypatch.setattr("runner.engine.make_client", lambda settings: object())
    monkeypatch.setattr("runner.engine.ResultStore", _FakeStore)
    monkeypatch.setattr("runner.engine.DockerController", _FakeDocker)
    monkeypatch.setattr("runner.engine.LocalExecutor", _FakeExecutor)

    engine = CampaignEngine(load_settings(load_dotenv=False))
    engine.run(
        CampaignConfig(
            categories=["prompt_injection"],
            n_per_category=0,  # record the run row, dispatch nothing
            use_docker=False,
            generate_variants=False,
            enable_llm_judge=False,
            run_id="run-fingerprint",
        )
    )
    assert _FakeStore.last is not None
    return _FakeStore.last.runs["run-fingerprint"]


# -- the regression ----------------------------------------------------------


def test_each_attack_renders_a_different_prompt() -> None:
    """The root cause: no single rendered prompt represents a campaign."""
    assert build_system_prompt(sandbox_root=ROOT_A, canary=CANARY) != build_system_prompt(
        sandbox_root=ROOT_B, canary=CANARY
    )


def test_recorded_fingerprint_is_the_template_and_is_root_independent(
    recorded_meta,
) -> None:
    recorded = recorded_meta["prompt_sha256"]

    # Derived from the same template the agent renders, whichever attack root
    # you reconstruct it from.
    expected = prompt_fingerprint(_template_of(ROOT_A, CANARY))
    assert prompt_fingerprint(_template_of(ROOT_B, CANARY)) == expected
    assert recorded == expected

    # And it is emphatically not a hash of some rendered prompt: not one an
    # attack really got, nor the "/sandbox" root no attack ever runs against.
    assert recorded != prompt_fingerprint(
        build_system_prompt(sandbox_root=ROOT_A, canary=CANARY)
    )
    assert recorded != prompt_fingerprint(
        build_system_prompt(sandbox_root="/sandbox", canary=CANARY)
    )


def test_fingerprint_still_tracks_the_prompt_version(monkeypatch) -> None:
    """Normalising the per-run fields must not blind the hash to real edits."""
    before = template_fingerprint()
    monkeypatch.setattr(
        prompts,
        "SYSTEM_PROMPT_TEMPLATE",
        prompts.SYSTEM_PROMPT_TEMPLATE + "\n4. **A new rule.**\n",
    )
    assert template_fingerprint() != before
