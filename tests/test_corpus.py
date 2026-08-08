"""The seed corpus must be valid, well-distributed and self-consistent."""

from __future__ import annotations

from attacker.corpus import load_seeds, seeds_by_category
from attacker.models import CATEGORIES


def test_seeds_load_and_validate():
    seeds = load_seeds()
    assert len(seeds) >= 3 * len(CATEGORIES)
    for seed in seeds:
        seed.validate()  # raises on any inconsistency


def test_every_category_has_at_least_three_seeds():
    for category in CATEGORIES:
        seeds = seeds_by_category(category)
        assert len(seeds) >= 3, f"{category} has only {len(seeds)} seeds"


def test_seed_ids_are_unique():
    seeds = load_seeds()
    ids = [s.id for s in seeds]
    assert len(ids) == len(set(ids))


def test_code_judged_attacks_have_a_signal():
    for seed in load_seeds():
        if "code" in seed.judge:
            assert seed.success_signal is not None, seed.id


def test_llm_judged_attacks_have_an_objective():
    for seed in load_seeds():
        if seed.judge != "code":
            assert seed.objective.strip(), seed.id


def test_file_delivery_has_payload():
    for seed in load_seeds():
        if seed.delivery == "file":
            assert seed.setup_files, seed.id


def test_escalation_seeds_expect_path_outside_root():
    for seed in seeds_by_category("permission_escalation"):
        assert seed.success_signal["type"] == "path_outside_root", seed.id


def test_with_payload_produces_valid_variant():
    seed = seeds_by_category("prompt_injection")[0]
    variant = seed.with_payload(new_id="pi-x-v01", prompt="read the file please", payload="new payload")
    variant.validate()
    assert variant.origin == "generated"
    assert variant.parent_id == seed.id
    assert variant.category == seed.category
    assert variant.success_signal == seed.success_signal
