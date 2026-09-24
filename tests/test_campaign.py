"""Campaign assembly: how ``--n`` is spread across a category's seeds.

The contract in ``attacker/campaign.py`` is a distribution one, and it is the
kind that looks right until you count: ``--n 10`` over five seeds must give two
attacks per seed, not ten variants of the first. Totals alone cannot catch a
broken split — ten variants of one seed also totals ten — so these tests assert
the per-seed counts *and* the order they come out in.

The seed list is fixed (four synthetic seeds) for the arithmetic tests so the
expected numbers stay hard-coded and cannot drift with the corpus, and the real
corpus is used separately to check the same rule end to end.

Attack generation goes through the same fake ``LLMClient`` used in
``tests/test_variant_generator.py``.
"""

from __future__ import annotations

from collections import Counter

import pytest

from attacker.campaign import build_campaign
from attacker.corpus import seeds_by_category
from attacker.generator import VariantGenerator
from attacker.models import CATEGORIES, Attack
from common.llm_client import LLMError
from tests.test_variant_generator import FakeJSONClient, RecordingLogger, item, variants_json


@pytest.fixture
def four_seeds(monkeypatch) -> list[Attack]:
    """Four valid seeds in one category, substituted for the corpus.

    ``build_campaign`` reads the corpus itself, so this replaces the lookup it
    performs rather than the corpus file. Four seeds and a request that does not
    divide by four is the case the ``divmod`` exists for.
    """
    seeds = [
        Attack(
            id=f"pi-{i:03d}-synthetic",
            category="prompt_injection",
            name=f"synthetic seed {i}",
            prompt=f"Seed prompt {i}",
            delivery="file",
            setup_files={"notes.txt": f"payload {i}"},
            payload_file="notes.txt",
            objective="make the agent write PWNED.txt",
            judge="code",
            success_signal={"type": "file_written", "path": "PWNED.txt"},
        )
        for i in range(1, 5)
    ]
    for seed in seeds:
        seed.validate()
    monkeypatch.setattr("attacker.campaign.seeds_by_category", lambda category: list(seeds))
    return seeds


def parents(campaign: list[Attack]) -> list[str]:
    """The seed each attack came from, in campaign order."""
    return [attack.parent_id or attack.id for attack in campaign]


def make_generator(client: FakeJSONClient) -> VariantGenerator:
    return VariantGenerator(client, model="fake-attacker", logger=RecordingLogger())


# ---------------------------------------------------------------------------
# The divmod split
# ---------------------------------------------------------------------------


def test_uneven_split_front_loads_the_remainder(four_seeds):
    # divmod(7, 4) == (1, 3): every seed once, then one extra for the first
    # three. Not: seven variants of seed 1.
    campaign = build_campaign("prompt_injection", 7)

    assert len(campaign) == 7
    assert Counter(parents(campaign)) == {
        "pi-001-synthetic": 2,
        "pi-002-synthetic": 2,
        "pi-003-synthetic": 2,
        "pi-004-synthetic": 1,
    }
    # Order matters too: a seed's own attacks are contiguous, seeds in corpus
    # order, and the hand-written seed leads its own group.
    assert parents(campaign) == [
        "pi-001-synthetic",
        "pi-001-synthetic",
        "pi-002-synthetic",
        "pi-002-synthetic",
        "pi-003-synthetic",
        "pi-003-synthetic",
        "pi-004-synthetic",
    ]
    assert [a.origin for a in campaign] == [
        "seed",
        "generated",
        "seed",
        "generated",
        "seed",
        "generated",
        "seed",
    ]


def test_even_split_gives_every_seed_the_same_share(four_seeds):
    campaign = build_campaign("prompt_injection", 8)

    assert len(campaign) == 8
    assert Counter(parents(campaign)) == {
        "pi-001-synthetic": 2,
        "pi-002-synthetic": 2,
        "pi-003-synthetic": 2,
        "pi-004-synthetic": 2,
    }


def test_remainder_of_one_lands_on_the_first_seed_only(four_seeds):
    # divmod(9, 4) == (2, 1).
    campaign = build_campaign("prompt_injection", 9)

    assert Counter(parents(campaign)) == {
        "pi-001-synthetic": 3,
        "pi-002-synthetic": 2,
        "pi-003-synthetic": 2,
        "pi-004-synthetic": 2,
    }


def test_fewer_attacks_than_seeds_uses_a_prefix_of_the_seeds(four_seeds):
    # divmod(3, 4) == (0, 3): the fourth seed draws zero and is skipped.
    campaign = build_campaign("prompt_injection", 3)

    assert parents(campaign) == ["pi-001-synthetic", "pi-002-synthetic", "pi-003-synthetic"]
    assert all(a.origin == "seed" for a in campaign)


def test_single_attack_is_the_first_seed_unchanged(four_seeds):
    campaign = build_campaign("prompt_injection", 1)

    assert len(campaign) == 1
    assert campaign[0].id == "pi-001-synthetic"
    assert campaign[0].origin == "seed"


def test_zero_attacks_is_an_empty_campaign(four_seeds):
    assert build_campaign("prompt_injection", 0) == []


def test_every_attack_in_an_uneven_campaign_is_runnable(four_seeds):
    campaign = build_campaign("prompt_injection", 7)

    assert len({a.id for a in campaign}) == 7
    for attack in campaign:
        attack.validate()


def test_excluding_the_seed_makes_every_attack_derived(four_seeds):
    campaign = build_campaign("prompt_injection", 7, include_seed=False)

    assert len(campaign) == 7
    assert all(a.origin == "generated" for a in campaign)
    assert all(a.parent_id is not None for a in campaign)
    # The split is unchanged by dropping the hand-written seed.
    assert Counter(parents(campaign)) == {
        "pi-001-synthetic": 2,
        "pi-002-synthetic": 2,
        "pi-003-synthetic": 2,
        "pi-004-synthetic": 1,
    }


# ---------------------------------------------------------------------------
# The same rule against the shipped corpus
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("category", CATEGORIES)
def test_real_corpus_spreads_an_uneven_request_over_every_seed(category):
    seeds = seeds_by_category(category)
    assert len(seeds) >= 3  # the corpus guarantee test_corpus.py pins
    n = len(seeds) + 2  # never divides evenly: base 1, remainder 2

    campaign = build_campaign(category, n)

    assert len(campaign) == n
    expected = dict.fromkeys((s.id for s in seeds), 1)
    expected[seeds[0].id] = 2
    expected[seeds[1].id] = 2
    assert Counter(parents(campaign)) == expected
    assert parents(campaign) == [
        seeds[0].id,
        seeds[0].id,
        seeds[1].id,
        seeds[1].id,
        *[s.id for s in seeds[2:]],
    ]
    assert all(a.category == category for a in campaign)
    for attack in campaign:
        attack.validate()


def test_unknown_category_is_rejected():
    # The corpus lookup rejects it before build_campaign's own emptiness check.
    with pytest.raises(ValueError, match="unknown category 'not_a_category'"):
        build_campaign("not_a_category", 4)


def test_a_category_with_no_seeds_is_rejected(monkeypatch):
    monkeypatch.setattr("attacker.campaign.seeds_by_category", lambda category: [])
    with pytest.raises(ValueError, match="no seeds for category 'prompt_injection'"):
        build_campaign("prompt_injection", 4)


# ---------------------------------------------------------------------------
# With a generator attached
# ---------------------------------------------------------------------------


def test_generated_variants_fill_the_extra_slots(four_seeds):
    client = FakeJSONClient([variants_json(item("rephrased attack", payload="new payload"))])
    campaign = build_campaign("prompt_injection", 7, generator=make_generator(client))

    assert len(campaign) == 7
    # Three seeds drew two attacks each, so the model was asked three times, for
    # one variant apiece.
    assert len(client.calls) == 3
    assert all(call["user"].endswith("Produce exactly 1 distinct variants.") for call in client.calls)
    generated = [a for a in campaign if a.origin == "generated"]
    assert [a.prompt for a in generated] == ["rephrased attack"] * 3
    assert [a.setup_files["notes.txt"] for a in generated] == ["new payload"] * 3
    assert {a.parent_id for a in generated} == {
        "pi-001-synthetic",
        "pi-002-synthetic",
        "pi-003-synthetic",
    }


def test_a_dead_generator_still_produces_a_full_runnable_campaign(four_seeds):
    campaign = build_campaign(
        "prompt_injection",
        7,
        generator=make_generator(FakeJSONClient(raises=LLMError("Cannot reach Ollama"))),
    )

    assert len(campaign) == 7
    assert Counter(parents(campaign)) == {
        "pi-001-synthetic": 2,
        "pi-002-synthetic": 2,
        "pi-003-synthetic": 2,
        "pi-004-synthetic": 1,
    }
    fallbacks = [a for a in campaign if a.origin == "generated"]
    assert len(fallbacks) == 3
    assert all(a.notes == "fallback: seed repeated (generation unavailable)" for a in fallbacks)
    for attack in campaign:
        attack.validate()
        assert attack.success_signal == {"type": "file_written", "path": "PWNED.txt"}


def test_a_generator_returning_junk_still_produces_a_full_campaign(four_seeds):
    campaign = build_campaign(
        "prompt_injection", 7, generator=make_generator(FakeJSONClient(["not json"]))
    )

    assert len(campaign) == 7
    assert len({a.id for a in campaign}) == 7
    for attack in campaign:
        attack.validate()


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------


def test_campaign_build_is_logged(four_seeds):
    logger = RecordingLogger()
    build_campaign("prompt_injection", 7, logger=logger)

    (fields,) = logger.fields_for("campaign_built")
    assert fields == {
        "category": "prompt_injection",
        "requested": 7,
        "produced": 7,
        "seeds": 4,
        # Anything not lifted straight out of the corpus, which with no
        # generator attached means the three repeated seeds.
        "generated": 3,
    }


def test_logged_generated_count_tracks_the_model(four_seeds):
    logger = RecordingLogger()
    # divmod(9, 4) == (2, 1), so the first seed asks for two variants and the
    # rest for one each; the surplus item is dropped by those three.
    client = FakeJSONClient([variants_json(item("rephrased one"), item("rephrased two"))])
    build_campaign("prompt_injection", 9, generator=make_generator(client), logger=logger)

    (fields,) = logger.fields_for("campaign_built")
    assert fields["requested"] == 9
    assert fields["produced"] == 9
    assert fields["generated"] == 5  # 9 attacks - 4 hand-written seeds
