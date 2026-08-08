"""Assemble a campaign: N attacks in a category, seeds expanded to variants.

Distributes the requested count across the category's seeds round-robin so that
``--n 10`` over five seeds gives two attacks per seed rather than ten variants of
the first one.
"""

from __future__ import annotations

from attacker.corpus import seeds_by_category
from attacker.generator import VariantGenerator, expand_seed
from attacker.models import Attack
from common.logging import EventLogger, NullLogger

__all__ = ["build_campaign"]


def build_campaign(
    category: str,
    n: int,
    *,
    generator: VariantGenerator | None = None,
    include_seed: bool = True,
    logger: EventLogger | None = None,
) -> list[Attack]:
    """Return exactly ``n`` attacks for ``category`` (or fewer if seeds run dry)."""
    log = logger or NullLogger()
    seeds = seeds_by_category(category)
    if not seeds:
        raise ValueError(f"no seeds for category {category!r}")

    # How many attacks to draw from each seed, spreading the remainder over the
    # first few seeds so totals always sum to n.
    base, extra = divmod(n, len(seeds))
    per_seed = [base + (1 if i < extra else 0) for i in range(len(seeds))]

    campaign: list[Attack] = []
    for seed, count in zip(seeds, per_seed, strict=True):
        if count == 0:
            continue
        derived = expand_seed(seed, count, generator, include_seed=include_seed)
        campaign.extend(derived)

    campaign = campaign[:n]
    log.emit(
        "campaign_built",
        category=category,
        requested=n,
        produced=len(campaign),
        seeds=len(seeds),
        generated=sum(1 for a in campaign if a.origin == "generated"),
    )
    return campaign
