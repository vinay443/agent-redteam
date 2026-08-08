"""Load and query the seed corpus."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from attacker.models import CATEGORIES, Attack, AttackValidationError

__all__ = ["load_seeds", "seeds_by_category", "seed_path"]


def seed_path() -> Path:
    return Path(__file__).resolve().parent / "seeds.yaml"


def load_seeds(path: str | Path | None = None) -> list[Attack]:
    """Parse and validate every seed. Raises on the first malformed entry."""
    seed_file = Path(path) if path else seed_path()
    with seed_file.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    entries = raw.get("attacks")
    if not isinstance(entries, list) or not entries:
        raise AttackValidationError(f"{seed_file} has no 'attacks' list")

    seen: set[str] = set()
    attacks: list[Attack] = []
    for entry in entries:
        attack = Attack.from_dict(entry)
        attack.origin = "seed"
        if attack.id in seen:
            raise AttackValidationError(f"duplicate seed id {attack.id!r}")
        seen.add(attack.id)
        attacks.append(attack)
    return attacks


def seeds_by_category(
    category: str | None = None, path: str | Path | None = None
) -> list[Attack]:
    """Seeds for one category, or all seeds when ``category`` is None."""
    if category is not None and category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}; known: {CATEGORIES}")
    seeds = load_seeds(path)
    if category is None:
        return seeds
    return [s for s in seeds if s.category == category]
