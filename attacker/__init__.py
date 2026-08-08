"""The attacker: seed corpus, LLM variant generator, campaign assembly."""

from attacker.campaign import build_campaign
from attacker.corpus import load_seeds, seeds_by_category
from attacker.generator import VariantGenerator, expand_seed
from attacker.models import CATEGORIES, Attack, AttackValidationError

__all__ = [
    "CATEGORIES",
    "Attack",
    "AttackValidationError",
    "VariantGenerator",
    "build_campaign",
    "expand_seed",
    "load_seeds",
    "seeds_by_category",
]
