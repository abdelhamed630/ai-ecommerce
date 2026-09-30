"""Deterministic, human-readable recommendation reasons.

Pure logic (no FastAPI/SQLAlchemy). A reason is derived ONLY from signals the
engine actually used for that item:

- `source`: which scorers proposed the product (hybrid.SOURCE_*, or "popular"
  for the cold-start / no-candidate fallback).
- `interaction_kinds`: which kinds of interaction make up the user's profile
  (the content signal is a weighted similarity to ALL of the user's
  interacted products, so the reason names the kinds present, not one
  specific product).

Nothing is invented: a content-only item never claims "similar users", a
collaborative-only item never claims "similar to products you viewed", and a
kind the user never performed is never mentioned.
"""

from typing import Iterable, Optional

from ml.recommendation.hybrid import SOURCE_COLLABORATIVE, SOURCE_CONTENT, SOURCE_HYBRID

SOURCE_POPULAR = "popular"

# Interaction kind (InteractionType.value) -> verb phrase, in a fixed display
# order so the output never depends on set iteration order.
_KIND_PHRASES = (
    ("VIEW", "viewed"),
    ("CART_ADD", "added to your cart"),
    ("PURCHASE", "purchased"),
)

REASON_COLLABORATIVE = "Popular among users with similar activity"
REASON_POPULAR = "Popular among shoppers"
REASON_GENERIC_CONTENT = "Similar to products from your activity"


def _join(phrases: list) -> str:
    if len(phrases) <= 1:
        return "".join(phrases)
    return ", ".join(phrases[:-1]) + " and " + phrases[-1]


def _content_reason(interaction_kinds: Iterable[str]) -> str:
    kinds = set(interaction_kinds)
    phrases = [phrase for kind, phrase in _KIND_PHRASES if kind in kinds]
    if not phrases:
        return REASON_GENERIC_CONTENT
    return f"Similar to products you {_join(phrases)}"


def build_reason(source: Optional[str], interaction_kinds: Iterable[str] = ()) -> Optional[str]:
    """Reason text for one recommendation, or None for an unknown source."""
    if source == SOURCE_CONTENT:
        return _content_reason(interaction_kinds)
    if source == SOURCE_COLLABORATIVE:
        return REASON_COLLABORATIVE
    if source == SOURCE_HYBRID:
        return f"{_content_reason(interaction_kinds)} and popular among users with similar activity"
    if source == SOURCE_POPULAR:
        return REASON_POPULAR
    return None
