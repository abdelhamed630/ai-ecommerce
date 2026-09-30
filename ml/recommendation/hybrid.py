"""Hybrid recommendation: combine content-based and collaborative scores.

Combination logic ONLY. It never computes similarity itself and touches
neither the database nor the ML models; it receives two {product_id: raw_score}
mappings (from content_based.ContentIndex.score_profile and
collaborative.score_candidates) and merges them.

    hybrid = content_weight * norm(content_raw) + collaborative_weight * norm(collaborative_raw)

Normalization
-------------
The two raw scales differ (content: sum of weight * cosine in [0, 1];
collaborative: sum of user_similarity * weight), so each raw score is mapped
to [0, 1) with a saturating function

    norm(x) = x / (x + saturation)

It is monotonic (order inside one source is preserved), needs no knowledge of
the other candidates (a candidate's score does not depend on what else is in
the result, so strength is still comparable between requests: one purchase
beats one view), and a raw score equal to `saturation` maps to 0.5. The
service passes the PURCHASE weight as `saturation`.

The two weights are normalized to sum to 1, so a hybrid score is in [0, 1).
A candidate found by only one source is scored by that source alone (the other
term is 0); scores are not rescaled when a source has no candidates.
"""

from dataclasses import dataclass
from typing import List, Mapping, Optional, Set

SOURCE_HYBRID = "hybrid"
SOURCE_CONTENT = "content"
SOURCE_COLLABORATIVE = "collaborative"

DEFAULT_SATURATION = 3.0


@dataclass(frozen=True)
class HybridWeights:
    """Relative importance of each signal. Normalized to sum to 1."""

    content: float = 0.5
    collaborative: float = 0.5

    def __post_init__(self):
        if self.content < 0 or self.collaborative < 0:
            raise ValueError("hybrid weights must be non-negative")
        total = self.content + self.collaborative
        if total <= 0:
            raise ValueError("at least one hybrid weight must be positive")
        object.__setattr__(self, "content", self.content / total)
        object.__setattr__(self, "collaborative", self.collaborative / total)


@dataclass(frozen=True)
class HybridCandidate:
    product_id: int
    content_score: float  # normalized [0, 1); 0.0 if content did not propose it
    collaborative_score: float  # normalized [0, 1); 0.0 if CF did not propose it
    hybrid_score: float
    source: str  # "hybrid" | "content" | "collaborative"


def normalize_score(raw: float, saturation: float = DEFAULT_SATURATION) -> float:
    """Saturating map of a raw non-negative score to [0, 1)."""
    if saturation <= 0:
        raise ValueError("saturation must be positive")
    if raw <= 0:
        return 0.0
    return float(raw / (raw + saturation))


def combine(
    content_scores: Mapping[int, float],
    collaborative_scores: Mapping[int, float],
    weights: Optional[HybridWeights] = None,
    limit: Optional[int] = None,
    exclude_product_ids: Optional[Set[int]] = None,
    saturation: float = DEFAULT_SATURATION,
) -> List[HybridCandidate]:
    """Merge candidate sets into one ranking.

    - A product present in both mappings is merged into ONE candidate whose
      score is the weighted sum of both normalized scores.
    - `exclude_product_ids` (e.g. already-interacted products, the target
      product) never appear in the output.
    - Only candidates with a positive hybrid score are returned, sorted by
      hybrid score descending (ties: ascending product id), truncated to
      `limit` when given.
    """
    weights = weights or HybridWeights()
    excluded = exclude_product_ids or set()
    if limit is not None and limit <= 0:
        return []

    candidates: List[HybridCandidate] = []
    for pid in set(content_scores) | set(collaborative_scores):
        if pid in excluded:
            continue
        content = normalize_score(content_scores.get(pid, 0.0), saturation)
        collaborative = normalize_score(collaborative_scores.get(pid, 0.0), saturation)
        # Zero-weight sources contribute nothing, including to `source`.
        content_active = content > 0 and weights.content > 0
        collab_active = collaborative > 0 and weights.collaborative > 0
        if not (content_active or collab_active):
            continue
        score = weights.content * content + weights.collaborative * collaborative
        if content_active and collab_active:
            source = SOURCE_HYBRID
        elif content_active:
            source = SOURCE_CONTENT
        else:
            source = SOURCE_COLLABORATIVE
        candidates.append(HybridCandidate(pid, content, collaborative, score, source))

    candidates.sort(key=lambda c: (-c.hybrid_score, c.product_id))
    return candidates if limit is None else candidates[:limit]
