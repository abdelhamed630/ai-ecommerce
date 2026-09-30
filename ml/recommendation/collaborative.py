"""User-based collaborative filtering: sparse user x product matrix + cosine.

Pure ML logic: no FastAPI, no SQLAlchemy. The service layer loads
interactions, converts each one to a positive weight (see
services/recommendation_service.INTERACTION_WEIGHTS) and hands plain
`(user_id, product_id, weight)` tuples to this module.

Pipeline:

    (user_id, product_id, weight) tuples
        -> build_interaction_matrix()      sparse users x products, duplicates summed
        -> find_similar_users()            cosine(target row, every other row)
        -> recommend()                     score(p) = sum_v sim(target, v) * weight(v, p)

Explainable by construction: a product scores high when users whose behaviour
resembles the target's interacted with it strongly (a purchase counts more
than a view because the caller passes larger weights for purchases).

Deliberately simple: no matrix factorization, no embeddings, no learned
parameters. Scores are raw (unbounded, >= 0); the hybrid layer normalizes them.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

# (user_id, product_id, weight)
InteractionEntry = Tuple[int, int, float]

DEFAULT_MAX_NEIGHBORS = 50


@dataclass(frozen=True)
class InteractionMatrix:
    """Sparse users x products matrix of accumulated interaction weights."""

    user_ids: Tuple[int, ...]
    product_ids: Tuple[int, ...]
    matrix: sparse.csr_matrix  # (n_users, n_products)

    @property
    def is_empty(self) -> bool:
        return self.matrix.shape[0] == 0 or self.matrix.shape[1] == 0

    def user_position(self, user_id: int) -> Optional[int]:
        try:
            return self.user_ids.index(user_id)
        except ValueError:
            return None


def build_interaction_matrix(entries: Iterable[InteractionEntry]) -> InteractionMatrix:
    """Build the sparse user x product matrix.

    Repeated (user, product) entries are summed (two views weigh more than
    one). Entries with a non-positive or non-finite weight are ignored.
    Users and products are ordered by ascending id, so the result is
    deterministic. An empty/invalid input yields a 0 x 0 matrix.
    """
    accumulated: Dict[Tuple[int, int], float] = {}
    for user_id, product_id, weight in entries:
        weight = float(weight)
        if not np.isfinite(weight) or weight <= 0:
            continue
        key = (int(user_id), int(product_id))
        accumulated[key] = accumulated.get(key, 0.0) + weight

    if not accumulated:
        return InteractionMatrix((), (), sparse.csr_matrix((0, 0), dtype=np.float64))

    user_ids = tuple(sorted({u for u, _ in accumulated}))
    product_ids = tuple(sorted({p for _, p in accumulated}))
    user_pos = {u: i for i, u in enumerate(user_ids)}
    product_pos = {p: j for j, p in enumerate(product_ids)}

    rows = [user_pos[u] for u, _ in accumulated]
    cols = [product_pos[p] for _, p in accumulated]
    data = list(accumulated.values())
    matrix = sparse.csr_matrix(
        (data, (rows, cols)), shape=(len(user_ids), len(product_ids)), dtype=np.float64
    )
    return InteractionMatrix(user_ids, product_ids, matrix)


def find_similar_users(
    im: InteractionMatrix,
    target_user_id: int,
    max_neighbors: int = DEFAULT_MAX_NEIGHBORS,
) -> List[Tuple[int, float]]:
    """Users most similar to `target_user_id`, best first.

    Similarity is cosine over the weighted interaction rows, in (0, 1].
    The target is never included and users with zero similarity (no shared
    product) are dropped. Ties keep ascending user-id order. Returns [] if the
    target is unknown, the matrix is empty, or nobody overlaps.
    """
    if im.is_empty or max_neighbors <= 0:
        return []
    target_pos = im.user_position(target_user_id)
    if target_pos is None or im.matrix.shape[0] < 2:
        return []

    sims = cosine_similarity(im.matrix[target_pos], im.matrix).ravel()
    order = np.argsort(-sims, kind="stable")
    neighbors: List[Tuple[int, float]] = []
    for pos in order:
        if pos == target_pos or sims[pos] <= 1e-12:
            continue
        neighbors.append((im.user_ids[pos], float(min(1.0, sims[pos]))))
        if len(neighbors) == max_neighbors:
            break
    return neighbors


def score_candidates(
    im: InteractionMatrix,
    target_user_id: int,
    max_neighbors: int = DEFAULT_MAX_NEIGHBORS,
    exclude_product_ids: Optional[Set[int]] = None,
) -> Dict[int, float]:
    """Collaborative score per candidate product.

        score(p) = sum over similar users v of  sim(target, v) * weight(v, p)

    Products the target already interacted with (always) and any in
    `exclude_product_ids` are excluded. Only strictly positive scores are
    returned. {} for a cold-start / isolated target or an empty matrix.
    """
    neighbors = find_similar_users(im, target_user_id, max_neighbors)
    if not neighbors:
        return {}

    target_pos = im.user_position(target_user_id)
    seen: Set[int] = set(exclude_product_ids or ())
    target_row = im.matrix[target_pos]
    seen.update(im.product_ids[j] for j in target_row.indices)

    neighbor_positions = [im.user_position(uid) for uid, _ in neighbors]
    similarity = sparse.csr_matrix(
        np.array([[s for _, s in neighbors]], dtype=np.float64)
    )  # (1, n_neighbors)
    scores = (similarity @ im.matrix[neighbor_positions]).toarray().ravel()

    result: Dict[int, float] = {}
    for j, score in enumerate(scores):
        pid = im.product_ids[j]
        if pid in seen or score <= 1e-12:
            continue
        result[pid] = float(score)
    return result


def recommend(
    entries: Iterable[InteractionEntry],
    target_user_id: int,
    limit: int,
    max_neighbors: int = DEFAULT_MAX_NEIGHBORS,
    exclude_product_ids: Optional[Set[int]] = None,
) -> List[Tuple[int, float]]:
    """Top `limit` (product_id, collaborative_score), sorted descending.

    Ties are broken by ascending product id (deterministic). Never raises for
    small/empty data: returns [] when no personalization is possible.
    """
    if limit <= 0:
        return []
    im = build_interaction_matrix(entries)
    scores = score_candidates(im, target_user_id, max_neighbors, exclude_product_ids)
    return rank(scores, limit)


def rank(scores: Dict[int, float], limit: Optional[int] = None) -> List[Tuple[int, float]]:
    """Sort {product_id: score} by score desc, product id asc."""
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return ordered if limit is None else ordered[:limit]
