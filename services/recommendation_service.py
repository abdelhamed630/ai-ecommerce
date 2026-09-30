"""Recommendation service: database access + orchestration.

Architecture (Phase 2):

    API (/recommendations/*)
        -> this module (queries, response shaping)
            -> ml.recommendation.hybrid         combines candidate scores
                -> ml.recommendation.content_based   TF-IDF + cosine (product text)
                -> ml.recommendation.collaborative   user-user cosine (user x product matrix)

The ML modules are pure (no SQLAlchemy/FastAPI): this file loads rows,
turns interactions into weights (INTERACTION_WEIGHTS below), and passes
plain values in.

- `get_similar_products`: content-based only, unchanged behavior.
- `get_user_recommendations`: hybrid (content + collaborative) for the
  authenticated user; cold start (no interactions) -> popular products.
- `get_collaborative_recommendations`: collaborative signal only.

Still in-process and per-request (no Redis/Celery); see docs/recommendations.md
for limitations.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.config import settings
from ml.recommendation import collaborative, content_based, hybrid, reasons
from models.interaction import InteractionType, ProductInteraction
from models.product import Product

# Heuristic, hand-picked interaction weights — NOT machine-learned.
# A purchase is the strongest signal of genuine interest, a cart-add is a
# medium signal, and a view is the weakest signal. These are simple,
# explainable multipliers used to combine similarity scores across a user's
# interaction history.
INTERACTION_WEIGHTS: Dict[InteractionType, float] = {
    InteractionType.VIEW: 1.0,
    InteractionType.CART_ADD: 2.0,
    InteractionType.PURCHASE: 3.0,
}


def build_product_text(product: Product) -> str:
    """Deterministic textual representation of a product (name, description,
    category, brand; None fields become empty). Delegates to
    ml.recommendation.content_based.build_document so both recommendation
    paths use one definition of "product text"."""
    return content_based.build_document(
        product.name, product.description, product.category, product.brand
    )


def _load_catalog_index(db: Session) -> Tuple[List[int], Optional[content_based.ContentIndex]]:
    """One query loading only the five text columns of the whole catalog,
    then the (cached) fitted TF-IDF index. Returns (product_ids, index);
    index is None when the catalog has no usable text."""
    rows = (
        db.query(
            Product.id,
            Product.name,
            Product.description,
            Product.category,
            Product.brand,
        )
        .order_by(Product.id.asc())
        .all()
    )
    ids = [row.id for row in rows]
    if len(ids) <= 1:
        return ids, None
    documents = [
        content_based.build_document(r.name, r.description, r.category, r.brand)
        for r in rows
    ]
    return ids, content_based.get_or_build_index(ids, documents)


def get_similar_products(db: Session, product_id: int, limit: int) -> List[dict]:
    """Returns up to `limit` products most similar to `product_id`, sorted
    by similarity score descending. The target product itself is excluded.

    Content-based (TF-IDF + cosine, see ml/recommendation/content_based.py).
    Efficiency: one query loads only the five text columns for the whole
    catalog (no ORM hydration), the fitted TF-IDF index is reused while the
    catalog text is unchanged, and full Product rows are fetched only for the
    top `limit` results.
    """
    ids, index = _load_catalog_index(db)
    if product_id not in ids:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")
    if index is None:
        return []

    ranked = index.similar_to(product_id, limit)
    if not ranked:
        return []

    products_by_id = {
        p.id: p
        for p in db.query(Product).filter(Product.id.in_([pid for pid, _ in ranked])).all()
    }
    return [
        {"product": products_by_id[pid], "similarity_score": score}
        for pid, score in ranked
        if pid in products_by_id
    ]


def _popular_item(product: Product, score: float) -> dict:
    return {
        "product": product,
        "score": score,
        "recommendation_source": "popular",
        "content_score": None,
        "collaborative_score": None,
        "reason": reasons.build_reason(reasons.SOURCE_POPULAR),
    }


def _get_popular_products(db: Session, limit: int, exclude_ids: Optional[set] = None) -> List[dict]:
    """Cold-start fallback: ranks products by weighted interaction count
    across ALL users. If there is no interaction data at all, falls back to
    the newest products (a deterministic, always-available ordering).
    """
    exclude_ids = exclude_ids or set()

    # Aggregated in SQL (one row per product/type, not per interaction) and
    # joined to `products` so interactions with deleted products never take up
    # a slot in the result.
    rows = (
        db.query(
            ProductInteraction.product_id,
            ProductInteraction.interaction_type,
            func.count(ProductInteraction.id),
        )
        .join(Product, Product.id == ProductInteraction.product_id)
        .group_by(ProductInteraction.product_id, ProductInteraction.interaction_type)
        .all()
    )

    if not rows:
        products = (
            db.query(Product)
            .order_by(Product.created_at.desc(), Product.id.desc())
            .limit(limit + len(exclude_ids))
            .all()
        )
        filtered = [p for p in products if p.id not in exclude_ids][:limit]
        return [_popular_item(p, 0.0) for p in filtered]

    weight_by_product: Dict[int, float] = defaultdict(float)
    for product_id, interaction_type, count in rows:
        weight_by_product[product_id] += INTERACTION_WEIGHTS[interaction_type] * count

    # Ties on weight are broken by ascending product id so the order never
    # depends on database row order.
    ranked_ids = [
        pid
        for pid, _ in sorted(weight_by_product.items(), key=lambda kv: (-kv[1], kv[0]))
        if pid not in exclude_ids
    ][:limit]

    if not ranked_ids:
        return []

    products_by_id = {
        p.id: p for p in db.query(Product).filter(Product.id.in_(ranked_ids)).all()
    }
    return [
        _popular_item(products_by_id[pid], weight_by_product[pid])
        for pid in ranked_ids
        if pid in products_by_id
    ]


def _default_weights() -> hybrid.HybridWeights:
    return hybrid.HybridWeights(
        content=settings.HYBRID_CONTENT_WEIGHT,
        collaborative=settings.HYBRID_COLLABORATIVE_WEIGHT,
    )


def _load_user_signals(db: Session, user_id: int) -> Tuple[Dict[int, float], Set[str]]:
    """The user's own interactions (one query, filtered by `user_id` only):
    ({product_id: summed weight}, {interaction kinds present}). This is the
    only place the target user's behavior enters the pipeline; the kinds are
    used solely to word recommendation reasons truthfully."""
    rows = (
        db.query(ProductInteraction.product_id, ProductInteraction.interaction_type)
        .filter(ProductInteraction.user_id == user_id)
        .all()
    )
    profile: Dict[int, float] = defaultdict(float)
    kinds: Set[str] = set()
    for product_id, interaction_type in rows:
        profile[product_id] += INTERACTION_WEIGHTS[interaction_type]
        kinds.add(interaction_type.value)
    return dict(profile), kinds


def _load_user_profile(db: Session, user_id: int) -> Dict[int, float]:
    """The user's own interactions as {product_id: summed weight}."""
    return _load_user_signals(db, user_id)[0]


def _load_neighbor_entries(db: Session, user_id: int) -> List[Tuple[int, int, float]]:
    """(user_id, product_id, weight) for every OTHER user who interacted with
    at least one product the target also interacted with, in a single query.

    Users with no overlap have cosine similarity 0 and could never be
    neighbors, so excluding them is exact, not an approximation; it just
    avoids loading the whole interaction table.
    """
    own_products = select(ProductInteraction.product_id).where(
        ProductInteraction.user_id == user_id
    )
    neighbor_users = select(ProductInteraction.user_id).where(
        ProductInteraction.product_id.in_(own_products),
        ProductInteraction.user_id != user_id,
    )
    rows = (
        db.query(
            ProductInteraction.user_id,
            ProductInteraction.product_id,
            ProductInteraction.interaction_type,
        )
        .filter(ProductInteraction.user_id.in_(neighbor_users))
        .all()
    )
    return [(uid, pid, INTERACTION_WEIGHTS[itype]) for uid, pid, itype in rows]


def _collaborative_scores(
    db: Session, user_id: int, profile: Dict[int, float]
) -> Dict[int, float]:
    entries = [(user_id, pid, weight) for pid, weight in profile.items()]
    entries.extend(_load_neighbor_entries(db, user_id))
    im = collaborative.build_interaction_matrix(entries)
    return collaborative.score_candidates(im, user_id, settings.CF_MAX_NEIGHBORS)


def _ranked_candidates(
    db: Session,
    profile: Dict[int, float],
    user_id: int,
    limit: int,
    weights: hybrid.HybridWeights,
) -> List[hybrid.HybridCandidate]:
    """Score with each enabled signal, then let the hybrid layer merge."""
    interacted_ids: Set[int] = set(profile)
    catalog_ids, index = _load_catalog_index(db)

    content_scores: Dict[int, float] = {}
    if weights.content > 0 and index is not None:
        content_scores = index.score_profile(profile)

    collaborative_scores: Dict[int, float] = {}
    if weights.collaborative > 0:
        catalog = set(catalog_ids)
        collaborative_scores = {
            pid: score
            for pid, score in _collaborative_scores(db, user_id, profile).items()
            if pid in catalog  # ignore interactions with products that no longer exist
        }

    return hybrid.combine(
        content_scores,
        collaborative_scores,
        weights=weights,
        limit=limit,
        exclude_product_ids=interacted_ids,
        saturation=INTERACTION_WEIGHTS[InteractionType.PURCHASE],
    )


def _to_items(
    db: Session,
    candidates: List[hybrid.HybridCandidate],
    interaction_kinds: Set[str] = frozenset(),
) -> List[dict]:
    if not candidates:
        return []
    products_by_id = {
        p.id: p
        for p in db.query(Product).filter(Product.id.in_([c.product_id for c in candidates])).all()
    }
    return [
        {
            "product": products_by_id[c.product_id],
            "score": c.hybrid_score,
            "recommendation_source": c.source,
            "content_score": c.content_score,
            "collaborative_score": c.collaborative_score,
            "reason": reasons.build_reason(c.source, interaction_kinds),
        }
        for c in candidates
        if c.product_id in products_by_id
    ]


def get_user_recommendations(
    db: Session,
    user_id: int,
    limit: int,
    weights: Optional[hybrid.HybridWeights] = None,
) -> List[dict]:
    """Personalized hybrid recommendations for ONE user (the caller passes the
    authenticated user's id; this function never accepts anyone else's data).

      1. Load the user's interactions -> {product_id: weight}.
      2. No interactions -> cold start: popular products (unchanged fallback).
      3. Content score: weighted TF-IDF cosine to what the user interacted with.
         Collaborative score: interactions of users with a similar history.
      4. Hybrid layer normalizes, merges duplicates, excludes already-interacted
         products, sorts descending, applies `limit`.
      5. If neither signal proposes anything (e.g. nothing textually similar
         and no overlapping users), fall back to popular products, still
         excluding already-interacted ones.

    `weights` defaults to the configured HYBRID_* settings; it exists so
    callers/tests can request content-only (1, 0) or collaborative-only (0, 1).
    """
    profile, kinds = _load_user_signals(db, user_id)
    if not profile:
        return _get_popular_products(db, limit)

    candidates = _ranked_candidates(db, profile, user_id, limit, weights or _default_weights())
    items = _to_items(db, candidates, kinds)
    if not items:
        return _get_popular_products(db, limit, exclude_ids=set(profile))
    return items


def get_collaborative_recommendations(db: Session, user_id: int, limit: int) -> List[dict]:
    """Collaborative-filtering signal only, with NO fallback: returns [] when
    the user has no interactions or no similar users. `score` is the
    normalized collaborative score."""
    profile, kinds = _load_user_signals(db, user_id)
    if not profile:
        return []
    candidates = _ranked_candidates(
        db, profile, user_id, limit, hybrid.HybridWeights(content=0.0, collaborative=1.0)
    )
    return _to_items(db, candidates, kinds)
