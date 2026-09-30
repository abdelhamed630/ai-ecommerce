"""Cache-aside wrapper for personalized recommendations.

The cache key is derived from the authenticated user's id only. One Redis hash
per user (`recommendations:user:{id}`) holds one field per requested `limit`,
so a single DEL invalidates all variants for that user. The recommendation
algorithms are untouched: on a miss (or any cache failure) the existing
`recommendation_service.get_user_recommendations` runs.
"""

from typing import List

from sqlalchemy.orm import Session

from core.cache import cache, recommendations_key
from core.config import settings
from schemas.recommendation import UserRecommendationsOut
from services import recommendation_service


def _serialize(items: List[dict]) -> List[dict]:
    return UserRecommendationsOut.model_validate({"recommendations": items}).model_dump(mode="json")[
        "recommendations"
    ]


def get_user_recommendations_cached(db: Session, user_id: int, limit: int) -> List[dict]:
    key, field = recommendations_key(user_id), f"limit:{int(limit)}"
    cached = cache.get(key, field)
    if cached is not None:
        return cached
    payload = _serialize(recommendation_service.get_user_recommendations(db, user_id, limit))
    cache.set(key, payload, settings.RECOMMENDATION_CACHE_TTL_SECONDS, field=field)
    return payload


def refresh_user_recommendations(db: Session, user_id: int, limit: int = 10) -> int:
    """Recompute and store (used by the background task). Returns item count."""
    cache.delete(recommendations_key(user_id))
    return len(get_user_recommendations_cached(db, user_id, limit))
