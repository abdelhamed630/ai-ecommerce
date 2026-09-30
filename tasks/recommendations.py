"""Recommendation cache background tasks."""

from celery_app import celery_app
from database import database
from services import recommendation_cache_service

REFRESH_USER_TASK_NAME = "recommendations.refresh_user_cache"


@celery_app.task(name=REFRESH_USER_TASK_NAME)
def refresh_user_recommendation_cache(user_id: int, limit: int = 10) -> dict:
    """Recompute and cache one user's personalized recommendations.

    `user_id` is supplied by trusted server code (never an HTTP parameter).
    A no-op result is returned if the cache is unavailable.
    """
    db = database.SessionLocal()
    try:
        count = recommendation_cache_service.refresh_user_recommendations(db, int(user_id), int(limit))
        return {"status": "refreshed", "user_id": int(user_id), "items": count}
    finally:
        db.close()
