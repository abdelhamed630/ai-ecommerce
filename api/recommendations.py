from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from core.security import get_current_user
from database.database import get_db
from models.user import User
from schemas.recommendation import ProductRecommendationsOut, UserRecommendationsOut
from services import recommendation_cache_service, recommendation_service

router = APIRouter(prefix="/recommendations", tags=["Recommendations"])


@router.get("/product/{product_id}", response_model=ProductRecommendationsOut)
def similar_products(
    product_id: int,
    limit: int = Query(default=10, ge=1, le=50),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    recommendations = recommendation_service.get_similar_products(db, product_id, limit)
    return {"product_id": product_id, "recommendations": recommendations}


@router.get("/me", response_model=UserRecommendationsOut)
def my_recommendations(
    limit: int = Query(default=10, ge=1, le=50),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # current_user always comes from the JWT — the client cannot supply a
    # user_id to view someone else's personalized recommendations. The cache
    # key is derived from this same JWT-derived id (Redis, with graceful fallback).
    recommendations = recommendation_cache_service.get_user_recommendations_cached(
        db, current_user.id, limit
    )
    return {"recommendations": recommendations}
