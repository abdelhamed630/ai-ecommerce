from typing import List, Optional

from pydantic import BaseModel

from schemas.product import ProductOut


class SimilarProductItem(BaseModel):
    product: ProductOut
    similarity_score: float


class ProductRecommendationsOut(BaseModel):
    product_id: int
    recommendations: List[SimilarProductItem]


class UserRecommendationItem(BaseModel):
    product: ProductOut
    # Final ranking score (hybrid score for personalized results; weighted
    # popularity for the cold-start fallback). Unchanged, required field.
    score: float
    # Additive, optional explainability fields (Phase 2). Older clients that
    # only read `product` and `score` are unaffected.
    # "hybrid" | "content" | "collaborative" | "popular"
    recommendation_source: Optional[str] = None
    content_score: Optional[float] = None
    collaborative_score: Optional[float] = None
    # Additive (Phase 3): short, deterministic explanation derived from the
    # signals that actually produced this item (see
    # ml/recommendation/reasons.py). Optional so older clients are unaffected.
    reason: Optional[str] = None


class UserRecommendationsOut(BaseModel):
    recommendations: List[UserRecommendationItem]
