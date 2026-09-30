from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from models.interaction import InteractionType, ProductInteraction
from models.product import Product
from services import cache_invalidation


def _get_product_or_404(db: Session, product_id: int) -> Product:
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")
    return product


def record_interaction(
    db: Session, user_id: int, product_id: int, interaction_type: InteractionType
) -> ProductInteraction:
    """Records a single user-product interaction.

    user_id always comes from the authenticated caller — never from client
    input — and created_at is always server-generated.
    """
    _get_product_or_404(db, product_id)

    interaction = ProductInteraction(
        user_id=user_id,
        product_id=product_id,
        interaction_type=interaction_type,
    )
    db.add(interaction)
    db.commit()
    db.refresh(interaction)
    # The user's own cached recommendations/churn prediction are now stale.
    cache_invalidation.invalidate_user_caches(user_id)
    return interaction


def get_user_interactions(
    db: Session, user_id: int, skip: int = 0, limit: int = 100
):
    """Returns only the calling user's own interaction history."""
    return (
        db.query(ProductInteraction)
        .filter(ProductInteraction.user_id == user_id)
        .order_by(ProductInteraction.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_product_interaction_summary(db: Session, product_id: int) -> dict:
    """Returns safe, aggregate-only interaction counts for a product.

    Never exposes which individual user performed which interaction — only
    per-type totals, which is enough signal for future recommendation work
    without leaking any single user's private activity.
    """
    _get_product_or_404(db, product_id)

    counts = {t.value: 0 for t in InteractionType}
    rows = (
        db.query(ProductInteraction.interaction_type)
        .filter(ProductInteraction.product_id == product_id)
        .all()
    )
    for (interaction_type,) in rows:
        counts[interaction_type.value] += 1

    return {
        "product_id": product_id,
        "interaction_counts": counts,
        "total_interactions": sum(counts.values()),
    }
