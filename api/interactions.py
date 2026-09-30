from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from core.security import get_current_user
from database.database import get_db
from models.interaction import InteractionType
from models.user import User
from schemas.interaction import InteractionCreate, InteractionOut
from services import interaction_service

router = APIRouter(prefix="/interactions", tags=["Interactions"])


@router.post("/", response_model=InteractionOut)
def create_interaction(
    payload: InteractionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return interaction_service.record_interaction(
        db, current_user.id, payload.product_id, payload.interaction_type
    )


@router.get("/me", response_model=List[InteractionOut])
def read_my_interactions(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return interaction_service.get_user_interactions(db, current_user.id, skip=skip, limit=limit)


@router.get("/product/{product_id}")
def read_product_interactions(
    product_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Aggregate-only response — never exposes which user did what.
    return interaction_service.get_product_interaction_summary(db, product_id)


@router.post("/view/{product_id}", response_model=InteractionOut)
def record_product_view(
    product_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return interaction_service.record_interaction(
        db, current_user.id, product_id, InteractionType.VIEW
    )
