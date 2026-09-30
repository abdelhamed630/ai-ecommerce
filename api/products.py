from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from core.security import get_current_admin_or_seller_user
from database.database import get_db
from models.user import User, UserRole
from schemas.product import ProductCreate, ProductOut, ProductUpdate
from services import product_service

router = APIRouter(prefix="/products", tags=["Products"])


def _get_product_or_404(db: Session, product_id: int):
    product = product_service.get_product(db, product_id)
    if not product:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")
    return product


def _check_owns_product_or_admin(current_user: User, product) -> None:
    """ADMIN may manage any product. SELLER may only manage their own
    (product.seller_id == current_user.id). Callers must already have
    gone through `get_current_admin_or_seller_user`, so `current_user` is
    guaranteed to be one of those two roles by this point — a plain USER
    never reaches here."""
    if current_user.role == UserRole.SELLER and product.seller_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You may only modify your own products.",
        )


@router.get("/", response_model=List[ProductOut])
def list_products(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    # Browsing is open to USER, SELLER, and ADMIN alike (and, as before
    # this change, to unauthenticated requests too) — no role check here.
    return product_service.get_products(db, skip=skip, limit=limit)


@router.get("/{product_id}", response_model=ProductOut)
def get_product(product_id: int, db: Session = Depends(get_db)):
    return _get_product_or_404(db, product_id)


@router.post("/", response_model=ProductOut)
def create_product(
    product: ProductCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_or_seller_user),
):
    # SELLER-created products are owned by that seller. ADMIN is a system
    # administrator, not automatically a seller/vendor, so an
    # admin-created product gets no owner (seller_id = NULL) — the same
    # state as a legacy product created before ownership existed. Either
    # way, seller_id always comes from the authenticated current user's
    # role — never from the request body (ProductCreate has no seller_id
    # field).
    seller_id = current_user.id if current_user.role == UserRole.SELLER else None
    return product_service.create_product(db, product, seller_id=seller_id)


@router.patch("/{product_id}", response_model=ProductOut)
def update_product(
    product_id: int,
    product: ProductUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_or_seller_user),
):
    existing = _get_product_or_404(db, product_id)
    _check_owns_product_or_admin(current_user, existing)

    updated = product_service.update_product(db, product_id, product)
    return updated


@router.delete("/{product_id}")
def delete_product(
    product_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_or_seller_user),
):
    existing = _get_product_or_404(db, product_id)
    _check_owns_product_or_admin(current_user, existing)

    product_service.delete_product(db, product_id)
    return {"message": "Product deleted successfully"}
