import math
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from core.security import get_current_admin_or_seller_user
from database.database import get_db
from models.user import User, UserRole
from schemas.product import ProductCreate, ProductOut, ProductUpdate
from services import product_service

router = APIRouter(prefix="/products", tags=["Products"])

# GET /products/ page size: the default is unchanged (100); anything above the maximum is a 422.
DEFAULT_LIST_LIMIT = 100
MAX_LIST_LIMIT = 200
MAX_QUERY_LENGTH = 100
MAX_CATEGORY_LENGTH = 100
MAX_PRICE_FILTER = 1e12  # keeps the bound a sane finite number for the database


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


def _validate_price_bound(name: str, value: Optional[float]) -> None:
    if value is None:
        return
    if not math.isfinite(value) or value < 0 or value > MAX_PRICE_FILTER:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{name} must be a finite number between 0 and {int(MAX_PRICE_FILTER)}",
        )


@router.get("/", response_model=List[ProductOut])
def list_products(
    skip: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_LIST_LIMIT, ge=0, le=MAX_LIST_LIMIT),
    q: Optional[str] = Query(None, max_length=MAX_QUERY_LENGTH, description="Search words (name, description, category, brand)"),
    category: Optional[str] = Query(None, max_length=MAX_CATEGORY_LENGTH),
    min_price: Optional[float] = Query(None),
    max_price: Optional[float] = Query(None),
    db: Session = Depends(get_db),
):
    # Browsing is open to USER, SELLER, and ADMIN alike (and, as before
    # this change, to unauthenticated requests too) — no role check here.
    # All filtering runs in the database; every value is a bound parameter.
    _validate_price_bound("min_price", min_price)
    _validate_price_bound("max_price", max_price)
    if min_price is not None and max_price is not None and min_price > max_price:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="min_price must not be greater than max_price",
        )
    return product_service.get_products(
        db, skip=skip, limit=limit, q=q, category=category, min_price=min_price, max_price=max_price
    )


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
