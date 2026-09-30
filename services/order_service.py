from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from models.user import User, UserRole
from models.cart import CartItem
from models.order import Order, OrderItem, OrderStatus
from models.product import Product
from services import cache_invalidation
from services.cart_service import get_or_create_cart

# Allowed forward transitions for PATCH /orders/{id}/complete
_ADVANCE_TRANSITIONS = {
    OrderStatus.PENDING: OrderStatus.CONFIRMED,
    OrderStatus.CONFIRMED: OrderStatus.COMPLETED,
}

# States from which cancellation is allowed
_CANCELLABLE_STATES = {OrderStatus.PENDING, OrderStatus.CONFIRMED}


def create_order_from_cart(db: Session, user_id: int) -> Order:
    cart = get_or_create_cart(db, user_id)
    cart_items = (
        db.query(CartItem).filter(CartItem.cart_id == cart.id).all()
    )

    if not cart_items:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot create an order from an empty cart",
        )

    # --- Validate stock for every item BEFORE writing anything ---
    # If any single product doesn't have enough stock, the whole operation
    # fails and nothing is created or modified (no partial orders).
    insufficient = []
    for item in cart_items:
        product = db.query(Product).filter(Product.id == item.product_id).first()
        if not product:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Product {item.product_id} no longer exists",
            )
        if item.quantity > product.stock:
            insufficient.append(
                f"'{product.name}' (requested {item.quantity}, available {product.stock})"
            )

    if insufficient:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Not enough stock for: " + ", ".join(insufficient),
        )

    # --- Atomic write phase ---
    try:
        total_price = 0.0
        order = Order(user_id=user_id, status=OrderStatus.PENDING, total_price=0.0)
        db.add(order)
        db.flush()  # assign order.id without committing

        for item in cart_items:
            product = db.query(Product).filter(Product.id == item.product_id).first()
            subtotal = product.price * item.quantity
            total_price += subtotal

            db.add(
                OrderItem(
                    order_id=order.id,
                    product_id=product.id,
                    product_name=product.name,
                    product_price=product.price,
                    quantity=item.quantity,
                    subtotal=subtotal,
                )
            )
            product.stock -= item.quantity

        order.total_price = total_price

        # Clear the cart now that the order has captured everything it needs.
        db.query(CartItem).filter(CartItem.cart_id == cart.id).delete()

        db.commit()
    except Exception:
        db.rollback()
        raise

    db.refresh(order)
    return order


def get_orders(db: Session, user_id: int):
    return db.query(Order).filter(Order.user_id == user_id).order_by(Order.id.desc()).all()


def _get_owned_order_or_404(db: Session, user_id: int, order_id: int) -> Order:
    order = (
        db.query(Order)
        .filter(Order.id == order_id, Order.user_id == user_id)
        .first()
    )
    if not order:
        # Same 404 whether the order doesn't exist or belongs to someone else,
        # so we never reveal another user's order exists.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Order not found")
    return order


def get_order(db: Session, user_id: int, order_id: int) -> Order:
    return _get_owned_order_or_404(db, user_id, order_id)


def cancel_order(db: Session, user_id: int, order_id: int) -> Order:
    order = _get_owned_order_or_404(db, user_id, order_id)

    if order.status not in _CANCELLABLE_STATES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel an order with status '{order.status.value}'",
        )

    try:
        for item in order.items:
            product = db.query(Product).filter(Product.id == item.product_id).first()
            if product:
                product.stock += item.quantity

        order.status = OrderStatus.CANCELLED
        db.commit()
    except Exception:
        db.rollback()
        raise

    db.refresh(order)
    return order


def advance_order(
    db: Session,
    current_user: User,
    order_id: int,
) -> Order:
    """Advance an order by an authorized seller or admin.

    ADMIN can advance any order.

    SELLER can advance an order only when every product in that
    order belongs to that seller.
    """

    order = db.query(Order).filter(Order.id == order_id).first()

    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Order not found",
        )

    if current_user.role == UserRole.SELLER:
        if not order.items:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Order has no items",
            )

        for item in order.items:
            product = (
                db.query(Product)
                .filter(Product.id == item.product_id)
                .first()
            )

            if not product or product.seller_id != current_user.id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You do not have permission to manage this order.",
                )

    next_status = _ADVANCE_TRANSITIONS.get(order.status)

    if next_status is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Cannot advance an order with status "
                f"'{order.status.value}'"
            ),
        )

    order.status = next_status

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise

    db.refresh(order)

    if next_status == OrderStatus.COMPLETED:
        cache_invalidation.invalidate_user_caches(order.user_id)

    return order