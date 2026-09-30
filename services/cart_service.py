from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from models.cart import Cart, CartItem
from models.product import Product
from schemas.cart import CartItemAdd, CartOut


def get_or_create_cart(db: Session, user_id: int) -> Cart:
    cart = db.query(Cart).filter(Cart.user_id == user_id).first()
    if cart:
        return cart
    cart = Cart(user_id=user_id)
    db.add(cart)
    db.commit()
    db.refresh(cart)
    return cart


def _serialize_cart(cart: Cart) -> CartOut:
    items = []
    for item in cart.items:
        subtotal = item.product.price * item.quantity
        items.append(
            {
                "id": item.id,
                "product_id": item.product_id,
                "quantity": item.quantity,
                "product": item.product,
                "subtotal": subtotal,
            }
        )

    return CartOut(
        id=cart.id,
        user_id=cart.user_id,
        items=items,
        total_items=sum(i["quantity"] for i in items),
        total_price=sum(i["subtotal"] for i in items),
        created_at=cart.created_at,
        updated_at=cart.updated_at,
    )


def get_cart(db: Session, user_id: int) -> CartOut:
    cart = get_or_create_cart(db, user_id)
    return _serialize_cart(cart)


def _get_product_or_404(db: Session, product_id: int) -> Product:
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")
    return product


def add_item_to_cart(db: Session, user_id: int, payload: CartItemAdd) -> CartOut:
    product = _get_product_or_404(db, payload.product_id)

    cart = get_or_create_cart(db, user_id)
    existing_item = (
        db.query(CartItem)
        .filter(CartItem.cart_id == cart.id, CartItem.product_id == payload.product_id)
        .first()
    )

    new_quantity = payload.quantity + (existing_item.quantity if existing_item else 0)
    if new_quantity > product.stock:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Not enough stock for '{product.name}'. Available: {product.stock}",
        )

    if existing_item:
        existing_item.quantity = new_quantity
    else:
        existing_item = CartItem(
            cart_id=cart.id, product_id=payload.product_id, quantity=payload.quantity
        )
        db.add(existing_item)

    db.commit()
    db.refresh(cart)
    return _serialize_cart(cart)


def _get_owned_item_or_404(db: Session, user_id: int, item_id: int) -> CartItem:
    cart = get_or_create_cart(db, user_id)
    item = (
        db.query(CartItem)
        .filter(CartItem.id == item_id, CartItem.cart_id == cart.id)
        .first()
    )
    if not item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Cart item not found")
    return item


def update_cart_item(db: Session, user_id: int, item_id: int, quantity: int) -> CartOut:
    item = _get_owned_item_or_404(db, user_id, item_id)

    product = _get_product_or_404(db, item.product_id)
    if quantity > product.stock:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Not enough stock for '{product.name}'. Available: {product.stock}",
        )

    item.quantity = quantity
    db.commit()

    cart = get_or_create_cart(db, user_id)
    db.refresh(cart)
    return _serialize_cart(cart)


def remove_cart_item(db: Session, user_id: int, item_id: int) -> CartOut:
    item = _get_owned_item_or_404(db, user_id, item_id)
    db.delete(item)
    db.commit()

    cart = get_or_create_cart(db, user_id)
    db.refresh(cart)
    return _serialize_cart(cart)


def clear_cart(db: Session, user_id: int) -> CartOut:
    cart = get_or_create_cart(db, user_id)
    db.query(CartItem).filter(CartItem.cart_id == cart.id).delete()
    db.commit()
    db.refresh(cart)
    return _serialize_cart(cart)
