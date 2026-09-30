"""Synthetic churn data shared by Phase 4 tests."""

import uuid
from datetime import datetime, timedelta

from models.interaction import InteractionType, ProductInteraction
from models.order import Order, OrderStatus
from models.product import Product
from models.user import User

WINDOW = 30


def naive_utc_now() -> datetime:
    return datetime.utcnow()


def make_users(session, n):
    users = [User(email=f"c{i}_{uuid.uuid4().hex[:8]}@ex.com", hashed_password="x") for i in range(n)]
    session.add_all(users)
    session.commit()
    return users


def seed_customers(session, now, n=80):
    """Loyal customers (even index) keep ordering every 10 days until now;
    churners stop 10-70 days ago. Ordered with COMPLETED status; a few
    CANCELLED orders are added that must never count."""
    product = Product(name="Widget", price=10.0, stock=100)
    session.add(product)
    session.commit()
    users = make_users(session, n)
    for i, user in enumerate(users):
        start = 150 + (i % 5) * 5
        end = 2 if i % 2 == 0 else 10 + (i % 7) * 10
        day = start
        while day >= end:
            session.add(
                Order(
                    user_id=user.id,
                    status=OrderStatus.COMPLETED,
                    total_price=20.0 + (i % 5) * 10,
                    created_at=now - timedelta(days=day),
                )
            )
            day -= 10
        session.add(
            Order(user_id=user.id, status=OrderStatus.CANCELLED, total_price=999.0,
                  created_at=now - timedelta(days=1))
        )
        for k in range(3 + (i % 3)):
            session.add(
                ProductInteraction(
                    user_id=user.id, product_id=product.id,
                    interaction_type=InteractionType.VIEW if k % 2 == 0 else InteractionType.CART_ADD,
                    created_at=now - timedelta(days=end + k),
                )
            )
    session.commit()
    return users
