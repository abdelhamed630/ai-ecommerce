"""
READ-ONLY diagnostic script.

Measures the actual state of the interaction dataset (users, products,
interactions, distributions, user-product matrix density) to help decide
whether Collaborative Filtering is viable yet.

This script:
  - Only ever runs SELECT queries.
  - Never inserts, updates, deletes, or migrates anything.
  - Never seeds fake data.
  - Does not import or touch any business-logic service beyond read access
    to the ORM models (User, Product, ProductInteraction).

USAGE
-----
    python scripts/analyze_cf_readiness.py

Run from the project root (so `core.config`/`database.database` are
importable), against whatever database `DATABASE_URL` currently points to.
"""

import statistics
from collections import Counter

from database.database import SessionLocal
from models.interaction import InteractionType, ProductInteraction
from models.product import Product
from models.user import User


def _bucket_counts(counts: list) -> dict:
    buckets = {"1": 0, "2-5": 0, "6-10": 0, "11-20": 0, ">20": 0}
    for c in counts:
        if c == 1:
            buckets["1"] += 1
        elif 2 <= c <= 5:
            buckets["2-5"] += 1
        elif 6 <= c <= 10:
            buckets["6-10"] += 1
        elif 11 <= c <= 20:
            buckets["11-20"] += 1
        else:
            buckets[">20"] += 1
    return buckets


def _product_bucket_counts(counts: list) -> dict:
    buckets = {"1": 0, "2-5": 0, "6-10": 0, ">10": 0}
    for c in counts:
        if c == 1:
            buckets["1"] += 1
        elif 2 <= c <= 5:
            buckets["2-5"] += 1
        elif 6 <= c <= 10:
            buckets["6-10"] += 1
        else:
            buckets[">10"] += 1
    return buckets


def analyze() -> dict:
    db = SessionLocal()
    try:
        total_users = db.query(User).count()
        total_products = db.query(Product).count()
        total_interactions = db.query(ProductInteraction).count()

        all_interactions = db.query(
            ProductInteraction.user_id,
            ProductInteraction.product_id,
            ProductInteraction.interaction_type,
        ).all()

        user_ids = {row.user_id for row in all_interactions}
        product_ids = {row.product_id for row in all_interactions}

        type_counts = Counter(row.interaction_type for row in all_interactions)

        per_user_counts = Counter(row.user_id for row in all_interactions)
        per_product_counts = Counter(row.product_id for row in all_interactions)

        unique_pairs = {(row.user_id, row.product_id) for row in all_interactions}

        active_users = len(user_ids)
        interacted_products = len(product_ids)

        matrix_cells = active_users * interacted_products
        density_pct = (len(unique_pairs) / matrix_cells * 100) if matrix_cells else 0.0

        purchase_rows = [r for r in all_interactions if r.interaction_type == InteractionType.PURCHASE]
        purchasing_users = {r.user_id for r in purchase_rows}
        purchased_products = {r.product_id for r in purchase_rows}
        purchases_per_user = Counter(r.user_id for r in purchase_rows)
        purchasers_per_product = Counter(r.product_id for r in purchase_rows)

        user_counts_list = list(per_user_counts.values())
        product_counts_list = list(per_product_counts.values())

        return {
            "total_users": total_users,
            "total_products": total_products,
            "total_interactions": total_interactions,
            "active_users": active_users,
            "interacted_products": interacted_products,
            "type_counts": dict(type_counts),
            "user_activity": {
                "avg": statistics.mean(user_counts_list) if user_counts_list else 0,
                "median": statistics.median(user_counts_list) if user_counts_list else 0,
                "min": min(user_counts_list) if user_counts_list else 0,
                "max": max(user_counts_list) if user_counts_list else 0,
                "buckets": _bucket_counts(user_counts_list),
            },
            "product_activity": {
                "avg": statistics.mean(product_counts_list) if product_counts_list else 0,
                "median": statistics.median(product_counts_list) if product_counts_list else 0,
                "min": min(product_counts_list) if product_counts_list else 0,
                "max": max(product_counts_list) if product_counts_list else 0,
                "buckets": _product_bucket_counts(product_counts_list),
            },
            "matrix": {
                "unique_pairs": len(unique_pairs),
                "users": active_users,
                "products": interacted_products,
                "cells": matrix_cells,
                "density_pct": density_pct,
                "sparsity_pct": 100 - density_pct if matrix_cells else 100.0,
            },
            "purchases": {
                "purchasing_users": len(purchasing_users),
                "purchased_products": len(purchased_products),
                "total_purchases": len(purchase_rows),
                "avg_purchases_per_purchasing_user": (
                    statistics.mean(purchases_per_user.values()) if purchases_per_user else 0
                ),
                "avg_purchasers_per_purchased_product": (
                    statistics.mean(purchasers_per_product.values()) if purchasers_per_product else 0
                ),
            },
        }
    finally:
        db.close()


def _pct(part: int, total: int) -> str:
    return f"{(part / total * 100):.1f}%" if total else "N/A (0 total)"


def print_report(stats: dict) -> None:
    print("=" * 60)
    print("COLLABORATIVE FILTERING READINESS — DATA AUDIT (read-only)")
    print("=" * 60)

    print("\n-- Dataset Overview --")
    print(f"Total users:                {stats['total_users']}")
    print(f"Total products:             {stats['total_products']}")
    print(f"Total interactions:         {stats['total_interactions']}")
    print(f"Users with >=1 interaction: {stats['active_users']}")
    print(f"Products with >=1 interaction: {stats['interacted_products']}")

    print("\n-- Interaction Type Distribution --")
    total_i = stats["total_interactions"]
    for t in ("VIEW", "CART_ADD", "PURCHASE"):
        c = stats["type_counts"].get(t, 0)
        print(f"{t:10} {c:6}  ({_pct(c, total_i)})")

    ua = stats["user_activity"]
    print("\n-- User Activity Distribution (active users only) --")
    print(f"avg={ua['avg']}, median={ua['median']}, min={ua['min']}, max={ua['max']}")
    print(f"buckets: {ua['buckets']}")

    pa = stats["product_activity"]
    print("\n-- Product Interaction Distribution (interacted products only) --")
    print(f"avg={pa['avg']}, median={pa['median']}, min={pa['min']}, max={pa['max']}")
    print(f"buckets: {pa['buckets']}")

    m = stats["matrix"]
    print("\n-- User-Product Matrix --")
    print(f"unique user-product pairs: {m['unique_pairs']}")
    print(f"matrix dimensions: {m['users']} users x {m['products']} products = {m['cells']} cells")
    print(f"density: {m['density_pct']:.4f}%")
    print(f"sparsity: {m['sparsity_pct']:.4f}%")

    p = stats["purchases"]
    print("\n-- Purchase-Specific Stats --")
    print(f"users with >=1 purchase:     {p['purchasing_users']}")
    print(f"products with >=1 purchase:  {p['purchased_products']}")
    print(f"total PURCHASE interactions: {p['total_purchases']}")
    print(f"avg purchases per purchasing user:    {p['avg_purchases_per_purchasing_user']}")
    print(f"avg purchasers per purchased product: {p['avg_purchasers_per_purchased_product']}")
    print("=" * 60)


if __name__ == "__main__":
    print_report(analyze())
