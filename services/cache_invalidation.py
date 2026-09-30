"""Targeted cache invalidation. Best effort: never raises (see core.cache).

- User behaviour changed (any interaction, order completed): delete THAT
  user's recommendation hash and churn key (2 keys, O(1)); other users'
  caches are untouched. Views are included on purpose: a viewed product must
  stop being recommended immediately, and a single-key DEL is cheap.
- Churn model retrained: drop all cached churn predictions (done in
  churn_service.train_and_persist).
- Segmentation refreshed: drop cached segmentation summaries.
Segmentation summaries are otherwise TTL-only (an order does not evict them).
"""

from core.cache import cache, churn_key, recommendations_key, segmentation_prefix


def invalidate_user_caches(user_id: int) -> None:
    cache.delete(recommendations_key(user_id), churn_key(user_id))


def invalidate_segmentation_caches() -> None:
    cache.delete_prefix(segmentation_prefix())
