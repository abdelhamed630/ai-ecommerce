"""Redis cache abstraction. Every failure degrades to "no cache".

Design rules
------------
- Never raises to callers: any Redis error is logged, treated as a miss (or a
  no-op write/delete) and puts the cache into a short back-off so a dead Redis
  does not add its connect timeout to every request.
- Lazy: no connection is made at import time.
- Keys are built ONLY by the helpers below from an integer user id (always the
  authenticated user's id), so one user's key can never address another's.
- Values are JSON. Per-user recommendation payloads live in one Redis hash
  (`recommendations:user:{id}`, one field per `limit`) so a single DEL
  invalidates every variant for that user.
"""

import json
import logging
import time
from typing import Any, Optional

from core.config import settings

logger = logging.getLogger(__name__)


def _key(*parts: Any) -> str:
    return ":".join([settings.CACHE_KEY_PREFIX, *[str(p) for p in parts]])


def recommendations_key(user_id: int) -> str:
    return _key("recommendations", "user", int(user_id))


def churn_key(user_id: int) -> str:
    return _key("churn", "user", int(user_id))


CHURN_PREFIX_PATTERN_ARGS = ("churn", "user")


def churn_prefix() -> str:
    return _key(*CHURN_PREFIX_PATTERN_ARGS) + ":"


def segmentation_key(n_clusters: int) -> str:
    return _key("segmentation", "summary", "k", int(n_clusters))


def segmentation_prefix() -> str:
    return _key("segmentation") + ":"


class Cache:
    def __init__(self) -> None:
        self._client = None
        self._disabled_until = 0.0

    # -- client management -------------------------------------------------
    def set_client(self, client) -> None:
        """Inject a client (tests). Also clears any failure back-off."""
        self._client = client
        self._disabled_until = 0.0

    def reset(self) -> None:
        self._client = None
        self._disabled_until = 0.0

    def _get_client(self):
        if self._client is None:
            import redis  # imported lazily

            self._client = redis.Redis.from_url(
                settings.REDIS_URL,
                socket_connect_timeout=settings.CACHE_SOCKET_TIMEOUT_SECONDS,
                socket_timeout=settings.CACHE_SOCKET_TIMEOUT_SECONDS,
                decode_responses=True,
            )
        return self._client

    def _active(self) -> bool:
        return settings.CACHE_ENABLED and time.monotonic() >= self._disabled_until

    def _fail(self, action: str, exc: Exception) -> None:
        logger.warning("cache %s failed (%s); bypassing cache", action, type(exc).__name__)
        self._disabled_until = time.monotonic() + settings.CACHE_FAILURE_BACKOFF_SECONDS

    # -- health ------------------------------------------------------------
    def ping(self) -> bool:
        """Readiness probe: True iff Redis answers PING right now.

        Ignores the failure back-off (so a recovered Redis is reported
        immediately) and never raises. Does not change the back-off state:
        request-path behaviour stays exactly as before.
        """
        try:
            return bool(self._get_client().ping())
        except Exception as exc:
            logger.warning("redis ping failed (%s)", type(exc).__name__)
            return False

    # -- operations --------------------------------------------------------
    def get(self, key: str, field: Optional[str] = None) -> Optional[Any]:
        if not self._active():
            return None
        try:
            client = self._get_client()
            raw = client.hget(key, field) if field is not None else client.get(key)
            return None if raw is None else json.loads(raw)
        except Exception as exc:  # redis errors, bad JSON, ...
            self._fail("get", exc)
            return None

    def set(self, key: str, value: Any, ttl: int, field: Optional[str] = None) -> bool:
        if not self._active() or ttl <= 0:
            return False
        try:
            client = self._get_client()
            payload = json.dumps(value)
            if field is not None:
                pipe = client.pipeline()
                pipe.hset(key, field, payload)
                pipe.expire(key, ttl)
                pipe.execute()
            else:
                client.set(key, payload, ex=ttl)
            return True
        except Exception as exc:
            self._fail("set", exc)
            return False

    def delete(self, *keys: str) -> bool:
        if not keys or not self._active():
            return False
        try:
            self._get_client().delete(*keys)
            return True
        except Exception as exc:
            self._fail("delete", exc)
            return False

    def delete_prefix(self, prefix: str) -> bool:
        """Delete every key starting with `prefix` (SCAN, never KEYS)."""
        if not self._active():
            return False
        try:
            client = self._get_client()
            batch = list(client.scan_iter(match=f"{prefix}*", count=500))
            if batch:
                client.delete(*batch)
            return True
        except Exception as exc:
            self._fail("delete_prefix", exc)
            return False


cache = Cache()
