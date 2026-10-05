"""Per-user rate limiting for expensive endpoints (currently POST /chatbot/chat).

Why: every chat request spends quota on the LLM provider (Groq). Without a limit
one authenticated client can exhaust it with a loop.

How:
- Redis available  -> fixed one-minute window counter shared by ALL workers/containers
                      (`core.cache.Cache.incr`, atomic INCR + EXPIRE).
- Redis unavailable or cache disabled (tests, local dev) -> DEGRADED MODE: a sliding
  window kept in this process's memory (it is NOT fail-open, requests are still
  limited, but NOT distributed). With several workers each one counts separately, so
  the effective limit is up to `limit x workers`, and a restart resets the counters.
  A warning is logged (at most once a minute) whenever this happens. nginx adds a
  second, per-IP limit in front (see nginx/nginx.conf).

Redis keys: `<CACHE_KEY_PREFIX>:<ENVIRONMENT>:ratelimit:<scope>:<user id>:<minute bucket>`.
The environment segment keeps staging and production (or two stacks sharing one
Redis) from sharing counters; the user id is always the authenticated user's id.
The counter is created and expired in one MULTI/EXEC (INCR + EXPIRE, `Cache.incr`)
and its TTL is one window + 1 s, so keys never outlive their minute for long.

The limit key is the authenticated user's id (endpoints are login-only), never a
client-supplied value. Exceeding it returns 429 with a Retry-After header.
"""
import logging
import re
import threading
import time
from collections import deque
from typing import Deque, Dict, Optional

from fastapi import Depends, HTTPException, status

from core.cache import cache
from core.config import settings
from core.security import get_current_user

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 60
_FALLBACK_LOG_INTERVAL = 60.0
_UNSAFE_KEY_CHARS = re.compile(r"[^a-z0-9_.-]")
_SWEEP_THRESHOLD = 10_000  # drop idle keys once the in-process table grows this large

_lock = threading.Lock()
_hits: Dict[str, Deque[float]] = {}
_last_fallback_log = 0.0


def reset_rate_limits() -> None:
    """Forget all in-process counters (used by tests)."""
    global _last_fallback_log
    with _lock:
        _hits.clear()
        _last_fallback_log = 0.0


def _check_memory(key: str, limit: int, window: int) -> Optional[int]:
    now = time.monotonic()
    with _lock:
        if len(_hits) > _SWEEP_THRESHOLD:
            for k in [k for k, q in _hits.items() if not q or now - q[-1] >= window]:
                del _hits[k]
        queue = _hits.setdefault(key, deque())
        while queue and now - queue[0] >= window:
            queue.popleft()
        if len(queue) >= limit:
            return max(1, int(window - (now - queue[0])) + 1)
        queue.append(now)
        return None


def _environment_segment() -> str:
    return _UNSAFE_KEY_CHARS.sub("_", settings.ENVIRONMENT.strip().lower()) or "default"


def _log_degraded() -> None:
    global _last_fallback_log
    now = time.monotonic()
    with _lock:
        if _last_fallback_log and now - _last_fallback_log < _FALLBACK_LOG_INTERVAL:
            return
        _last_fallback_log = now
    logger.warning(
        "rate limit: Redis unavailable, using the per-process in-memory limiter "
        "(limit is per worker, not shared across workers)"
    )


def check_rate_limit(scope: str, identity: object, limit: int, window: int = WINDOW_SECONDS) -> Optional[int]:
    """Count one hit. Returns None when allowed, else the seconds to wait."""
    if limit <= 0:
        return None
    now = time.time()
    bucket = int(now // window)
    redis_key = f"{settings.CACHE_KEY_PREFIX}:{_environment_segment()}:ratelimit:{scope}:{identity}:{bucket}"
    count = cache.incr(redis_key, window + 1)
    if count is not None:
        return None if count <= limit else max(1, int(window - (now % window)) + 1)
    if settings.CACHE_ENABLED:  # Redis was expected but did not answer (cache disabled = intentional, no warning)
        _log_degraded()
    return _check_memory(f"{scope}:{identity}", limit, window)


def chatbot_rate_limit(current_user=Depends(get_current_user)):
    """Dependency for POST /chatbot/chat: authenticates, then rate-limits per user."""
    retry_after = check_rate_limit("chatbot", current_user.id, settings.CHATBOT_RATE_LIMIT_PER_MINUTE)
    if retry_after is not None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many chat requests. Please wait a moment and try again.",
            headers={"Retry-After": str(retry_after)},
        )
    return current_user
