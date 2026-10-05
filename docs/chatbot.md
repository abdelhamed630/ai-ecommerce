# Shopping chatbot (backend)

Flow of `POST /chatbot/chat` and `POST /chatbot/chat/stream`:

    JWT auth -> per-user rate limit (429) -> intent -> product search OR recommendations
      -> compact context (max 10 products, bounded fields) -> LLM (Groq) -> response

The rate limiter runs before anything is looked up or sent to the LLM.

## Endpoints

- `POST /chatbot/chat` -> `{"answer": str, "products": [...]}` (unchanged shape).
- `POST /chatbot/chat/stream` -> `text/event-stream`. Events: `products` (real DB rows,
  first), `token` (`{"text"}` pieces), then `done`, or `error` (`{"message"}`: a fixed,
  localized text, never provider details). Same body, auth and rate limit as `/chat`
  (one shared counter). The database is read before the stream starts. Backend only.

Request body: `message` (1-2000 chars) and optional `history`: `[{"role": "user"|"assistant",
"content": 1-2000 chars}]`, at most 50 items; roles other than user/assistant are a 422.
Only the last `CHATBOT_MAX_HISTORY_TURNS` (default 6, 0 = off) are sent to the LLM, each cut
to 600 chars. The server keeps no conversation state.

## Rate limiting

`CHATBOT_RATE_LIMIT_PER_MINUTE` (default 20, `0` = off, valid range 0-10000, anything else
fails at startup). Key = authenticated user id, never client input.

- Redis reachable: fixed one-minute window, `INCR`+`EXPIRE` in one `MULTI/EXEC`, key
  `<CACHE_KEY_PREFIX>:<ENVIRONMENT>:ratelimit:chatbot:<user_id>:<minute>`, TTL 61 s. Shared by
  all workers/containers.
- Redis down or cache disabled: **degraded mode = per-process in-memory sliding window**
  (not fail-open, but NOT distributed: the real limit is up to `limit x workers`, and it
  resets on restart). A warning is logged at most once a minute.

nginx adds a per-IP limit on `/chatbot/` (60 r/min, burst 20). Keep it above the per-user limit.

## Product filtering

`GET /products/?q=&category=&min_price=&max_price=&skip=&limit=` - all optional, combinable,
applied in SQL, ordered by id. `limit` default 100, max 200 (above = 422), `skip >= 0`.
Prices must be finite, 0..1e12; `min_price > max_price` is a 422. `q`: every word must occur
(case-insensitive, Arabic spelling variants folded) in name/description/category/brand.
`category`: exact, case-insensitive. Index: `ix_products_price` (migration 0003). No index can
serve `%term%` ILIKE; `pg_trgm` was deliberately not added.

## Arabic normalization

`core/text_normalization.py`: `أإآٱ->ا`, `ى->ي`, `ة->ه` (Arabic only; English untouched), used by the
chatbot search and the product filter, in Python and in SQL (`sql_fold`). Matching only: stored
text is never rewritten. Diacritics in *stored* text are not stripped in SQL.

## Verify locally

    python -m pytest
    docker compose run --rm nginx nginx -t        # or: docker run --rm -v "$PWD/nginx/nginx.conf:/etc/nginx/nginx.conf:ro" nginx:1.27-alpine nginx -t
    alembic upgrade head                          # applies 0003 (additive index)
