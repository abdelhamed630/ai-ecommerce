# Frontend API Contract

Derived from the actual routers/schemas in the repository (`api/auth.py`, `api/products.py`,
`api/chatbot.py`, `schemas/*`, `core/security.py`, `core/rate_limit.py`, `main.py`, `nginx/nginx.conf`).
Nothing here is invented; anything not verifiable from code is marked **(not verified at runtime)**.

- **Base URL (Docker, through Nginx):** `http://localhost:8080`
- **Interactive docs:** `http://localhost:8080/docs` (only when `ENABLE_API_DOCS=true`)
- **Format:** JSON, except `POST /auth/login` (form-encoded) and `POST /chatbot/chat/stream` (SSE response).
- **Auth header:** `Authorization: Bearer <access_token>`

## 1. Conventions

### Errors
Application errors use `{"detail": "<string>"}`.
Request-validation errors (HTTP 422) use FastAPI's default shape:

```json
{"detail": [{"type": "string_too_short", "loc": ["body", "message"], "msg": "String should have at least 1 character", "input": "", "ctx": {"min_length": 1}}]}
```

Some 422s raised manually (product price filters) use a plain string `detail`.
So the frontend must handle `detail` being **either a string or an array**.
Unhandled server errors return `500 {"detail": "Internal server error"}`.

### Errors that do NOT come from the API (not JSON)
- **Nginx 429** on `/chatbot/*` (per-IP limit, 60 req/min, burst 20) returns Nginx's own **HTML** body, not `{"detail": ...}`.
- **Nginx 413** (body too large: 3 MB globally, 256 KB on `/chatbot/`) and **502/504** are also HTML.
- **Wrong Host header** is rejected by the app (TrustedHostMiddleware) with a plain-text 400.
  Always handle non-JSON error bodies defensively (check `Content-Type` before `response.json()`).

### Roles
`USER`, `SELLER`, `ADMIN` (`role` field on the user object). Public registration always creates `USER`.

---

## 2. Auth

### POST `/auth/register`
- Auth: none. Headers: `Content-Type: application/json`

Body:

| field | type | required | notes |
|---|---|---|---|
| `email` | string (valid email) | yes | |
| `full_name` | string \| null | no | |
| `password` | string | yes | no length/strength rule is enforced by the server |
| `confirm_password` | string | yes | must equal `password` |

Response `200` (`UserOut`): `id` (int), `email`, `full_name`, `is_active` (bool), `role` (`"USER"`), `created_at` (ISO datetime).

| status | when |
|---|---|
| 200 | created |
| 400 | `{"detail": "Email already registered"}` |
| 422 | invalid email, missing field, or passwords differ (`msg` is `"Value error, Passwords do not match"`, `loc` is `["body"]`) |

```bash
curl -X POST http://localhost:8080/auth/register -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","full_name":"Test User","password":"<password>","confirm_password":"<password>"}'
```
```json
{"email":"user@example.com","full_name":"Test User","id":1,"is_active":true,"role":"USER","created_at":"2026-10-04T12:00:00Z"}
```
(Response values illustrative; field set is from `UserOut`.)

### POST `/auth/login`
- Auth: none. **Content-Type: `application/x-www-form-urlencoded`** (OAuth2 password form, NOT JSON).
- Form fields: `username` (**the user's email**), `password`.

Response `200`: `{"access_token": "<jwt>", "token_type": "bearer"}`

| status | when |
|---|---|
| 200 | ok |
| 401 | `{"detail": "Incorrect email or password"}` (+ `WWW-Authenticate: Bearer`) |
| 422 | missing `username` / `password` |

```bash
curl -X POST http://localhost:8080/auth/login -H "Content-Type: application/x-www-form-urlencoded" \
  --data-urlencode "username=user@example.com" --data-urlencode "password=<password>"
```
```js
const body = new URLSearchParams({ username: email, password });
const res = await fetch(`${BASE}/auth/login`, { method: "POST", body }); // header is set automatically
```

### GET `/auth/me`
- Auth: **required**. Response `200`: same `UserOut` as register.

| status | when |
|---|---|
| 401 | missing/invalid/expired token: `{"detail": "Could not validate credentials"}` |
| 403 | `{"detail": "Inactive user"}` (account deactivated) |

---

## 3. Products

`ProductOut`: `id` (int), `name` (string), `description` (string\|null), `price` (number), `stock` (int),
`image_url` (string\|null), `category` (string\|null), `brand` (string\|null), `seller_id` (int\|null), `created_at` (ISO datetime).

### GET `/products/`
- Auth: **none required** (public browsing; a token is accepted but not needed).
- Response `200`: array of `ProductOut`, ordered by `id` ascending (stable paging).

| query param | type | default | rules |
|---|---|---|---|
| `skip` | int | 0 | `>= 0` |
| `limit` | int | 100 | `0..200` |
| `q` | string | - | max 100 chars; every word must match name/description/category/brand; case-insensitive, Arabic spelling variants folded |
| `category` | string | - | max 100 chars; exact, case-insensitive match |
| `min_price` | number | - | inclusive; finite, `0..1e12` |
| `max_price` | number | - | inclusive; finite, `0..1e12`; `min_price` must be `<= max_price` |

| status | when |
|---|---|
| 200 | ok (empty array when nothing matches) |
| 422 | out-of-range `skip`/`limit`, `q`/`category` too long, bad price (`{"detail": "min_price must not be greater than max_price"}`, `{"detail": "min_price must be a finite number between 0 and 1000000000000"}`) |

```bash
curl "http://localhost:8080/products/?q=laptop&category=electronics&min_price=100&max_price=900&limit=20&skip=0"
```
No total count is returned. To paginate, request `limit` items and stop when fewer than `limit` come back.

### POST `/products/`
- Auth: **required, role SELLER or ADMIN** (USER gets 403). `Content-Type: application/json`.
- Body (`ProductCreate`):

| field | type | required | rules |
|---|---|---|---|
| `name` | string | yes | 2-200 chars |
| `description` | string \| null | no | |
| `price` | number | yes | `> 0` |
| `stock` | int | yes | `>= 0` |
| `image_url` | string \| null | no | |
| `category` | string \| null | no | |
| `brand` | string \| null | no | |

`seller_id` cannot be sent; the server sets it (SELLER: own id, ADMIN: `null`).
Response `200`: `ProductOut`. Statuses: 401 (no/invalid token), 403 (`"You do not have permission to perform this action."`), 422.

```bash
curl -X POST http://localhost:8080/products/ -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"name":"Laptop X","description":"14 inch","price":799.99,"stock":10,"category":"electronics","brand":"Acme"}'
```

(Outside the requested scope but present in the code: `GET /products/{product_id}` public, 404 `"Product not found"`;
`PATCH` and `DELETE /products/{product_id}` for SELLER (own products only, else 403 `"You may only modify your own products."`) / ADMIN.)

---

## 4. Chatbot

Both endpoints: **auth required**, JSON body, **shared per-user rate limit** (`CHATBOT_RATE_LIMIT_PER_MINUTE`, default 20, `0` = off).

Body (`ChatMessageRequest`):

| field | type | rules |
|---|---|---|
| `message` | string | required, 1-2000 chars |
| `history` | array of `{role, content}` | optional, max 50 items, oldest first; `role` is `"user"` or `"assistant"`; `content` 1-2000 chars. The server is stateless: the client keeps and resends the conversation, and the server uses only the last `CHATBOT_MAX_HISTORY_TURNS` (default 6) items |

### POST `/chatbot/chat`
Response `200` (`ChatResponse`): `{"answer": string, "products": [ChatProduct]}`.
`ChatProduct`: `id`, `name`, `description`, `price`, `stock`, `image_url`, `category`, `brand` (no `seller_id`, no `created_at`).
If the LLM is unavailable (e.g. no `GROQ_API_KEY`), the response is still `200` with a fixed "temporarily unavailable" answer (Arabic or English by message language); products are still returned.

| status | when |
|---|---|
| 200 | ok |
| 401 / 403 | no/invalid/expired token / inactive user |
| 422 | empty or >2000-char message, bad `history` |
| 429 | rate limited: `{"detail": "Too many chat requests. Please wait a moment and try again."}` + `Retry-After` header (seconds). Nginx's own 429 (HTML) can also occur |

```bash
curl -X POST http://localhost:8080/chatbot/chat -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"message":"I need a laptop under 900","history":[]}'
```

### POST `/chatbot/chat/stream` (Server-Sent Events)
Same body, auth and statuses as `/chatbot/chat`. The 401/403/422/429 checks happen **before** the stream starts, so they arrive as normal JSON error responses.
Success: `200`, `Content-Type: text/event-stream`. Frames are `event: <name>\ndata: <json>\n\n`:

| event | data | notes |
|---|---|---|
| `products` | `{"products": [ChatProduct, ...]}` | always first |
| `token` | `{"text": "..."}` | zero or more, concatenate in order |
| `done` | `{}` | finished normally |
| `error` | `{"message": "..."}` | LLM failed; localized fixed text; stream ends |

The stream always ends with `done` or `error`.
**Do not use `EventSource`**: it cannot send POST bodies or an `Authorization` header. Use `fetch` and read `response.body`:

```js
const res = await fetch(`${BASE}/chatbot/chat/stream`, {
  method: "POST",
  headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
  body: JSON.stringify({ message, history }),
  signal: abortController.signal,
});
if (!res.ok) { /* handle 401/403/422/429 first (body may be JSON or HTML) */ }
const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
let buf = "";
for (;;) {
  const { value, done } = await reader.read();
  if (done) break;
  buf += value;
  let i;
  while ((i = buf.indexOf("\n\n")) >= 0) {
    const frame = buf.slice(0, i); buf = buf.slice(i + 2);
    const event = /^event: (.*)$/m.exec(frame)?.[1];
    const data = JSON.parse(/^data: (.*)$/m.exec(frame)?.[1] ?? "{}");
    // handle event: "products" | "token" | "done" | "error"
  }
}
```
Aborting the request closes the stream server-side. Nginx is configured with `proxy_buffering off` for `/chatbot/`, read timeout 120 s.

---

## 5. Frontend behavior rules

- **Token storage/expiry:** the JWT lifetime is `ACCESS_TOKEN_EXPIRE_MINUTES` (60 by default; configurable through `ACCESS_TOKEN_EXPIRE_MINUTES` in `.env.docker`, which compose passes to the api container). There is **no refresh-token endpoint**. On any `401` from an authenticated call: discard the token and send the user to login.
- **Startup:** if a token is stored, call `GET /auth/me`; 200 = signed in, 401 = signed out.
- **Loading states:** show a spinner for every request; for `/chatbot/chat/stream` show the typing state until the first `token`/`products` event. LLM calls can take several seconds.
- **429:** read `Retry-After` when present (JSON case) and disable the send button for that long; if the body is HTML (Nginx), use a default wait (about 10-60 s).
- **Validation errors:** map `detail[].loc` (last element) to the form field for arrays; show the string for string details.
- **Chat history:** keep it in the client; send at most the recent turns (server trims anyway).

## 6. CORS (important)

CORS is enforced by the FastAPI app, not Nginx. Allowed origins come from `CORS_ALLOWED_ORIGINS` (comma-separated, no `*` in production). Methods and headers are allowed for all.
**In the current `.env.docker` the value is empty, which means no browser origin is allowed.** A frontend served from another origin (for example `http://localhost:3000` or `http://localhost:5173`) will be blocked by the browser until its origin is added to `CORS_ALLOWED_ORIGINS` in `.env.docker` and the `api` container is recreated. Serving the frontend from the same origin (behind the same Nginx) avoids CORS entirely.
`ALLOWED_HOSTS` must also contain the hostname the frontend uses to reach the API.

## 7. Health
`GET /health` and `GET /health/live`: `{"status":"ok"}`. `GET /health/ready`: `200 {"status":"ready","checks":{"database":"ok","redis":"ok"}}`, or `503 {"status":"not_ready",...}`.
