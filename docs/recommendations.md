# Recommendations: Content-Based (Phase 1) + Collaborative Filtering & Hybrid (Phase 2)

## Endpoint

`GET /recommendations/product/{product_id}?limit=10` (existing route, unchanged)

- Auth: valid JWT (any role). `401` without one.
- `limit`: 1-50, default 10; out of range -> `422`.
- Unknown product -> `404 {"detail": "Product not found"}`.
- Response (unchanged): `{"product_id": 1, "recommendations": [{"product": {...ProductOut}, "similarity_score": 0.42}, ...]}`

`GET /recommendations/me` is now the **Hybrid** personalized endpoint (see Phase 2 below).

## Pipeline

```
Product rows (id, name, description, category, brand)
  -> ml.recommendation.content_based.build_document()   normalize + join, None -> ""
  -> TfidfVectorizer (english stop words, strip_accents, sublinear_tf)
  -> ContentIndex.similar_to(product_id, limit)         cosine = dot product of L2-normalized rows
  -> services.recommendation_service.get_similar_products()   fetch top-N ORM rows, shape response
  -> api/recommendations.py
```

| Layer | File | Responsibility |
|---|---|---|
| API | `api/recommendations.py` | auth, `limit` validation |
| Service | `services/recommendation_service.py` | DB access, 404, response shape |
| ML | `ml/recommendation/content_based.py` | text building, TF-IDF, similarity, cache (no FastAPI/SQLAlchemy imports) |

## Behavior

- The target is never returned; results are sorted by score descending; ties keep product-id order (deterministic).
- Scores are floats in `[0, 1]`. Products sharing no vocabulary with the target score `0.0` and are still returned after the related ones (this preserved the existing behavior).
- Price and image URL are not part of the text: similarity means "what the product is", not cost.
- Missing description/category/brand are treated as empty. One-product or empty catalog -> `[]`. A catalog with no usable text -> `[]`. A catalog made only of stop words (e.g. "The One") falls back to a vectorizer without stop-word removal.
- Non-English text (e.g. Arabic) is tokenized and scored the same way; English stop words are only removed for English words.
- Every product is a candidate: the `Product` model has no active/inactive flag.

## Efficiency

- One query loads only the five text columns; ORM objects are built only for the returned top-N.
- Only the target's similarity row is computed (no N x N matrix) for similar-products requests.
- The fitted TF-IDF index is cached in-process (single entry) and keyed by a hash of every `(id, document)`; any product add/edit/delete changes the key and refits automatically, so there is nothing to invalidate. Each worker process keeps its own cache.

## Known limitations / technical debt

- Still O(catalog) per request (loading + hashing all texts); at large scale, precompute and store the matrix/index (a later phase with a background job).
- Field weighting (e.g. boosting name/category over description), n-grams and synonyms are not tuned.

---

# Phase 2: Collaborative Filtering + Hybrid

## Architecture

```
GET /recommendations/me              (api/recommendations.py: auth + limit only)
  -> services.recommendation_service.get_user_recommendations()   DB access, orchestration
       -> ml.recommendation.hybrid          normalize + merge + rank
            |-> ml.recommendation.content_based   TF-IDF / cosine   (ContentIndex.score_profile)
            '-> ml.recommendation.collaborative   user-user cosine  (sparse user x product matrix)
```

Pure ML modules (`content_based`, `collaborative`, `hybrid`) import no SQLAlchemy/FastAPI; the
service loads rows, converts interactions to weights and passes plain values in. `hybrid.py`
only combines scores: it never recomputes similarity.

Other service entry points: `get_similar_products` (content-based only, unchanged),
`get_collaborative_recommendations` (CF signal only, no fallback; used by tests/future callers).

## Interaction weighting

Single source of truth: `services.recommendation_service.INTERACTION_WEIGHTS`
(`VIEW=1.0 < CART_ADD=2.0 < PURCHASE=3.0`, heuristic, not learned). Repeated interactions with the
same product are **summed** per (user, product), for both content and collaborative paths.
A purchase therefore contributes more than a view to both signals.

## Content-based (personalized)

`ContentIndex.score_profile({product_id: weight})` = `sum_i weight_i * cosine(i, candidate)`,
computed as one sparse profile vector times the TF-IDF matrix (no N x N matrix). Same math as
before; the old per-request full-matrix code was removed. Already-interacted products never score.

## Collaborative filtering (user-based, `ml/recommendation/collaborative.py`)

1. Build a sparse **user x product** matrix of summed interaction weights.
2. `find_similar_users`: cosine between the target row and all others; keep users with
   similarity > 0 (top `CF_MAX_NEIGHBORS`, default 50; ties by user id).
3. `score(p) = sum over similar users v of sim(target, v) * weight(v, p)`.
4. Products the target already interacted with are excluded; sort by score desc, ties by product id.

Candidate generation in the service: the target's own history (1 query) + the interactions of
every user sharing >= 1 product with them (1 query, single SQL statement; users with no overlap
have cosine 0 so leaving them out is exact). Query count does not grow with users/products.
Data comes only from the authenticated `user_id`; other users' identities are never returned.

## Hybrid scoring

```
norm(x)      = x / (x + saturation)                 saturation = INTERACTION_WEIGHTS[PURCHASE] = 3.0
hybrid_score = content_weight * norm(content_raw) + collaborative_weight * norm(collaborative_raw)
```

- Weights are configured **only** in `core/config.py` (`HYBRID_CONTENT_WEIGHT`, `HYBRID_COLLABORATIVE_WEIGHT`,
  both `0.5`; override via `.env`). They are normalized to sum to 1, so scores are in `[0, 1)`.
- The saturating `norm` puts both signals on one scale and is monotonic. It needs no other
  candidates, so a score means the same across requests (one purchase-backed signal scores higher
  than one view-backed signal). Per-request max-scaling was rejected: it would erase that property.
- A product proposed by both sources is merged into one candidate (weighted sum). A product proposed
  by one source is scored by that term alone (scores are not rescaled when the other source is empty).
- Already-interacted products are excluded; results sort by score desc (ties: product id), then `limit`.

## Cold start / fallbacks (`/recommendations/me`)

| Situation | Behavior | `recommendation_source` |
|---|---|---|
| User has no interactions | popular products (weighted interaction count across all users; newest products if there is no interaction data at all) | `popular` |
| Interactions, but neither signal yields a candidate | popular products, excluding already-interacted ones | `popular` |
| Only content proposes it / only CF / both | ranked hybrid list | `content` / `collaborative` / `hybrid` |
| Empty or one-product catalog, no similar users, one interaction | never crashes; may return `[]` | n/a |

## API behavior

`GET /recommendations/me?limit=10` (JWT; `limit` 1-50). The user always comes from the token; a
`user_id` query parameter is ignored. Response items keep `product` and `score` and gain three
**optional, additive** fields: `recommendation_source`, `content_score`, `collaborative_score`
(normalized `[0, 1)`; `null` for `popular`). `/recommendations/product/{id}` is unchanged.

## Limitations

- The content/collaborative scoring itself is still in-process and per request; content path is still
  O(catalog) (load + hash all texts). Precomputing the TF-IDF/CF matrices is still future work.
- CF loads all interactions of every overlapping user. A product viewed by a huge share of users makes
  that set large; no sampling or cap on loaded rows yet.
- User-based CF on sparse data is weak: with few users/overlaps CF contributes nothing and the result is
  content-only (by design, not an error). Popularity bias is not corrected.
- No time decay: an old view weighs the same as a recent one. Repeated views accumulate linearly.
- Weights, saturation constant and neighbor count are heuristics, not tuned on data; no offline evaluation yet.
- Every product is a candidate (no active/in-stock flag exists in the model).

---

# Phase 3: Personalized API + Customer Segmentation

## Personalized recommendation API

`GET /recommendations/me?limit=10` (JWT required; `limit` 1-50). Contract is backward compatible:
`product` and `score` are unchanged, the Phase 2 optional fields are preserved, and **one** optional
field was added:

| Field | Meaning |
|---|---|
| `score` | final ranking score (hybrid score; weighted popularity for `popular`) |
| `recommendation_source` | `hybrid` \| `content` \| `collaborative` \| `popular` |
| `content_score`, `collaborative_score` | normalized `[0, 1)`; `null` for `popular` |
| `reason` (new) | short human-readable explanation (below) |

A separate `hybrid_score` was deliberately not added: `score` already is the hybrid score.

**Identity.** The user comes only from the JWT (`Depends(get_current_user)`). The endpoint declares no
`user_id` parameter; any `user_id`/`userId`/`email` query string is ignored (covered by tests).
No other user's profile, id or interactions appear in a response.

## Recommendation reasons (`ml/recommendation/reasons.py`)

Pure, deterministic, derived only from the item's `recommendation_source` and the *kinds* of
interaction in the caller's own profile (VIEW / CART_ADD / PURCHASE). Kinds are listed in a fixed
order, and only kinds the user actually performed are mentioned.

| Source | Reason |
|---|---|
| `content` | `Similar to products you viewed` / `... you purchased` / `... you added to your cart` / combos, e.g. `... you viewed and purchased` |
| `collaborative` | `Popular among users with similar activity` |
| `hybrid` | both, e.g. `Similar to products you viewed and popular among users with similar activity` |
| `popular` | `Popular among shoppers` |

The content signal is a weighted similarity to *all* of the user's interacted products, so the reason
names the interaction kinds in the profile, never a single specific product.

## Personalization rules and cold start

- Already-interacted products are excluded (also in the popular fallback). `limit` is respected.
- Equal scores order by ascending product id, including the popular fallback (Phase 3 fix: it
  previously depended on database row order).
- Cold start is unchanged: no interactions -> popular products (weighted interaction count; newest
  products when there is no interaction data). Interactions but no candidates -> popular, excluding
  interacted products.
- Popular ranking is now aggregated in SQL (`GROUP BY product, type`) and joined to `products`, so
  interactions with deleted products cannot consume result slots. Missing optional product fields
  (description/category/brand/image) never fail a request.
- The segmentation layer is **not** an input to recommendations (independent by design in this phase).

## Customer segmentation (summary; details in `docs/customer_segmentation.md` section 13)

```
Database (2 aggregate SQL queries)  -> services/segmentation_analysis_service.py
  -> ml/segmentation/features.py     build_customer_features()   recency/frequency/monetary/AOV/views/cart-adds
  -> ml/segmentation/clustering.py   log1p -> StandardScaler -> KMeans (model.build_segmentation_pipeline)
  -> segments + per-segment statistics
```

- Only COMPLETED orders count (PENDING, CONFIRMED, CANCELLED are ignored), matching the existing adapter.
- `GET /segmentation/segments` (ADMIN only): read-only analysis, aggregate statistics only.
- `GET /segmentation/me` (any authenticated user): the caller's own persisted segment.

## Limitations (Phase 3)

- Reasons describe signal families, not the specific product that drove a score.
- No time decay, no diversity/re-ranking, popularity bias not corrected (unchanged from Phase 2).
- The scoring pipeline still runs in-process per request (unchanged from Phase 2/3); what Phase 4 added
  is a Redis cache *in front of* `GET /recommendations/me` (below) — a cache hit skips this pipeline
  entirely, but a miss still pays its full cost.


---

# Phase 4: Redis caching + background refresh

`GET /recommendations/me` is now cache-aside (`services/recommendation_cache_service.py`):

```
GET /recommendations/me
  -> cache.get("recommendations:user:{id}", field="limit:{limit}")   Redis hash, keyed by the JWT user id only
       hit  -> return cached payload (recommendation_service is never called)
       miss -> recommendation_service.get_user_recommendations()     unchanged Phase 2/3 pipeline
                 -> cache.set(..., ttl=RECOMMENDATION_CACHE_TTL_SECONDS)
```

- **Isolation**: the cache key is built only from the authenticated user's id (`core/cache.py`); there is
  no request parameter that can construct or address another user's key.
- **Invalidation**: any recorded interaction (including a plain VIEW) or an order status change deletes
  that one user's cache entry (`services/cache_invalidation.py`) — O(1), no effect on any other user.
  Otherwise entries expire after `RECOMMENDATION_CACHE_TTL_SECONDS` (default 300s).
- **Failure behavior**: a Redis error is treated as a cache miss; the endpoint still returns a normal
  `200` from the uncached pipeline. See `docs/churn_prediction.md` §12/§17 for the shared cache/Celery
  architecture (it isn't duplicated per feature) and `services/recommendation_cache_service.py`'s
  docstring for this feature's specifics.
- **Background refresh**: `tasks/recommendations.py`'s `recommendations.refresh_user_cache` Celery task
  recomputes and re-caches one user's recommendations; it is server-triggered only (no public endpoint).
