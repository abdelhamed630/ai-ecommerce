# Customer Segmentation

## 1. What it does

Groups customers into a small number of behavioral segments (e.g.
"Champions", "At Risk") based on their purchase history, using RFM features
and K-Means clustering. This is offline, batch-style analysis — it is
**not** wired into any API endpoint yet (see "Future production
integration" below).

## 2. Why RFM

RFM (Recency, Frequency, Monetary) is a standard, simple, and interpretable
way to summarize a customer's purchase behavior into three numbers that
correlate strongly with future value and churn risk. It doesn't require any
product-level detail — only transaction history — which makes it a natural
fit for a first segmentation model.

## 3. How R, F, and M are calculated

Given cleaned transaction data (`InvoiceNo`, `CustomerID`, `InvoiceDate`,
`Quantity`, `UnitPrice`):

- **Recency**: `reference_date - customer's most recent invoice date`, in
  days. `reference_date` is configurable (see `features.compute_rfm`), not
  hardcoded — it defaults to one day after the latest invoice date in the
  dataset.
- **Frequency**: number of *unique* invoices (`InvoiceNo`) for the customer.
- **Monetary**: `sum(Quantity * UnitPrice)` across all of the customer's
  transactions.

## 4. Why log1p is applied to Frequency and Monetary

Retail transaction data is heavily right-skewed: a small number of
customers buy far more often and spend far more than the median customer.
Left untransformed, a distance-based algorithm like K-Means would be
dominated by these extreme values, effectively drowning out differences
among the other 95% of customers. `log1p` (i.e. `log(1 + x)`) compresses
that skew while staying defined at zero.

Recency is **not** log-transformed — it doesn't exhibit the same
extreme-outlier problem in this dataset, and the validated pipeline only
transforms Frequency/Monetary.

## 5. Why K=4, even though K=2 had a higher silhouette score

Testing K from 2 to 10 with silhouette score showed:

- K=2 produced the highest silhouette (~0.43), but only splits customers
  into a coarse "engaged vs. not" binary — not useful for targeted business
  action.
- K=4 scores only slightly lower (~0.337, still in the "reasonable
  structure" range on the commonly cited Kaufman & Rousseeuw scale) but
  produces four distinct, well-populated, interpretable segments:
  Champions/Loyal, Potential Loyalists, Low Engagement, and At Risk/Churned.

This was a deliberate human judgment call trading a small amount of
statistical "tightness" for a segmentation that's actually actionable — not
an automated "pick the highest silhouette" decision. K remains configurable
in `model.py` (`DEFAULT_N_CLUSTERS`) so a different value can be
re-evaluated later without code changes.

## 6. Why cluster IDs are not permanent segment names

K-Means assigns cluster ids (0, 1, 2, 3, ...) arbitrarily — they depend on
random initialization and can shift on every retrain, even on the same
data. Cluster "1" being "Champions" today is not guaranteed to still be
true after the next training run.

For this reason, `interpretation.py` never hardcodes a mapping like
`{1: "Champions"}`. Instead, every time a model is trained, the actual
Recency/Frequency/Monetary profile of each resulting cluster is computed
fresh, and a label is derived from that profile using simple, transparent,
median-relative rules (documented in `interpretation.py`'s module
docstring). The `{cluster_id: label}` map is regenerated per training run
and saved alongside the model (see `predict.py`), so a segment's *meaning*
stays stable even when the underlying *id* doesn't.

## 7. Segmentation vs. Recommendation

These are two separate systems that should not be mixed:

| | Recommendation | Segmentation |
|---|---|---|
| Input | `ProductInteraction` (VIEW/CART_ADD/PURCHASE) | Orders/transaction history (RFM) |
| Output | Similar/relevant *products* for a user | A behavioral *group* a customer belongs to |
| Technique | TF-IDF + cosine similarity over product text | K-Means over RFM features |
| Answers | "What should this user see next?" | "What kind of customer is this?" |

`ProductInteraction` remains the recommendation system's data source.
Segmentation is intentionally built on order/transaction data instead, so
the two systems stay decoupled.

## 8. Future production integration (not part of this task)

The current module (`ml/segmentation/`) was built and validated offline
against the UCI Online Retail dataset. Production integration will
eventually look like:

```
Completed/Paid Orders
    ↓
OrderItems
    ↓
RFM (recomputed from the app's own Orders/OrderItems tables)
    ↓
log1p(Frequency), log1p(Monetary)
    ↓
Scaling
    ↓
K-Means
    ↓
Customer Segment
    ↓
Personalization
```

RFM extraction from real Orders/OrderItems (Section 9) and persistent
storage for a calculated result (Section 10) now exist. Still not
implemented: an API endpoint, a scheduled/automatic retraining or
recalculation job, and Celery/Redis-based orchestration connecting the two.
This module is currently:

- Never imported by any API route (the `CustomerSegment` model is
  registered in `main.py` only so its table gets created — see Section 10 —
  not because any endpoint reads or writes it yet).
- Never dependent on the Online Retail Excel file at runtime — that file is
  for offline model development only.
- A standalone, testable set of functions (`preprocessing`, `features`,
  `model`, `interpretation`, `predict`) ready to be called from a future
  batch job or endpoint once that phase is scoped.

## 9. Production RFM data source: Orders/OrderItems (`ml/segmentation/adapters.py`)

A small adapter (`get_completed_order_transactions` /
`compute_rfm_from_orders` in `adapters.py`) now exists to feed this app's
*real* purchase data into the same, unmodified `compute_rfm()` function
used above — without reimplementing any RFM math and without touching
`ProductInteraction` at all.

### Field mapping

| Online Retail dataset (offline validation) | This application (production) |
|---|---|
| `CustomerID` | `User.id`, via `Order.user_id` — this app has no separate `Customer` model; the User's own id **is** the customer id used for segmentation |
| `InvoiceNo` | `Order.id` |
| `InvoiceDate` | `Order.created_at` |
| `Quantity` | `OrderItem.quantity` |
| `UnitPrice` | `OrderItem.product_price` — the **price snapshot** captured at order-creation time, never the product's current live price. This matters: if a product's price changes after an order was placed, historical Monetary figures must keep reflecting what the customer actually paid, not today's price. |

An order with multiple `OrderItem` rows becomes multiple transaction rows
sharing the same `CustomerID`/`InvoiceNo` — exactly like a multi-line
invoice in the original dataset — so `Frequency` (unique `InvoiceNo` per
customer) and `Monetary` (`sum(Quantity × UnitPrice)`) both come out of the
*existing, unmodified* `compute_rfm()` function correctly without any
special-casing in the adapter.

### Which Order status counts as a completed purchase

This project's `OrderStatus` enum (`models/order.py`) is `PENDING` /
`CONFIRMED` / `CANCELLED` / `COMPLETED`. Only **`COMPLETED`** orders are
included in RFM calculation.

`CONFIRMED` was deliberately excluded, even though it sounds like it could
qualify: the existing order-service transition rules
(`services/order_service.py`) explicitly allow `CONFIRMED -> CANCELLED`.
A `CONFIRMED` order is not yet final — counting it in RFM now and having it
cancelled later would leave the segmentation quietly wrong with no
mechanism here to retroactively correct it. `COMPLETED` is the only status
with no outgoing transition at all in the existing state machine, i.e. it's
truly final. `PENDING` and `CANCELLED` are excluded for the obvious reason
that neither represents a completed purchase.

### Why `ProductInteraction` is NOT used here

`ProductInteraction` (VIEW / CART_ADD / PURCHASE) remains exclusively the
recommendation system's data source (see Section 7 above). Customer
segmentation is intentionally built on a completely separate source of
truth — actual completed Orders/OrderItems — so a browsing/cart-adding
pattern can never influence which behavioral segment a customer is placed
in. Only money that has actually, finally changed hands does.

### What this adapter does *not* do (yet)

- It doesn't call any persistence code itself — no API endpoint, no
  scheduled job. It's a pure function you can call from a future batch
  process once that phase is scoped. (Persisting an already-calculated
  result is now possible — see Section 10 — but nothing yet calls this
  adapter and feeds its output there automatically.)
- (Superseded by §14 below for the on-demand analysis endpoint, which now caches its aggregate summary.)
It doesn't retrain or cache a model — every call re-derives the
  transaction data fresh from the current database state.
- It only reads `Order`/`OrderItem` columns (and nothing from `Product`,
  since `OrderItem` already carries the historical name/price snapshot) —
  a single joined query, no N+1 queries.

## 10. Persistent storage: the `CustomerSegment` table

`models/customer_segment.py` defines a `customer_segments` table that
stores the **latest calculated segmentation result per user**. This is the
first piece of the eventual `RFM Adapter -> Segmentation Pipeline ->
CustomerSegment Service -> CustomerSegment DB table` flow — see the
architecture diagram in Section 8. It is a storage layer only: nothing in
this project currently computes a segment and writes it here automatically
(no API endpoint, no Celery task, no scheduled job — those are later
phases).

### Why only one *current* segment is stored per user

`user_id` is unique on this table (`uq_customer_segment_user_id`), so
there is at most one row per user. This table intentionally holds the
**current** state, not a full history of every past calculation — the
same pattern this project already uses for `Cart` (one cart per user).
Recalculating a user's segment replaces their existing row in place via
`services/customer_segment_service.upsert_current_segment` rather than
inserting a new one. A history of past segments, if ever needed, would be
a separate table — this one is deliberately kept simple as "what is true
right now."

### Why `cluster_id` and `segment_label` are stored separately

As explained in Section 6, a K-Means `cluster_id` is arbitrary and can
mean something different after every retrain — cluster `1` being
"Champions" today is not guaranteed after the next training run. Storing
only `cluster_id` in the database would make old rows silently
misleading the moment the model is retrained.

`segment_label` is the human-readable interpretation produced by
`ml.segmentation.interpretation` **at calculation time**, and is stored
as its own column precisely so it stays correct even after `cluster_id`'s
meaning shifts later. The database model itself never hardcodes a
`cluster_id -> label` mapping — that mapping only ever lives, transiently,
inside a single training run (see Section 6).

### The RFM profile is stored, not just the cluster/label

`recency`, `frequency`, and `monetary` are stored alongside `cluster_id`
and `segment_label` so that a segment can be explained and debugged
without re-running the pipeline, and so that future analytics (e.g.
detecting model drift, or checking whether a "Champion" still actually
looks like one) has real numbers to work with — not just a label.

### Why `model_version` is stored

Segmentation models may change over time (a different feature set,
a different `K`, a different algorithm entirely). `model_version`
(e.g. `"rfm_kmeans_v1"`) records which configuration produced a given row,
so segments computed by different model versions are never silently
compared as if they were computed the same way. The value itself is
defined once, as `SEGMENTATION_MODEL_VERSION` in `ml/segmentation/model.py`
— the single source of truth — rather than hardcoded in the database model,
the service, or anywhere else.

### `calculated_at` vs. order dates vs. `created_at`/`updated_at`

`calculated_at` is when the segmentation pipeline actually ran — it is
**not** the customer's last order date, and it is not the same as this
row's own `created_at`/`updated_at` (which track the row's lifecycle, not
the calculation). For example, a customer's last order might be from
2026-09-20, but the segment that reflects it might not be calculated until
2026-09-27 — `calculated_at` records the latter.

### `services/customer_segment_service.py`

A thin persistence-only service sits in front of the table:

- `get_current_segment(db, user_id)` — returns the user's current
  segment, or `None` if one has never been calculated.
- `upsert_current_segment(db, user_id, cluster_id, segment_label,
  recency, frequency, monetary, model_version=..., calculated_at=...)` —
  creates the user's first segment row, or updates their existing one
  in place.

This service does **not** compute RFM, run K-Means, or derive a label —
it only persists an already-calculated result. Wiring a real caller (an
API endpoint or scheduled job that pulls RFM via `adapters.py`, runs the
existing K-Means/interpretation pipeline, and calls
`upsert_current_segment`) is a future phase.

## 11. Orchestration: `services/segmentation_service.py`

`run_customer_segmentation(db, reference_date=None)` is the application-
layer function that actually connects every piece above into one runnable
pass. It is a **bridge**, not a new implementation — every step delegates
to something that already existed and was already tested; this module's
own job is sequencing, empty/insufficient-data handling, and transaction
safety.

```
Completed Orders
    ↓
adapters.compute_rfm_from_orders   (adapter + compute_rfm — Section 9)
    ↓
features.apply_rfm_transformations (log1p Frequency/Monetary)
    ↓
model.train_segmentation_model     (K-Means, K=4, random_state=42 — Section 5)
    ↓
interpretation.get_segment_label_map (Section 6 — fresh per run, never hardcoded)
    ↓
customer_segment_service.upsert_current_segment  (Section 10, one per customer)
```

Nothing calls this function automatically yet — no API endpoint, no
Celery task, no scheduler. It's meant to be invoked explicitly (a script,
or a future endpoint/job once that phase is scoped), the same way
`adapters.py`'s functions were before this phase.

### Why `CustomerID` is never a clustering feature

`compute_rfm_from_orders` keeps `CustomerID` on every row so a cluster
assignment can be traced back to a user, and `apply_rfm_transformations`
only *adds* `Frequency_log`/`Monetary_log` — it never drops columns. But
when the model is actually trained/predicted, only
`model.FEATURE_COLUMNS` (`Recency`, `Frequency_log`, `Monetary_log`) is
selected as the feature matrix `X`. `CustomerID` is an identifier, not a
behavioral signal — including it would make K-Means cluster on an
arbitrary integer instead of actual purchase behavior.

### Why cluster ids are still not business labels here either

Exactly as in Section 6: `run_customer_segmentation` never hardcodes a
`{cluster_id: label}` map. Every run derives fresh cluster profiles from
that run's own data and passes them into
`interpretation.get_segment_label_map`, then persists whatever label comes
back. A future retrain can freely reshuffle which integer is which segment
without any code change here.

### Why insufficient data raises an error instead of reducing K

The segmentation model is configured for `K=4`
(`model.DEFAULT_N_CLUSTERS`) — that's a deliberate business decision (see
Section 5), not just "however many groups the data happens to support."
K-Means requires at least as many samples as clusters, but automatically
falling back to `K=3` or `K=2` when the database is thin would silently
produce a *different* model than the configured one, with no record that
this happened. Instead, `run_customer_segmentation` counts distinct
customers with completed orders before training and raises
`InsufficientSegmentationDataError` — a clear, catchable, explicit
failure — if there are fewer customers than `K`. `K` itself is never
modified anywhere in this module.

An empty database (zero completed orders at all) is treated differently
from *insufficient* data: it's a normal state for a fresh installation, so
it returns a zero-value summary (`{"customers_processed": 0, "segments":
{}, "model_version": ...}`) rather than raising — no CustomerSegment rows
are created and no training is attempted.

### Transaction safety

All of a run's `upsert_current_segment` calls happen inside one
transaction: `customer_segment_service.upsert_current_segment` now accepts
a `commit: bool = True` parameter (default preserves its original
single-call behavior for any other caller). The orchestration service
passes `commit=False` for every customer in the batch — each row is
flushed (so constraint violations surface immediately) but not committed —
and calls `db.commit()` exactly once after the whole batch succeeds. If
anything raises partway through, the service calls `db.rollback()` and
re-raises, so a failed run never leaves some customers updated and others
not.

### Why segmentation currently runs explicitly, not automatically

This phase deliberately stops at a callable service function. Wiring it to
run automatically — an API endpoint, a scheduled job, a Celery task
triggered by new completed orders — is out of scope here and left for a
later, explicitly-scoped phase, so that persistence, orchestration, and
automation each get reviewed as separate, independently-testable steps.

### Schema changes: no migration framework needed for this table

This project doesn't use Alembic (see
`scripts/migrate_add_category_brand.py` for how it handles the one case
that has come up so far: adding a column to an *existing* table). Adding
`customer_segments` didn't need that kind of script: `Base.metadata.create_all()`
(called from `main.py` on every startup, exactly as it already is for every
other table) creates tables that don't exist yet without touching any
existing table — it only fails to *ALTER* an existing table, which isn't
what's happening here. `customer_segments` is a brand-new table, so no
special migration step was required; running the app (or the test suite)
against an existing `ecommerce.db` creates the new table alongside the
untouched existing ones.

## 12. API trigger: `POST /segmentation/run`

`api/segmentation.py` exposes exactly one endpoint that runs everything
above on demand. It is deliberately thin — see the "Architecture rule"
subsection below — every actual computation still lives in
`services/segmentation_service.py` and the `ml/segmentation/` modules.

### Who can call it

- **Authentication**: a valid JWT is required, via the project's existing
  `core.security.get_current_user` dependency (no second authentication
  mechanism was introduced).
- **Authorization**: the caller must have the `ADMIN` role — segmentation
  execution requires `UserRole.ADMIN`. `POST /segmentation/run` is
  available only to `ADMIN` users; `SELLER` and `USER` both get `403
  Forbidden`. This project has a full `USER` / `SELLER` / `ADMIN` role
  system (see `docs/authentication.md` for the complete authorization
  architecture); `models/user.py` defines the canonical `UserRole` enum
  and the `role` column, and `core.security.get_current_admin_user`
  wraps the base `get_current_user` authentication dependency with one
  extra role check. There is intentionally no API to self-promote to
  admin — an admin account is provisioned directly at the database level
  via `scripts/set_user_role.py`, the same way this project's test suite
  does it in `tests/test_segmentation_api.py`. (An earlier version of
  this project used a simpler `User.is_admin` boolean instead of a role
  system; see `scripts/migrate_is_admin_to_role.py` if you're migrating a
  database that predates the role system.)
- A non-admin authenticated user (`USER` or `SELLER`) gets `403
  Forbidden`; a request with no
  token, or an invalid/expired one, gets `401 Unauthorized`.

### What it does

Calls `services.segmentation_service.run_customer_segmentation(db,
reference_date=...)` — the same function described in Section 11 — and
returns its summary directly:

```json
{
  "customers_processed": 8,
  "segments": {
    "Champions / Loyal": 2,
    "Potential Loyalist": 2,
    "Low Engagement": 2,
    "At Risk / Churned": 2
  },
  "model_version": "rfm_kmeans_v1"
}
```

An optional `reference_date` (`YYYY-MM-DD`) may be sent in the request
body; a malformed value is rejected with `422` by pydantic before it ever
reaches the RFM calculation. No other field is accepted — the request
schema uses `extra="forbid"`, so an attempted `n_clusters`, `random_state`,
or any other model parameter is rejected with `422` rather than silently
ignored. `K`, `random_state`, and the feature set are fixed by
`ml/segmentation/model.py` and are never client-configurable.

`InsufficientSegmentationDataError` (Section 11 — fewer customers than the
configured `K=4`) is translated to `400 Bad Request` with an explanatory
message; `K` is never silently reduced to accommodate a thin database. A
database error during persistence is translated to `500` without leaking
driver internals or a raw traceback.

### It runs synchronously, and persists the *current* result only

The request blocks until the whole pipeline (RFM → K-Means →
interpretation → persistence) finishes — there is no background job,
Celery, or Redis involved yet. As in Section 10, a successful run does not
create a history of past segments; it replaces each customer's current
`CustomerSegment` row in place, all in one transaction.

### Future: this should move off the request/response cycle

As the customer base grows, running the full pipeline synchronously
inside an HTTP request will become slow and eventually unacceptable for a
request/response cycle. The intended future shape is:

```
API -> background job -> Celery -> Redis
```

That architecture is intentionally **not** implemented in this phase —
`POST /segmentation/run` today is a synchronous, on-demand, admin-only
trigger, nothing more.

---

## 13. Phase 3: on-demand segmentation analysis + current-user lookup

Sections 1-12 (RFM adapter, persisted `CustomerSegment`, `POST /segmentation/run` with fixed K=4
and business labels) are unchanged. Phase 3 adds a second, **read-only** layer that reuses the same
building blocks (`COMPLETED_PURCHASE_STATUS`, `model.build_segmentation_pipeline`, log1p/StandardScaler/
KMeans) and adds configurable K, behavioural features and small-dataset safety.

### Architecture

```
Database -> services/segmentation_analysis_service.py   (2 SQL aggregations: orders, interactions)
         -> ml/segmentation/features.py    CustomerActivity -> CustomerFeatures (plain Python)
         -> ml/segmentation/clustering.py  transform -> StandardScaler -> KMeans -> SegmentationResult
         -> api/segmentation.py            GET /segmentation/segments (admin), GET /segmentation/me
```

`features.py` (new functions) and `clustering.py` import no FastAPI/SQLAlchemy and are unit-tested with
plain data. (`adapters.py`, which predates Phase 3, does import SQLAlchemy; see debt below.)

### Customer features

| Feature | Definition |
|---|---|
| recency | whole days from the last COMPLETED order to `reference_date` (default: now, UTC; never negative) |
| frequency | number of COMPLETED orders |
| monetary | sum of `Order.total_price` over COMPLETED orders (price snapshot at order time) |
| average_order_value | monetary / frequency (0 without orders) |
| views / cart_adds | count of `VIEW` / `CART_ADD` interaction events |

Order status rules: `COMPLETED` only. `PENDING`, `CONFIRMED` (can still be cancelled) and `CANCELLED`
orders never count. A customer with only non-completed orders is not a customer for this analysis.
Refunds are not modelled (see limitations). `PURCHASE` interactions are not used: the completed order is
the authoritative purchase record. Customers without a completed order are excluded by default;
`SEGMENTATION_INCLUDE_NON_PURCHASERS=true` includes browse-only users with frequency 0, monetary 0 and
an **imputed** recency (the largest recency among buyers, or 0 if nobody bought).

### Preprocessing and clustering

- Missing/NaN/negative counts and amounts -> 0.
- `frequency`, `monetary`, `views`, `cart_adds` get `log1p` (right-skewed); `recency` is used as is.
- `StandardScaler` on the selected features so no feature dominates by scale (zero-variance columns
  are left unscaled, no division by zero).
- KMeans (`n_init=10`, `random_state` from config). Default clustering features are
  `recency, frequency, monetary` (the validated set); views/cart_adds/AOV appear in the statistics but
  are opt-in for clustering via `SEGMENTATION_CLUSTER_FEATURES`. AOV is not clustered on (it is
  monetary/frequency, i.e. redundant).
- **Small data:** effective K = `min(requested K, number of distinct feature vectors)`. 0 customers ->
  empty result; 1 customer or identical vectors -> one segment; 2 customers -> at most 2. The response
  reports `requested_n_clusters` and the effective `n_clusters`. (Unlike `POST /segmentation/run`, which
  raises when customers < 4, this analysis degrades gracefully.)

### Configuration (`core/config.py`)

`SEGMENTATION_N_CLUSTERS` (4), `SEGMENTATION_MAX_CLUSTERS` (10, API cap), `SEGMENTATION_RANDOM_STATE` (42),
`SEGMENTATION_CLUSTER_FEATURES` (`["recency","frequency","monetary"]`; env override must be JSON),
`SEGMENTATION_INCLUDE_NON_PURCHASERS` (false).

### Segment ids and interpretation

KMeans ids are arbitrary, so ids are reassigned deterministically: ordered by average monetary
(descending), then average recency (ascending), then size (descending). **Segment 0 is the
highest-spending segment of this run.** Ids are *not* stable across runs on different data and carry no
business label. Read a segment through its statistics (all per-customer means): `customer_count`,
`average_recency` (lower = more recent), `average_frequency`, `average_monetary`, `average_order_value`,
`average_views`, `average_cart_adds`. E.g. a segment with low recency, high frequency and high monetary
is a recent, frequent, high-spend group. No "VIP"/"bad customer" labels are generated; the persisted
`POST /segmentation/run` result keeps its own documented median-relative labels.

### API and authorization

| Endpoint | Access | Behavior |
|---|---|---|
| `GET /segmentation/segments?n_clusters=&include_assignments=&limit=&offset=` | ADMIN only (`401` no token, `403` USER/SELLER) | Computes on demand, persists nothing. `n_clusters` 1..`SEGMENTATION_MAX_CLUSTERS` (else `422`). Returns segment statistics; with `include_assignments=true` also paginated `{customer_id, segment_id}` pairs. Never returns emails or other personal data. |
| `GET /segmentation/me` | any authenticated user | Returns the caller's **own** persisted segment (from `POST /segmentation/run`): `segment_id` (= stored `cluster_id`), `segment_label`, RFM, `model_version`, `calculated_at`; `segmented:false` with null fields if never segmented. The user comes only from the JWT; `user_id` parameters are ignored. |

`/segmentation/me` deliberately reads the persisted row instead of re-clustering all customers per
request. Its `segment_id` therefore belongs to the persisted model (K=4, labelled), not to the ad-hoc
`GET /segmentation/segments` analysis; ids from the two are not comparable.

### Limitations

- `GET /segmentation/segments` loads aggregates for all customers and clusters synchronously per request
  A Redis cache now sits in front of it for the common case (see §14); an admin `POST /segmentation/refresh`
  Celery task also exists to run the persisted pipeline asynchronously.
- Two segmentation results coexist (persisted K=4 labelled vs on-demand configurable); unifying them
  (e.g. persisting the configurable run) is future work. `POST /segmentation/run` still uses
  `model.DEFAULT_N_CLUSTERS`, not the new config.
- Refunded payments are not subtracted: an order that is COMPLETED still counts even if its payment was
  REFUNDED (the project does not link order status to payment status).
- Recency is relative to "now"; there is no time-windowing of orders. Imputed recency for non-purchasers
  is a heuristic. `ml/segmentation/adapters.py` (pre-existing) imports SQLAlchemy inside the ML package.


---

## 14. Phase 4: Redis caching + Celery refresh

`GET /segmentation/segments` (the read-only, on-demand analysis from §13) caches only its **aggregate
summary** (never per-customer assignments) at `segmentation:summary:k:{n_clusters}`
(`SEGMENTATION_CACHE_TTL_SECONDS`, default 300s); `include_assignments=true` always recomputes, so
assignment pairs are never served stale from cache. A Redis failure falls straight through to a normal
on-demand computation (`200`, not `500`).

`POST /segmentation/refresh` (ADMIN) queues the `segmentation.refresh` Celery task
(`tasks/segmentation.py`), which calls the **same** `services.segmentation_service.run_customer_segmentation`
used by `POST /segmentation/run` in §12 — the clustering algorithm is not duplicated — and then evicts all
cached `GET /segmentation/segments` summaries. `503` (nothing queued) if the broker is unreachable.

**Documented limitation carried over from §13**: the persisted, labelled K=4 model (`POST /segmentation/run`
/ `segmentation.refresh`) and the configurable on-demand analysis (`GET /segmentation/segments`) remain two
separate paths with incomparable segment ids; Phase 4 did not unify them, only added caching/async triggering
on top of each as it already existed.

See `docs/churn_prediction.md` §12/§13/§17 for the shared Redis/Celery architecture and failure-handling
rules (not repeated per feature here).
