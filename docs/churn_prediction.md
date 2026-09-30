# Customer Churn Prediction (Phase 4)

## 1. Churn definition

For a snapshot taken at cutoff `T`, with window `W = CHURN_INACTIVITY_DAYS`
(config, default 60 days):

> A customer (who has made at least one COMPLETED order at or before `T`) is
> **churned** if they make **no COMPLETED order** in `(T, T + W]`.

Only customers with `>= 1` completed order at the cutoff are in the
population — churn is defined relative to having been a customer at all. A
browse-only visitor is not "churned", they simply haven't purchased yet
(see `ml/churn/features.is_eligible`). This mirrors the existing
`COMPLETED`-only rule the segmentation adapter already uses
(`ml/segmentation/adapters.COMPLETED_PURCHASE_STATUS`) — PENDING, CONFIRMED
and CANCELLED orders never count toward churn either.

`CHURN_INACTIVITY_DAYS` is the only place this window is defined; nothing
hardcodes 60 (or any other number) elsewhere.

## 2. Training window and historical snapshots

A single cutoff wastes most of the history and yields one label per customer.
Instead, training uses **multiple historical snapshots**, spaced `W` days
apart, oldest first:

```
cutoffs = [as_of - W, as_of - 2W, ..., as_of - N*W]     (N = CHURN_SNAPSHOT_COUNT, default 3)
```

Every snapshot's label window `(cutoff, cutoff + W]` ends at or before
`as_of` (normally "now"), so **no label ever depends on data from the
future relative to `as_of`.** Each snapshot contributes its own rows to the
training set (`ml/churn/dataset.build_dataset`), so 3 snapshots over 80
customers can yield up to 240 labelled rows from one training run.

## 3. Features (`ml/churn/features.py`)

All features are computed **only from events with `timestamp <= cutoff`**:

| Feature | Definition |
|---|---|
| `recency_days` | days since the last completed order, at the cutoff |
| `frequency` | completed orders so far |
| `monetary` | total spent on completed orders so far |
| `average_order_value` | `monetary / frequency` |
| `views_count` | VIEW interactions so far |
| `cart_add_count` | CART_ADD interactions so far |
| `days_since_last_activity` | days since the latest order OR interaction |
| `avg_days_between_orders` | `(last - first order) / (orders - 1)`; NaN with < 2 orders |
| `customer_lifetime_days` | days since the first order OR interaction |
| `recent_activity_count` | orders + interactions in the last `CHURN_RECENT_ACTIVITY_DAYS` days |

`PURCHASE` interaction events are **not** used as a feature — the completed
order record is the authoritative purchase source and counting both would
just duplicate `frequency`.

Reused from earlier phases: the `COMPLETED`-only purchase rule
(`ml/segmentation/adapters.py`), and the general recency/frequency/monetary
vocabulary from `ml/segmentation/features.py` — the churn feature set is a
distinct, purpose-built module (`ml/churn/features.py`) rather than importing
the segmentation one, because churn needs event-level timestamps (for the
cutoff filter and the extra time-based features above) that the
customer-level segmentation aggregates don't carry.

## 4. Label generation

```
label = 0 (retained) if the customer has a COMPLETED order in (cutoff, cutoff + W]
label = 1 (churned)  otherwise
```

Implemented in `ml/churn/dataset.label_churn`. The set of "purchasers in the
window" (`services/churn_service.load_purchasers_in_window`) is used **only**
to derive labels — it is never merged into the feature aggregate, so it can
never leak into `X`.

## 5. Leakage prevention

Three independent guards:

1. **Query-level**: `services/churn_service.load_aggregates` filters every
   order/interaction row by `timestamp <= cutoff` in SQL, for both training
   snapshots and live inference (`cutoff = now`). The *same* function feeds
   both paths, so training and serving features cannot drift apart.
2. **Feature-level**: `ml/churn/features.build_feature_row` re-checks every
   timestamp in the aggregate against the cutoff and **raises** if anything
   is after it — a defensive check independent of the query filter above.
3. **Label-level**: the label source (`load_purchasers_in_window`, events in
   `(cutoff, cutoff + W]`) is a completely separate query result from the
   feature source (events `<= cutoff`) and is never passed into
   `build_feature_row`.

`tests/test_churn_ml.py::test_future_data_in_aggregate_is_rejected` and
`tests/test_churn_ml.py::test_labels_do_not_influence_features` assert this
directly; `tests/test_churn_service_api.py::test_future_events_never_change_features_but_do_change_labels`
verifies it end-to-end against the database.

**Known look-ahead (documented, not fixed in this phase):** order *status* is
read as of "now", not as of the cutoff, because the project stores no order
status history. An order created before a cutoff but cancelled afterward is
excluded from that historical snapshot even though it looked COMPLETED at the
time. This is a narrow, explicit limitation (see §18), not a leakage bug: it
can only ever make training slightly more conservative (fewer counted
purchases), never leak a future outcome into a feature.

## 6. Model (`ml/churn/model.py`)

`sklearn.pipeline.Pipeline`:

```
ColumnTransformer
  - skewed columns  (frequency, monetary, average_order_value, views_count,
                      cart_add_count, recent_activity_count):
        log1p -> median impute (keep_empty_features) -> StandardScaler
  - other columns    (recency_days, days_since_last_activity,
                      avg_days_between_orders, customer_lifetime_days):
        median impute (+missing indicator) -> StandardScaler
  -> LogisticRegression(C=CHURN_LOGREG_C, class_weight=CHURN_CLASS_WEIGHT)
```

Logistic Regression, as specified — explainable (signed coefficients) and a
reasonable baseline; no other model is introduced. The **entire** pipeline
(preprocessing + classifier) is one fitted object: it is persisted and reused
unchanged for inference. Preprocessing is **never** refit at prediction time.

Missing values (`NaN`, e.g. `avg_days_between_orders` for a one-order
customer) are median-imputed inside the fitted pipeline; a missing-indicator
column is added for the non-skewed features so "was this missing" is itself
informative to the model. Class imbalance (churners are typically the
minority class) is handled by `class_weight="balanced"` (configurable;
`None` disables it — note that with it enabled the output is a *ranking*
probability, not a calibrated frequency; see §8).

## 7. Evaluation split and metrics

- **>= 2 snapshots with data → temporal split**: validation = the *newest*
  snapshot only, training = all older snapshots. Every training label window
  ends at or before the validation cutoff, so validation never overlaps
  training in time.
- **1 snapshot → stratified random split** (`CHURN_VALIDATION_FRACTION`,
  default 25%), since there's no second time period to hold out.

Metrics (`ml/churn/model.evaluate`, decision threshold 0.5 for the
classification metrics only — the *probability* itself doesn't use a
threshold): `accuracy`, `precision`, `recall`, `f1`, `roc_auc`,
`majority_class_accuracy` (context for `accuracy` — the score of always
predicting the majority class), `validation_churn_rate`. If the validation
set has a single class, `roc_auc` is `None` with an explanatory
`roc_auc_note` — never a fabricated number.

**Insufficient data never trains a model.** `CHURN_MIN_SAMPLES` (default 50
total) and `CHURN_MIN_CLASS_SAMPLES` (default 5 per class) are checked before
any split; a training split that ends up single-class also fails safely.
Either raises `InsufficientChurnDataError` — the training task reports
`{"status": "insufficient_data", "reason": "..."}` and **persists nothing**;
no metrics are invented for a model that wasn't trained.

## 8. Probability interpretation

`churn_probability = P(no completed order in the next CHURN_INACTIVITY_DAYS
days)`, from `pipeline.predict_proba(...)[class == 1]`, clipped to `[0, 1]`.
This is the primary output of `GET /churn/me` — never a bare `true`/`false`.
With `CHURN_CLASS_WEIGHT="balanced"` (the default), the value is a *ranking*
signal (higher = relatively more likely to churn) rather than a calibrated
real-world frequency; set `CHURN_CLASS_WEIGHT=null` for calibrated
probabilities at the cost of the model favoring the majority class more.

## 9. Risk thresholds

Business bands derived from the probability, **not** presented as ground
truth:

```
LOW:    probability <  CHURN_RISK_MEDIUM_THRESHOLD   (default 0.4)
MEDIUM: CHURN_RISK_MEDIUM_THRESHOLD <= probability < CHURN_RISK_HIGH_THRESHOLD
HIGH:   probability >= CHURN_RISK_HIGH_THRESHOLD      (default 0.7)
```

Both thresholds live only in `core/config.py`; `ml/churn/predict.risk_level`
validates `0 <= medium < high <= 1` and raises otherwise.

## 10. Model persistence (`ml/churn/predict.py`)

One `joblib` file at `CHURN_MODEL_PATH` (default
`ml/artifacts/churn/churn_model.joblib`), written atomically (temp file +
`os.replace`, so a crash mid-write never leaves a corrupt artifact in place).
Contains: the fitted pipeline, `feature_names`, `model_version`
(`churn_logreg_v1-YYYYmmddHHMMSS`), `trained_at`, the full `TrainingConfig`
used, the evaluation `metrics`, and run info (`as_of`, snapshot cutoffs,
split strategy, sample counts, churn rate). **No database objects are ever
stored in the artifact.** A `metadata.json` sidecar mirrors everything except
the pipeline itself, so `GET /churn/model` can read metadata without
unpickling the model. Loading validates the artifact's `feature_names`
against the current `FEATURE_NAMES` and refuses a mismatched/corrupt artifact
(`ChurnArtifactError`) rather than silently mis-scoring. The in-process
service cache (`services/churn_service.get_artifact`) reloads only when the
file's mtime/size change, so a request never re-reads the file when nothing
changed, and a retrain immediately takes effect for the next request.

Per the phase's scope, this is filesystem-only — no new database table.
`CustomerSegment` (Phase 3) was reused where it already existed for
segmentation; churn is a different artifact class (a fitted model with a
version and metrics, not a per-customer row) and doesn't fit that table's
shape, so no attempt was made to force it in.

## 11. Prediction API

| Endpoint | Access | Behavior |
|---|---|---|
| `GET /churn/me` | any authenticated user | The caller's **own** `churn_probability`, `risk_level`, `eligible`, `reason` (`"no_completed_orders"` when not eligible), plus `model_version`/`trained_at`/`features_as_of`/`risk_thresholds`. User comes only from the JWT — no `user_id` parameter exists, so `GET /churn/me?user_id=<other>` cannot address another customer. `503` if no model has been trained yet, or the artifact is corrupt/incompatible — never a fabricated prediction. |
| `GET /churn/model` | ADMIN | Current model's metadata + evaluation metrics (no per-customer data). `404` if nothing has been trained. |
| `POST /churn/train` | ADMIN | Queues the `churn.train_model` Celery task; `202` + `task_id` (poll `GET /tasks/{task_id}`). Never trains inside the request. `503` if the task queue is unreachable — nothing is queued in that case. |

## 12. Redis architecture

`core/cache.py` wraps a lazily-constructed `redis.Redis` client
(`REDIS_URL`). No connection is opened at import time. Every operation is
wrapped in `try/except`: **any** Redis error is logged, treated as a miss (or
a no-op write/delete), and puts the cache into a short back-off
(`CACHE_FAILURE_BACKOFF_SECONDS`) so a dead Redis doesn't add its own connect
timeout to every subsequent request. `CACHE_ENABLED=false` bypasses Redis
entirely. Keys are built only by named helpers (`recommendations_key`,
`churn_key`, `segmentation_key`) from a plain integer id — there is no path
by which client-supplied input can construct or address another key.

## 13. Celery architecture

`celery_app.py` builds a module-level `Celery` app (`broker`/`backend` from
`CELERY_BROKER_URL`/`CELERY_RESULT_BACKEND`) but touches no network at import
time — building the app object doesn't connect. **`main.py`/FastAPI never
imports `celery_app`**, so starting the API process never starts (or
duplicates) a worker; `services/task_dispatch.py` imports it lazily, only
when a task is actually submitted. Run a worker separately with:

```
celery -A celery_app.celery_app worker --loglevel=info
```

`CELERY_TASK_ALWAYS_EAGER=true` (a config flag, not used in normal
development/production) runs tasks synchronously in-process — this is how
the test suite exercises real Celery task registration/execution without a
broker or worker.

## 14. Background tasks (`tasks/`)

| Task | Name | Triggered by | Does |
|---|---|---|---|
| `tasks.churn.train_churn_model` | `churn.train_model` | `POST /churn/train` (admin) | Builds the historical dataset, trains, evaluates, persists the artifact (`services.churn_service.train_and_persist`). `{"status": "insufficient_data", ...}` is a normal, non-failure result; unexpected errors propagate so Celery marks the task FAILURE. |
| `tasks.segmentation.refresh_customer_segmentation` | `segmentation.refresh` | `POST /segmentation/refresh` (admin) | Runs the **existing** `services.segmentation_service.run_customer_segmentation` — the algorithm is never duplicated — then evicts cached segmentation summaries. |
| `tasks.recommendations.refresh_user_recommendation_cache` | `recommendations.refresh_user_cache` | server-side only (not exposed over HTTP; `user_id` is a trusted argument, never an HTTP parameter) | Recomputes and re-caches one user's personalized recommendations. |

Simple CRUD is intentionally **not** moved into Celery — only the genuinely
expensive ML/aggregation operations above.

`services/task_dispatch.py` is the one place the API submits/queries tasks:
`submit()` imports the Celery app lazily and calls `apply_async` on the
already-registered task (not `send_task`, so eager mode works in tests);
`get_status()` wraps `AsyncResult`. Both translate any broker/backend error
into `TaskQueueUnavailableError`, mapped to `503` by the API — **the task is
never reported as queued/running when it wasn't** (see §17).

## 15. Cache strategy

| Data | Key | TTL setting | Notes |
|---|---|---|---|
| Personalized recommendations | `recommendations:user:{id}` (one Redis **hash**, one field per `limit`) | `RECOMMENDATION_CACHE_TTL_SECONDS` (300s) | One `DEL` invalidates every `limit` variant for that user. |
| Churn prediction | `churn:user:{id}` | `CHURN_CACHE_TTL_SECONDS` (600s) | An "unavailable model" result is never cached (only successful predictions are). |
| Segmentation summary | `segmentation:summary:k:{n_clusters}` | `SEGMENTATION_CACHE_TTL_SECONDS` (300s) | Only the aggregate summary is cached, **never** per-customer assignments (`include_assignments=true` always recomputes). |

Every key is built from either the authenticated user's id or, for
segmentation, a plain integer `n_clusters` — never from unauthenticated
request data.

## 16. Cache invalidation

`services/cache_invalidation.py`:

- **`invalidate_user_caches(user_id)`** — deletes that one user's
  recommendation hash and churn key (2 `DEL`s, O(1), no scan). Called on:
  - any recorded product interaction, **including views**
    (`services/interaction_service.record_interaction`) — a viewed product
    should stop being "recommended" immediately, and a single-key delete is
    cheap enough to not reserve for cart-adds/purchases only.
  - an order-status change (`services/order_service.py`, e.g. an order
    completing) — the RFM-relevant behaviour just changed.
- **`invalidate_segmentation_caches()`** — prefix-deletes every cached
  `segmentation:summary:*` entry. Called after a successful
  `churn.train_model` run's own cache (drops all cached predictions from the
  now-superseded model, since old predictions came from a model that no
  longer represents the current state) and after `segmentation.refresh`.
- **Otherwise TTL-only.** A single order does not evict the segmentation
  summary cache (that's a global, admin-facing view — evicting it per-order
  from every user's checkout would be wasteful for a value nobody but an
  admin reads); it simply expires after its TTL.

## 17. Failure behavior

- **Cache down**: every `core.cache.Cache` method catches its exception,
  logs, and returns "miss"/`False` — callers fall straight through to the
  database/ML path. No endpoint returns 500 because Redis is unavailable.
- **Broker/backend down**: `POST /churn/train`, `POST /segmentation/refresh`,
  and `GET /tasks/{id}` return **`503`** with an explanatory message; nothing
  is ever reported as "queued" or "running" when it wasn't actually
  submitted/found.
- **No trained churn model**: `GET /churn/me` and `GET /churn/model` return
  `503`/`404` respectively with a clear message — never a fabricated
  probability.

## 18. Current limitations

- Order status is read as of "now" for historical snapshots (§5) — a
  documented, narrow look-ahead, not a leakage bug.
- `class_weight="balanced"` (default) means `churn_probability` is a ranking
  signal, not a calibrated frequency (§8).
- No model registry/versioned rollback beyond the single artifact at
  `CHURN_MODEL_PATH`; retraining overwrites it in place (the old file is
  gone once the atomic replace completes — no history is kept, matching how
  `CustomerSegment` also stores only the *current* result, not a history).
- No periodic/scheduled retraining (Celery beat) — training is triggered
  explicitly via `POST /churn/train`, same as segmentation's
  `POST /segmentation/refresh`.
- Recommendation cache refresh (`recommendations.refresh_user_cache`) is not
  triggered automatically on every interaction — invalidation is immediate
  (§16), but *repopulating* the cache happens lazily on the next
  `GET /recommendations/me`, not proactively.
- Development/test environment: Redis/Celery are **not** running by default
  in this sandbox; `fakeredis` (dev/test dependency) and
  `CELERY_TASK_ALWAYS_EAGER` let the cache and task code paths be exercised
  in tests without a live broker. A real deployment needs both services
  running (see `docs/authentication.md`-style operational notes to be added
  in a later phase alongside Docker/Phase 5).
