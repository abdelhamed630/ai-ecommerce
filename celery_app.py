"""Celery application (broker/backend from core.config).

Importing this module only builds the Celery app object: it starts no worker
and opens no connection. The FastAPI app never imports it at startup; it is
imported lazily by services.task_dispatch when a job is submitted. Run a
worker with:

    celery -A celery_app.celery_app worker --loglevel=info

Scheduled jobs (Celery Beat) are OFF by default; enable with
CELERY_BEAT_ENABLED=true and run a single beat process:

    celery -A celery_app.celery_app beat --loglevel=info
"""

from celery import Celery

from core.config import settings

celery_app = Celery(
    "ai_ecommerce",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["tasks.churn", "tasks.segmentation", "tasks.recommendations"],
)

celery_app.conf.update(
    task_always_eager=settings.CELERY_TASK_ALWAYS_EAGER,
    task_eager_propagates=False,  # failures are stored on the result, not raised
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=settings.CELERY_RESULT_EXPIRES_SECONDS,
    task_track_started=True,
    # Fail fast when the broker is down instead of blocking an HTTP request.
    broker_connection_timeout=2,
    broker_connection_retry_on_startup=True,
    task_publish_retry=True,
    task_publish_retry_policy={
        "max_retries": 1,
        "interval_start": 0,
        "interval_step": 0.2,
        "interval_max": 0.2,
    },
    broker_transport_options={"socket_connect_timeout": 2, "socket_timeout": 2},
    redis_socket_connect_timeout=2,
    redis_socket_timeout=2,
)


# ---- Scheduled jobs (opt-in). Only reuses the existing tasks; run exactly ONE
# beat process. An interval of 0 disables that job. See docs/production.md for
# when scheduled retraining is (not) appropriate.
if settings.CELERY_BEAT_ENABLED:
    _schedule = {}
    if settings.CELERY_BEAT_CHURN_TRAIN_INTERVAL_SECONDS > 0:
        _schedule["churn-train-model"] = {
            "task": "churn.train_model",
            "schedule": float(settings.CELERY_BEAT_CHURN_TRAIN_INTERVAL_SECONDS),
        }
    if settings.CELERY_BEAT_SEGMENTATION_INTERVAL_SECONDS > 0:
        _schedule["segmentation-refresh"] = {
            "task": "segmentation.refresh",
            "schedule": float(settings.CELERY_BEAT_SEGMENTATION_INTERVAL_SECONDS),
        }
    celery_app.conf.beat_schedule = _schedule
