"""Submit Celery tasks from request handlers without pretending success.

The Celery app is imported lazily here, so importing the FastAPI app never
builds a Celery app or touches a broker. If the broker/backend is unreachable
`submit` raises TaskQueueUnavailableError (API -> 503, "nothing was queued").
"""

from typing import Any, Dict


class TaskQueueUnavailableError(Exception):
    """The task could not be queued (broker unreachable or misconfigured)."""


def get_celery_app():
    from celery_app import celery_app

    return celery_app


def submit(task_name: str, *args, **kwargs) -> str:
    """Queue `task_name` and return its task id."""
    try:
        app = get_celery_app()
        app.loader.import_default_modules()  # registers tasks.* (celery_app `include`)
        # apply_async on the registered task (not send_task) so eager mode works.
        return app.tasks[task_name].apply_async(args=args, kwargs=kwargs).id
    except Exception as exc:  # kombu OperationalError, redis errors, ...
        raise TaskQueueUnavailableError(
            f"background task queue is unavailable ({type(exc).__name__}); the task was NOT queued"
        ) from exc


def get_status(task_id: str) -> Dict[str, Any]:
    try:
        result = get_celery_app().AsyncResult(task_id)
        state = result.state
        payload: Dict[str, Any] = {"task_id": task_id, "state": state}
        if state == "SUCCESS":
            payload["result"] = result.result
        elif state == "FAILURE":
            payload["error"] = str(result.result)
        return payload
    except Exception as exc:
        raise TaskQueueUnavailableError(
            f"task result backend is unavailable ({type(exc).__name__})"
        ) from exc
