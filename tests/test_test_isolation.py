"""Regression guards for the test harness itself (Phase 5)."""

import os

from celery import Celery

from core.config import settings


def test_celery_env_vars_do_not_leak_into_the_process_environment():
    # Celery lets these env vars override constructor arguments; see tests/conftest.py.
    assert "CELERY_BROKER_URL" not in os.environ
    assert "CELERY_RESULT_BACKEND" not in os.environ


def test_project_celery_app_still_uses_isolated_test_redis_databases():
    assert settings.CELERY_BROKER_URL.endswith("/14")
    assert settings.CELERY_RESULT_BACKEND.endswith("/13")


def test_test_local_celery_app_keeps_its_own_in_memory_broker_and_backend():
    app = Celery("isolation_probe", broker="memory://", backend="cache+memory://")
    assert app.conf.broker_url == "memory://"
    assert app.conf.result_backend == "cache+memory://"
