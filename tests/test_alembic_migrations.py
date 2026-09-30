"""Alembic baseline vs. the SQLAlchemy models (Phase 5).

Runs `upgrade head` on a fresh throw-away database and asserts that the
resulting schema matches Base.metadata (no autogenerate differences), that the
Phase 5 indexes exist, and that downgrade/upgrade is reversible.

- Always runs on SQLite (temp file).
- Also runs on PostgreSQL when ALEMBIC_TEST_DATABASE_URL is set (its database
  name must contain "test"; the public schema is reset before each run).

NOT VERIFIED when written: not executed by the author.
"""

import os
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect, text

from database.database import Base

ROOT = Path(__file__).resolve().parent.parent


def _config(url: str) -> Config:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.attributes["sqlalchemy_url"] = url
    cfg.attributes["skip_logging_config"] = True
    return cfg


@pytest.fixture(params=["sqlite", "postgresql"])
def migrated_url(request, tmp_path):
    if request.param == "sqlite":
        url = f"sqlite:///{(tmp_path / 'alembic_test.db').as_posix()}"
    else:
        url = os.environ.get("ALEMBIC_TEST_DATABASE_URL")
        if not url:
            pytest.skip("ALEMBIC_TEST_DATABASE_URL not set")
        assert "test" in url.rsplit("/", 1)[-1].lower(), "refusing: database name must contain 'test'"
        eng = create_engine(url)
        with eng.begin() as conn:
            conn.execute(text("DROP SCHEMA public CASCADE"))
            conn.execute(text("CREATE SCHEMA public"))
        eng.dispose()
    yield url
    eng = create_engine(url)
    Base.metadata.drop_all(bind=eng)
    with eng.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
    eng.dispose()


def test_upgrade_head_matches_models(migrated_url):
    command.upgrade(_config(migrated_url), "head")
    eng = create_engine(migrated_url)
    try:
        with eng.connect() as conn:
            diffs = compare_metadata(MigrationContext.configure(conn, opts={"compare_type": True}), Base.metadata)
    finally:
        eng.dispose()
    assert diffs == [], f"models and migrations disagree: {diffs}"


def test_all_tables_and_new_indexes_exist(migrated_url):
    command.upgrade(_config(migrated_url), "head")
    eng = create_engine(migrated_url)
    try:
        insp = inspect(eng)
        assert set(Base.metadata.tables) <= set(insp.get_table_names())
        assert "ix_orders_status_created_at" in {i["name"] for i in insp.get_indexes("orders")}
        assert "ix_order_items_order_id" in {i["name"] for i in insp.get_indexes("order_items")}
    finally:
        eng.dispose()


def test_downgrade_then_upgrade_is_reversible(migrated_url):
    cfg = _config(migrated_url)
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")
    eng = create_engine(migrated_url)
    try:
        assert set(inspect(eng).get_table_names()) <= {"alembic_version"}
    finally:
        eng.dispose()
    command.upgrade(cfg, "head")
