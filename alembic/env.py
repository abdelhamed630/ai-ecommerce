"""Alembic environment.

The URL comes from (in order): config.attributes["sqlalchemy_url"] (used by
tests), then settings.DATABASE_URL (the DATABASE_URL env var / .env). It is
never read from alembic.ini, so no credentials live in the repository.

Model metadata comes from the application's own Base, with every model
imported so `alembic revision --autogenerate` sees the full schema.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from core.config import settings
from database.database import Base
from models import (  # noqa: F401  (register all models on Base.metadata)
    cart,
    customer_segment,
    interaction,
    order,
    payment,
    product,
    user,
)

config = context.config
# Tests set this attribute so running migrations in-process does not replace
# the test runner's logging handlers.
if config.config_file_name is not None and not config.attributes.get("skip_logging_config"):
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def _database_url() -> str:
    return config.attributes.get("sqlalchemy_url") or settings.DATABASE_URL


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


def run_migrations_offline() -> None:
    url = _database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        render_as_batch=_is_sqlite(url),
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    url = _database_url()
    connectable = create_engine(url, poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            # SQLite cannot ALTER most things in place; batch mode recreates tables.
            render_as_batch=_is_sqlite(url),
        )
        with context.begin_transaction():
            context.run_migrations()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
