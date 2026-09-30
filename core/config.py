from typing import List, Optional

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Environments that get the strict, fail-fast validation below.
PRODUCTION_ENVIRONMENTS = ("production", "staging")

# Placeholder/dev secrets that must never be accepted in production.
UNSAFE_SECRET_KEYS = {
    "",
    "change-this-secret-key",
    "changeme",
    "change-me",
    "secret",
    "secret-key",
    "dev-secret-key-not-for-production",
}
MIN_SECRET_KEY_LENGTH = 32

# Used ONLY outside production when SECRET_KEY is not provided, so a fresh
# local checkout still runs. Never accepted in production (see validator).
DEV_FALLBACK_SECRET_KEY = "dev-secret-key-not-for-production"


class ConfigurationError(ValueError):
    """Unsafe/invalid production configuration.

    Deliberately NOT a pydantic ValidationError: pydantic includes the invalid
    input (for a model-level validator, the whole settings dict, SECRET_KEY
    included) in its error text. This error carries only the names of the
    offending settings, never their values.
    """


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


class Settings(BaseSettings):
    APP_NAME: str = "AI E-Commerce API"
    # development | test | staging | production. "production" and "staging"
    # enable fail-fast validation of the settings below.
    ENVIRONMENT: str = "development"
    # Defaults to False (safe). Local development opts in via .env
    # (DEBUG=true); production must never enable it.
    DEBUG: bool = False

    DATABASE_URL: str = "sqlite:///./ecommerce.db"

    # ---- Database engine / connection pool. Applied to non-SQLite databases
    # only (SQLite keeps its simple dev/test engine). pool_pre_ping is always
    # on for those databases so stale connections are replaced transparently.
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT_SECONDS: int = 30
    DB_POOL_RECYCLE_SECONDS: int = 1800

    # ---- HTTP security (see docs/production.md)
    # Comma-separated. "*" allows any Host header and is rejected in
    # production. Must include every public hostname AND the names used by
    # in-container health checks (localhost, 127.0.0.1).
    ALLOWED_HOSTS: str = "*"
    # Comma-separated browser origins allowed by CORS, e.g.
    # "https://shop.example.com,https://admin.example.com". Empty = the CORS
    # middleware is not installed (no cross-origin browser access).
    # "*" is only accepted outside production and is then sent WITHOUT
    # credentials (browsers reject wildcard + credentials).
    CORS_ALLOWED_ORIGINS: str = ""
    # Serve /docs, /redoc and /openapi.json.
    ENABLE_API_DOCS: bool = True

    # ---- Logging
    LOG_LEVEL: str = "INFO"
    # "json" (one JSON object per line, for log aggregation) or "text".
    LOG_FORMAT: str = "text"

    # ---- Readiness (/health/ready)
    # None = automatic: Redis is required for readiness in production (when
    # the cache is enabled), optional elsewhere. Set true/false to override.
    READINESS_REQUIRE_REDIS: Optional[bool] = None

    # Local development storage for user-uploaded files (avatars). Never
    # commit this directory. In Docker this is a mounted volume. A cloud
    # object store is a future phase.
    MEDIA_ROOT: str = "./media"

    # No usable default: outside production an empty value falls back to a
    # clearly-marked dev key (with a warning); in production/staging an
    # empty, placeholder or short (< 32 chars) key aborts startup.
    SECRET_KEY: str = ""
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Hybrid recommendation tuning (see docs/recommendations.md). Weights are
    # relative and are normalized to sum to 1 by the hybrid layer; override
    # via .env / environment variables. This is the ONLY place they live.
    HYBRID_CONTENT_WEIGHT: float = 0.5
    HYBRID_COLLABORATIVE_WEIGHT: float = 0.5
    # Max similar users used for collaborative scoring.
    CF_MAX_NEIGHBORS: int = 50

    # Customer segmentation analysis (see docs/customer_segmentation.md and
    # docs/recommendations.md). The ONLY place these live. They govern the
    # on-demand analysis (GET /segmentation/segments); the persisted
    # POST /segmentation/run model keeps its own fixed K in
    # ml/segmentation/model.py.
    SEGMENTATION_N_CLUSTERS: int = 4  # default requested K
    SEGMENTATION_MAX_CLUSTERS: int = 10  # upper bound accepted from clients
    SEGMENTATION_RANDOM_STATE: int = 42
    # Clustering features, from: recency, frequency, monetary, views, cart_adds.
    # (env override must be JSON, e.g. '["recency","frequency","monetary"]')
    SEGMENTATION_CLUSTER_FEATURES: List[str] = ["recency", "frequency", "monetary"]
    # False: only customers with >= 1 COMPLETED order are segmented.
    # True: customers with only views/cart-adds are included too (their
    # recency is imputed, see ml/segmentation/features.py).
    SEGMENTATION_INCLUDE_NON_PURCHASERS: bool = False

    # ---- Redis / Celery (Phase 4). Env vars override; no secrets in defaults.
    # Development defaults assume a local Redis on the default port. Nothing
    # connects at import time: clients are created lazily.
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    # Run tasks inline in the calling process (no broker/worker needed).
    CELERY_TASK_ALWAYS_EAGER: bool = False
    CELERY_RESULT_EXPIRES_SECONDS: int = 86400
    # Celery Beat (scheduled jobs). OFF by default: retraining/re-segmenting
    # on a schedule is only sensible once the dataset is large enough (see
    # docs/production.md). Intervals are in seconds; 0 disables that job.
    CELERY_BEAT_ENABLED: bool = False
    CELERY_BEAT_CHURN_TRAIN_INTERVAL_SECONDS: int = 604800  # weekly
    CELERY_BEAT_SEGMENTATION_INTERVAL_SECONDS: int = 86400  # daily

    # ---- Cache (Redis). Every cache failure degrades to the uncached path.
    CACHE_ENABLED: bool = True
    CACHE_KEY_PREFIX: str = "ai-ecommerce"
    CACHE_SOCKET_TIMEOUT_SECONDS: float = 0.5
    # After a Redis failure, skip Redis entirely for this long (avoids paying
    # the connect timeout on every request while Redis is down).
    CACHE_FAILURE_BACKOFF_SECONDS: float = 5.0
    RECOMMENDATION_CACHE_TTL_SECONDS: int = 300
    CHURN_CACHE_TTL_SECONDS: int = 600
    SEGMENTATION_CACHE_TTL_SECONDS: int = 300

    # ---- Churn prediction (see docs/churn_prediction.md)
    # A customer (>= 1 completed order before the cutoff) churns if they make
    # NO completed purchase in the following N days.
    CHURN_INACTIVITY_DAYS: int = 60
    # Number of historical snapshots (cutoffs) used to build training data,
    # spaced CHURN_INACTIVITY_DAYS apart, the newest being as_of - N days.
    CHURN_SNAPSHOT_COUNT: int = 3
    CHURN_RECENT_ACTIVITY_DAYS: int = 30
    CHURN_MIN_SAMPLES: int = 50
    CHURN_MIN_CLASS_SAMPLES: int = 5
    CHURN_VALIDATION_FRACTION: float = 0.25
    # "balanced" re-weights classes (better recall on rare churn, but the
    # probabilities are then not calibrated frequencies). None disables it.
    CHURN_CLASS_WEIGHT: Optional[str] = "balanced"
    CHURN_LOGREG_C: float = 1.0
    CHURN_RANDOM_STATE: int = 42
    # Risk bands on churn_probability: p < MEDIUM -> LOW, MEDIUM <= p < HIGH
    # -> MEDIUM, p >= HIGH -> HIGH. Business thresholds, not ground truth.
    CHURN_RISK_MEDIUM_THRESHOLD: float = 0.4
    CHURN_RISK_HIGH_THRESHOLD: float = 0.7
    CHURN_MODEL_PATH: str = "ml/artifacts/churn/churn_model.joblib"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", hide_input_in_errors=True)

    # ------------------------------------------------------------ derived
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.strip().lower() in PRODUCTION_ENVIRONMENTS

    @property
    def allowed_hosts_list(self) -> List[str]:
        return _split_csv(self.ALLOWED_HOSTS)

    @property
    def cors_origins_list(self) -> List[str]:
        return _split_csv(self.CORS_ALLOWED_ORIGINS)

    @property
    def redis_required_for_readiness(self) -> bool:
        if not self.CACHE_ENABLED:
            return False
        if self.READINESS_REQUIRE_REDIS is not None:
            return self.READINESS_REQUIRE_REDIS
        return self.is_production

    # --------------------------------------------------------- validation
    def __init__(self, **values):
        super().__init__(**values)
        # Runs AFTER pydantic validation succeeded, so a failure here is a plain
        # ConfigurationError whose text never contains any setting's value.
        self._check_production_safety()

    @model_validator(mode="after")
    def _normalize(self) -> "Settings":
        # Never raises (so pydantic never builds an error message containing input).
        # Bare "postgresql://" / "postgres://" URLs would make SQLAlchemy load
        # the legacy psycopg2 driver, which is not a dependency. The project
        # uses psycopg (v3).
        for prefix in ("postgres://", "postgresql://"):
            if self.DATABASE_URL.startswith(prefix):
                self.DATABASE_URL = "postgresql+psycopg://" + self.DATABASE_URL[len(prefix):]
                break

        if not self.is_production and self.SECRET_KEY.strip() == "":
            self.SECRET_KEY = DEV_FALLBACK_SECRET_KEY
        return self

    def _check_production_safety(self) -> None:
        if not self.is_production:
            return

        problems = []
        if self.SECRET_KEY.strip() in UNSAFE_SECRET_KEYS or len(self.SECRET_KEY) < MIN_SECRET_KEY_LENGTH:
            problems.append(
                f"SECRET_KEY is invalid: it must be a random value of at least {MIN_SECRET_KEY_LENGTH} "
                "characters (e.g. `openssl rand -hex 32`); missing, placeholder or short keys are rejected"
            )
        if self.DEBUG:
            problems.append("DEBUG must be false")
        if self.DATABASE_URL.startswith("sqlite"):
            problems.append("DATABASE_URL must point to PostgreSQL (SQLite is development/test only)")
        if "*" in self.allowed_hosts_list or not self.allowed_hosts_list:
            problems.append("ALLOWED_HOSTS must list explicit hostnames (no '*', not empty)")
        if "*" in self.cors_origins_list:
            problems.append("CORS_ALLOWED_ORIGINS must not contain '*' (list explicit origins)")
        if problems:
            raise ConfigurationError(
                f"Unsafe configuration for ENVIRONMENT={self.ENVIRONMENT!r}: " + "; ".join(problems)
            )


settings = Settings()
