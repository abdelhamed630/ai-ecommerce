import re
from typing import List, Optional
from urllib.parse import unquote, urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Environments that get the strict, fail-fast validation below.
PRODUCTION_ENVIRONMENTS = ("production", "staging")

# Placeholder/dev secrets that must never be accepted in production. Matching
# is case-insensitive and ignores "-", "_" and whitespace (see _compact), so
# "Change-Me", "CHANGE_ME" and " changeme " are all the same entry.
UNSAFE_SECRET_KEYS = {
    "",
    "change-this-secret-key",
    "changeme",
    "change-me",
    "secret",
    "secret-key",
    "dev-secret-key-not-for-production",
}
# A secret that merely CONTAINS one of these (after normalisation) is a
# placeholder too, e.g. "CHANGE_ME_TO_A_RANDOM_SECRET_KEY" or a template value
# padded to look long enough.
_PLACEHOLDER_MARKERS = ("changeme", "changethis", "notforproduction")

MIN_SECRET_KEY_LENGTH = 32
# A 32-char key made of one or two repeating characters is guessable even if
# it is long enough. `openssl rand -hex 32` yields 16 distinct characters.
MIN_SECRET_KEY_DISTINCT_CHARS = 8

# JWT signing algorithms this app accepts: HMAC only, because SECRET_KEY is a
# shared secret. "none" and asymmetric algorithms (RS*/ES*/PS*) need a key
# pair that this app does not configure, so they are rejected up front.
ALLOWED_JWT_ALGORITHMS = ("HS256", "HS384", "HS512")

# Redis / Celery URLs must use one of these schemes (rediss = TLS).
ALLOWED_REDIS_SCHEMES = ("redis", "rediss")

# Used ONLY outside production when SECRET_KEY is not provided, so a fresh
# local checkout still runs. Never accepted in production (see validator).
DEV_FALLBACK_SECRET_KEY = "dev-secret-key-not-for-production"

_UINT32_MAX = 2**32 - 1

# Used when GROQ_MODEL is unset or blank (e.g. a bare `GROQ_MODEL=` line).
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

# One DNS-ish label: letters, digits, "_" and "-" (no leading/trailing "-"),
# max 63 chars. "_" is allowed because docker-compose service names use it.
_HOST_LABEL = r"[A-Za-z0-9_](?:[A-Za-z0-9_-]{0,61}[A-Za-z0-9_])?"
_HOSTNAME_RE = re.compile(rf"{_HOST_LABEL}(?:\.{_HOST_LABEL})*")
_ORIGIN_RE = re.compile(
    r"(?P<scheme>https?)://(?P<host>[^/\s?#@:\[\]]+|\[[0-9A-Fa-f:.]+\])(?::(?P<port>[0-9]{1,5}))?",
    re.ASCII,
)
_REDIS_URL_RE = re.compile(rf"(?:{'|'.join(ALLOWED_REDIS_SCHEMES)})://", re.IGNORECASE)


class ConfigurationError(ValueError):
    """Unsafe/invalid production configuration.

    Deliberately NOT a pydantic ValidationError: pydantic includes the invalid
    input (for a model-level validator, the whole settings dict, SECRET_KEY
    included) in its error text. This error carries only the names of the
    offending settings, never their values.
    """


def _compact(value: str) -> str:
    """Lower-case and drop "-", "_" and whitespace, for placeholder matching."""
    return re.sub(r"[\s_-]+", "", (value or "").lower())


_UNSAFE_SECRET_KEYS_COMPACT = {_compact(k) for k in UNSAFE_SECRET_KEYS}


def is_placeholder_secret(value: str) -> bool:
    """True if `value` is empty, a known dev/placeholder secret, or contains a
    CHANGE_ME-style / "not for production" marker. Case- and separator-insensitive."""
    compact = _compact(value)
    if compact in _UNSAFE_SECRET_KEYS_COMPACT:
        return True
    return any(marker in compact for marker in _PLACEHOLDER_MARKERS)


def _secret_key_problems(key: str) -> List[str]:
    """Reasons `key` is unacceptable in production. Static text only: the key
    itself must never end up in an error message."""
    reasons: List[str] = []
    stripped = key.strip()
    if is_placeholder_secret(key):
        reasons.append("it is a placeholder or known-unsafe value")
    if key != stripped:
        reasons.append("it has leading or trailing whitespace")
    if len(stripped) < MIN_SECRET_KEY_LENGTH:
        reasons.append(f"it is shorter than {MIN_SECRET_KEY_LENGTH} characters")
    if len(set(stripped)) < MIN_SECRET_KEY_DISTINCT_CHARS:
        reasons.append(f"it has fewer than {MIN_SECRET_KEY_DISTINCT_CHARS} distinct characters")
    return reasons


def _url_password(url: str) -> Optional[str]:
    """The (URL-decoded) password embedded in `url`, or None. Never raises."""
    try:
        password = urlsplit(url).password
    except ValueError:
        return None
    return unquote(password) if password else None


def _is_valid_hostname(host: str) -> bool:
    return 0 < len(host) <= 253 and _HOSTNAME_RE.fullmatch(host) is not None


def _is_valid_allowed_host(entry: str) -> bool:
    # Starlette's TrustedHostMiddleware: "*" (any), "*.example.com" (subdomains
    # of example.com) or an exact hostname. It compares the Host header WITHOUT
    # its port, so an entry containing ":" could never match.
    if entry == "*":
        return True
    if entry.startswith("*."):
        return _is_valid_hostname(entry[2:])
    return _is_valid_hostname(entry)


def _is_valid_cors_origin(entry: str) -> bool:
    # A browser Origin is exactly scheme://host[:port]: no path (so no trailing
    # slash), query, fragment or credentials. Anything else never matches.
    if entry == "*":  # wildcard shape is fine; production rejects it separately
        return True
    match = _ORIGIN_RE.fullmatch(entry)
    if match is None:
        return False
    host = match.group("host")
    if not host.startswith("[") and not _is_valid_hostname(host):
        return False
    port = match.group("port")
    return port is None or 1 <= int(port) <= 65535


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
    DB_POOL_SIZE: int = Field(10, ge=1, le=100)
    DB_MAX_OVERFLOW: int = Field(20, ge=0, le=200)
    DB_POOL_TIMEOUT_SECONDS: int = Field(30, ge=1, le=300)
    DB_POOL_RECYCLE_SECONDS: int = Field(1800, ge=30, le=86400)

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
    # HS256 | HS384 | HS512 only (see ALLOWED_JWT_ALGORITHMS); enforced in every environment.
    ALGORITHM: str = "HS256"
    # 1 minute .. 7 days.
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(60, ge=1, le=10080)

    # Hybrid recommendation tuning (see docs/recommendations.md). Weights are
    # relative and are normalized to sum to 1 by the hybrid layer; override
    # via .env / environment variables. This is the ONLY place they live.
    # Each >= 0 and finite; they must not both be 0 (checked in _check_consistency).
    HYBRID_CONTENT_WEIGHT: float = Field(0.5, ge=0, le=1000, allow_inf_nan=False)
    HYBRID_COLLABORATIVE_WEIGHT: float = Field(0.5, ge=0, le=1000, allow_inf_nan=False)
    # Max similar users used for collaborative scoring.
    CF_MAX_NEIGHBORS: int = Field(50, ge=1, le=10000)

    # Customer segmentation analysis (see docs/customer_segmentation.md and
    # docs/recommendations.md). The ONLY place these live. They govern the
    # on-demand analysis (GET /segmentation/segments); the persisted
    # POST /segmentation/run model keeps its own fixed K in
    # ml/segmentation/model.py.
    # N_CLUSTERS must not exceed MAX_CLUSTERS (checked in _check_consistency).
    SEGMENTATION_N_CLUSTERS: int = Field(4, ge=1, le=100)  # default requested K
    SEGMENTATION_MAX_CLUSTERS: int = Field(10, ge=1, le=100)  # upper bound accepted from clients
    SEGMENTATION_RANDOM_STATE: int = Field(42, ge=0, le=_UINT32_MAX)
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
    # Rejected in production/staging: tasks would run inside web requests.
    CELERY_TASK_ALWAYS_EAGER: bool = False
    CELERY_RESULT_EXPIRES_SECONDS: int = Field(86400, ge=1, le=2592000)  # <= 30 days
    # Celery Beat (scheduled jobs). OFF by default: retraining/re-segmenting
    # on a schedule is only sensible once the dataset is large enough (see
    # docs/production.md). Intervals are in seconds; 0 disables that job.
    CELERY_BEAT_ENABLED: bool = False
    CELERY_BEAT_CHURN_TRAIN_INTERVAL_SECONDS: int = Field(604800, ge=0, le=31536000)  # weekly
    CELERY_BEAT_SEGMENTATION_INTERVAL_SECONDS: int = Field(86400, ge=0, le=31536000)  # daily

    # ---- Cache (Redis). Every cache failure degrades to the uncached path.
    CACHE_ENABLED: bool = True
    CACHE_KEY_PREFIX: str = "ai-ecommerce"
    CACHE_SOCKET_TIMEOUT_SECONDS: float = Field(0.5, ge=0.01, le=30, allow_inf_nan=False)
    # After a Redis failure, skip Redis entirely for this long (avoids paying
    # the connect timeout on every request while Redis is down).
    CACHE_FAILURE_BACKOFF_SECONDS: float = Field(5.0, ge=0, le=3600, allow_inf_nan=False)
    # TTLs: 0 .. 7 days.
    RECOMMENDATION_CACHE_TTL_SECONDS: int = Field(300, ge=0, le=604800)
    CHURN_CACHE_TTL_SECONDS: int = Field(600, ge=0, le=604800)
    SEGMENTATION_CACHE_TTL_SECONDS: int = Field(300, ge=0, le=604800)

    # ---- Churn prediction (see docs/churn_prediction.md)
    # A customer (>= 1 completed order before the cutoff) churns if they make
    # NO completed purchase in the following N days.
    CHURN_INACTIVITY_DAYS: int = Field(60, ge=1, le=3650)
    # Number of historical snapshots (cutoffs) used to build training data,
    # spaced CHURN_INACTIVITY_DAYS apart, the newest being as_of - N days.
    CHURN_SNAPSHOT_COUNT: int = Field(3, ge=1, le=24)
    CHURN_RECENT_ACTIVITY_DAYS: int = Field(30, ge=1, le=3650)
    CHURN_MIN_SAMPLES: int = Field(50, ge=1, le=1_000_000)
    CHURN_MIN_CLASS_SAMPLES: int = Field(5, ge=1, le=1_000_000)
    CHURN_VALIDATION_FRACTION: float = Field(0.25, gt=0, lt=1, allow_inf_nan=False)
    # "balanced" re-weights classes (better recall on rare churn, but the
    # probabilities are then not calibrated frequencies). None disables it.
    CHURN_CLASS_WEIGHT: Optional[str] = "balanced"
    CHURN_LOGREG_C: float = Field(1.0, gt=0, le=1e6, allow_inf_nan=False)
    CHURN_RANDOM_STATE: int = Field(42, ge=0, le=_UINT32_MAX)
    # Risk bands on churn_probability: p < MEDIUM -> LOW, MEDIUM <= p < HIGH
    # -> MEDIUM, p >= HIGH -> HIGH. Business thresholds, not ground truth.
    # Both in [0, 1] and MEDIUM < HIGH (checked in _check_consistency).
    CHURN_RISK_MEDIUM_THRESHOLD: float = Field(0.4, ge=0, le=1, allow_inf_nan=False)
    CHURN_RISK_HIGH_THRESHOLD: float = Field(0.7, ge=0, le=1, allow_inf_nan=False)
    CHURN_MODEL_PATH: str = "ml/artifacts/churn/churn_model.joblib"

    # ---- LLM (shopping chatbot, see services/llm_service.py). Provider: Groq
    # (OpenAI-compatible API). The key comes ONLY from the environment / .env (never from code). It is a
    # SecretStr, so repr()/logging of the settings object shows "**********".
    # Empty = the assistant is "not configured": POST /chatbot/chat still works
    # and answers with a graceful "temporarily unavailable" message.
    GROQ_API_KEY: SecretStr = SecretStr("")
    GROQ_MODEL: str = DEFAULT_GROQ_MODEL
    # Per-request timeout for the provider call (the endpoint is synchronous).
    GROQ_TIMEOUT_SECONDS: float = Field(15.0, ge=1, le=120, allow_inf_nan=False)
    # Chatbot abuse protection / conversation memory.
    # Max POST /chatbot/chat calls per user per minute (0 = no limit). The limit
    # protects the paid/quota-limited LLM key. Shared across workers when Redis is
    # reachable; otherwise counted per worker process.
    CHATBOT_RATE_LIMIT_PER_MINUTE: int = Field(20, ge=0, le=10000)
    # How many previous messages of the conversation (sent by the client in
    # `history`) are passed to the LLM for follow-up questions. 0 = no memory.
    CHATBOT_MAX_HISTORY_TURNS: int = Field(6, ge=0, le=20)

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
    # Field validators raise plain ValueErrors with STATIC text (never the
    # value), and hide_input_in_errors=True keeps pydantic from echoing it.
    @field_validator("ALGORITHM")
    @classmethod
    def _validate_algorithm(cls, value: str) -> str:
        if value not in ALLOWED_JWT_ALGORITHMS:  # exact match: no case folding, no whitespace
            raise ValueError(f"must be one of {', '.join(ALLOWED_JWT_ALGORITHMS)}")
        return value

    @field_validator("GROQ_MODEL")
    @classmethod
    def _default_blank_groq_model(cls, value: str) -> str:
        return value.strip() or DEFAULT_GROQ_MODEL

    @field_validator("REDIS_URL", "CELERY_BROKER_URL", "CELERY_RESULT_BACKEND")
    @classmethod
    def _validate_redis_url_scheme(cls, value: str) -> str:
        if _REDIS_URL_RE.match(value.strip()) is None:
            raise ValueError(f"must be a URL starting with {' or '.join(s + '://' for s in ALLOWED_REDIS_SCHEMES)}")
        return value

    @field_validator("CORS_ALLOWED_ORIGINS")
    @classmethod
    def _validate_cors_origins_shape(cls, value: str) -> str:
        for number, entry in enumerate(_split_csv(value), start=1):
            if not _is_valid_cors_origin(entry):
                raise ValueError(
                    f"entry #{number} is malformed: expected scheme://host[:port] with http or https, "
                    "without path, trailing slash, credentials, query or fragment"
                )
        return value

    @field_validator("ALLOWED_HOSTS")
    @classmethod
    def _validate_allowed_hosts_shape(cls, value: str) -> str:
        for number, entry in enumerate(_split_csv(value), start=1):
            if not _is_valid_allowed_host(entry):
                raise ValueError(
                    f"entry #{number} is malformed: expected a bare hostname (no scheme, port or path), "
                    "'*.domain' or '*'"
                )
        return value

    def __init__(self, **values):
        super().__init__(**values)
        # Run AFTER pydantic validation succeeded, so a failure here is a plain
        # ConfigurationError whose text never contains any setting's value.
        self._check_consistency()
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

    def _check_consistency(self) -> None:
        """Cross-field rules that hold in EVERY environment."""
        problems = []
        if self.SEGMENTATION_N_CLUSTERS > self.SEGMENTATION_MAX_CLUSTERS:
            problems.append("SEGMENTATION_N_CLUSTERS must not exceed SEGMENTATION_MAX_CLUSTERS")
        if self.HYBRID_CONTENT_WEIGHT + self.HYBRID_COLLABORATIVE_WEIGHT <= 0:
            problems.append("HYBRID_CONTENT_WEIGHT and HYBRID_COLLABORATIVE_WEIGHT must not both be 0")
        if self.CHURN_RISK_MEDIUM_THRESHOLD >= self.CHURN_RISK_HIGH_THRESHOLD:
            problems.append("CHURN_RISK_MEDIUM_THRESHOLD must be lower than CHURN_RISK_HIGH_THRESHOLD")
        if problems:
            raise ConfigurationError("Inconsistent configuration: " + "; ".join(problems))

    def _check_production_safety(self) -> None:
        if not self.is_production:
            return

        problems = []
        secret_reasons = _secret_key_problems(self.SECRET_KEY)
        if secret_reasons:
            problems.append(
                "SECRET_KEY is invalid: "
                + "; ".join(secret_reasons)
                + " (use a random value, e.g. `openssl rand -hex 32`)"
            )
        if self.DEBUG:
            problems.append("DEBUG must be false")
        if self.DATABASE_URL.startswith("sqlite"):
            problems.append("DATABASE_URL must point to PostgreSQL (SQLite is development/test only)")
        if "*" in self.allowed_hosts_list or not self.allowed_hosts_list:
            problems.append("ALLOWED_HOSTS must list explicit hostnames (no '*', not empty)")
        if "*" in self.cors_origins_list:
            problems.append("CORS_ALLOWED_ORIGINS must not contain '*' (list explicit origins)")
        if self.CELERY_TASK_ALWAYS_EAGER:
            problems.append("CELERY_TASK_ALWAYS_EAGER must be false (tasks would run inside web requests)")
        for name in ("DATABASE_URL", "REDIS_URL", "CELERY_BROKER_URL", "CELERY_RESULT_BACKEND"):
            password = _url_password(getattr(self, name))
            if password and is_placeholder_secret(password):
                problems.append(f"{name} contains a placeholder password (replace the CHANGE_ME value)")
        if problems:
            raise ConfigurationError(
                f"Unsafe configuration for ENVIRONMENT={self.ENVIRONMENT!r}: " + "; ".join(problems)
            )


settings = Settings()
