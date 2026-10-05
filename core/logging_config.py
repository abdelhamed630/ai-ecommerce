"""Application logging: level/format from settings, secrets redacted.

- LOG_FORMAT=json emits one JSON object per line (log aggregators); "text" is
  human readable.
- A filter masks credentials embedded in URLs (redis://:pw@host,
  postgresql://user:pw@host), `Bearer <token>` strings and `password=...`
  style pairs in every log message and in formatted exception text.
- Request bodies and query strings are never logged by this project's
  middleware (see core/middleware.py).
"""

import json
import logging
import re
import sys
from datetime import datetime, timezone

from core.config import settings

_URL_CREDENTIALS = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.\-]*://)[^\s/@:]*:?[^\s/@]*@")
_BEARER = re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=\-]+")
_API_KEY = re.compile(r"\b(?:gsk_|sk-)[A-Za-z0-9_\-]{8,}")  # LLM provider keys (Groq gsk_..., OpenAI-style sk-...)
_KV_SECRET = re.compile(
    r"(?i)\b(password|passwd|secret|secret_key|token|access_token|authorization)\b([\"']?\s*[=:]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;&]+)"
)

_HANDLER_MARK = "_ai_ecommerce_handler"


def redact(text: str) -> str:
    """Mask credentials in `text`. Best-effort defense in depth: code must
    still avoid logging secrets in the first place."""
    text = _URL_CREDENTIALS.sub(lambda m: f"{m.group('scheme')}***@", text)
    text = _BEARER.sub(lambda m: f"{m.group(1)} ***", text)
    text = _API_KEY.sub("***", text)
    text = _KV_SECRET.sub(lambda m: f"{m.group(1)}{m.group(2)}***", text)
    return text


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # malformed format args: never break logging
            return True
        redacted = redact(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


_STANDARD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():  # `extra=` fields (request_id, path, ...)
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


class _TextFormatter(logging.Formatter):
    def formatException(self, ei) -> str:  # noqa: N802 (stdlib name)
        return redact(super().formatException(ei))


def setup_logging() -> None:
    """Idempotent. Adds ONE stream handler to the root logger (never removes
    other handlers, so pytest's log capture keeps working)."""
    root = logging.getLogger()
    level = getattr(logging, str(settings.LOG_LEVEL).upper(), logging.INFO)
    root.setLevel(level)

    for handler in root.handlers:
        if getattr(handler, _HANDLER_MARK, False):
            return

    handler = logging.StreamHandler(sys.stdout)
    setattr(handler, _HANDLER_MARK, True)
    handler.addFilter(RedactingFilter())
    if str(settings.LOG_FORMAT).lower() == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            _TextFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
    root.addHandler(handler)
