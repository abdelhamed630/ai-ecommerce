import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from api import admin
from api.auth import router as auth_router
from api.cart import router as cart_router
from api.chatbot import router as chatbot_router
from api.churn import router as churn_router
from api.health import router as health_router
from api.interactions import router as interactions_router
from api.orders import router as orders_router
from api.payments import router as payments_router
from api.products import router as products_router
from api.recommendations import router as recommendations_router
from api.segmentation import router as segmentation_router
from api.tasks import router as tasks_router
from api.users import router as users_router
from core.config import DEV_FALLBACK_SECRET_KEY, settings
from core.logging_config import setup_logging
from core.middleware import RequestLoggingMiddleware
from models import (
    cart,
    customer_segment,
    interaction,
    order,
    payment,
    product,
    user,
)

# NOTE: the schema is NOT created here. Production schema management is
# Alembic only:  alembic upgrade head   (see docs/production.md). Tests create
# their own isolated schemas (tests/conftest.py).

setup_logging()
logger = logging.getLogger(__name__)


def create_app(app_settings=settings) -> FastAPI:
    """Application factory. `app` below is the instance used everywhere; the
    factory exists so configuration-dependent behaviour (CORS, trusted hosts,
    error handling) can be tested with different settings."""
    docs = app_settings.ENABLE_API_DOCS
    application = FastAPI(
        title=app_settings.APP_NAME,
        debug=app_settings.DEBUG,
        docs_url="/docs" if docs else None,
        redoc_url="/redoc" if docs else None,
        openapi_url="/openapi.json" if docs else None,
    )

    # Middleware: the LAST one added is the OUTERMOST. Order (outer -> inner):
    # request logging -> trusted hosts -> CORS -> routes.
    origins = app_settings.cors_origins_list
    if origins:
        wildcard = "*" in origins
        application.add_middleware(
            CORSMiddleware,
            allow_origins=["*"] if wildcard else origins,
            allow_credentials=not wildcard,  # wildcard + credentials is invalid; wildcard is dev-only
            allow_methods=["*"],
            allow_headers=["*"],
        )
    hosts = app_settings.allowed_hosts_list
    if hosts and "*" not in hosts:
        application.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)
    application.add_middleware(RequestLoggingMiddleware)

    @application.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception):
        # Same {"detail": ...} shape as the API's HTTPException responses.
        # The exception text/traceback can contain SQL, paths or credentials,
        # so it is only logged server-side (redacted), never returned.
        logger.error(
            "unhandled exception on %s %s (%s)",
            request.method,
            request.url.path,
            type(exc).__name__,
            exc_info=exc,
        )
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    application.include_router(health_router)
    application.include_router(auth_router)
    application.include_router(products_router)
    application.include_router(cart_router)
    application.include_router(orders_router)
    application.include_router(payments_router)
    application.include_router(interactions_router)
    application.include_router(recommendations_router)
    application.include_router(segmentation_router)
    application.include_router(churn_router)
    application.include_router(chatbot_router)
    application.include_router(tasks_router)
    application.include_router(users_router)
    application.include_router(admin.router)

    # Serve ONLY the avatars directory (not all of MEDIA_ROOT). Uploaded files are
    # validated as real JPEG/PNG/WEBP and stored under server-generated names.
    avatar_dir = os.path.join(app_settings.MEDIA_ROOT, "avatars")
    os.makedirs(avatar_dir, exist_ok=True)
    application.mount("/media/avatars", StaticFiles(directory=avatar_dir), name="avatars")

    @application.get("/")
    def root():
        return {"message": f"{app_settings.APP_NAME} is running"}

    return application


app = create_app()

if settings.SECRET_KEY == DEV_FALLBACK_SECRET_KEY:
    logger.warning("SECRET_KEY is not set: using an insecure development key (never allowed in production)")

logger.info(
    "application configured",
    extra={
        "environment": settings.ENVIRONMENT,
        "debug": settings.DEBUG,
        "database_backend": settings.DATABASE_URL.split(":", 1)[0].split("+", 1)[0],
        "cache_enabled": settings.CACHE_ENABLED,
    },
)
