import secrets

from fastapi import Depends, FastAPI, Header, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import Settings, get_settings
from app.core.exceptions import ForbiddenError

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains",
}

_settings = get_settings()

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=_settings.rate_limit_storage_uri,
    enabled=_settings.rate_limit_enabled,
)


def add_security_headers(app: FastAPI) -> None:
    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        return response


def require_admin_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    expected = settings.admin_api_key
    # compare_digest evita ataques de tempo na comparação
    if expected is None or x_api_key is None or not secrets.compare_digest(
        x_api_key.encode(), expected.get_secret_value().encode()
    ):
        raise ForbiddenError("Acesso negado.")
