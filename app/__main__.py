"""Inicia o servidor com host e porta do .env: python -m app"""

import uvicorn

from app.core.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        proxy_headers=False,
    )


if __name__ == "__main__":
    main()
