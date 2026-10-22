"""Inicia o servidor com host e porta do .env: python -m app"""

import uvicorn

from app.core.config import Settings, get_settings


def server_options(settings: Settings) -> dict:
    # Só confia no X-Forwarded-For vindo dos proxies cadastrados; assim o limite por IP vê o usuário real
    return {
        "host": settings.host,
        "port": settings.port,
        "reload": settings.reload,
        "proxy_headers": bool(settings.trusted_proxies),
        "forwarded_allow_ips": ",".join(settings.trusted_proxies) or None,
    }


def main() -> None:
    uvicorn.run("app.main:app", **server_options(get_settings()))


if __name__ == "__main__":
    main()
