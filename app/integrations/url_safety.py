"""Validação de URLs externas contra SSRF (acesso indevido a recursos internos)."""

import ipaddress
import socket
from collections.abc import Callable
from urllib.parse import urlparse

ALLOWED_SCHEMES = ("http", "https")
ALLOWED_PORTS = (None, 80, 443)
BLOCKED_HOSTNAMES = ("localhost",)
BLOCKED_SUFFIXES = (".localhost", ".local", ".internal", ".localdomain")

Resolver = Callable[[str], list[str]]


class UnsafeUrlError(Exception):
    pass


def resolve_host(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UnsafeUrlError(f"Não foi possível resolver o domínio {host}") from exc
    return [str(info[4][0]) for info in infos]


def is_public_ip(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value.split("%")[0])
    except ValueError:
        return False
    # IPv4 mapeado em IPv6 (::ffff:127.0.0.1) é avaliado pelo IPv4 real
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return address.is_global and not address.is_multicast


def _parse(url: str):
    try:
        parsed = urlparse(url)
        port = parsed.port
    except ValueError as exc:
        raise UnsafeUrlError("URL inválida") from exc
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme not in ALLOWED_SCHEMES or not host:
        raise UnsafeUrlError("Esquema não permitido")
    if parsed.username or parsed.password:
        raise UnsafeUrlError("Credenciais na URL não são permitidas")
    if port not in ALLOWED_PORTS:
        raise UnsafeUrlError("Porta não permitida")
    if host in BLOCKED_HOSTNAMES or host.endswith(BLOCKED_SUFFIXES):
        raise UnsafeUrlError("Endereço local não permitido")
    return host


def is_public_web_url(url: str) -> bool:
    """Checagem sem DNS, para links devolvidos ao cliente."""
    try:
        host = _parse(url)
    except UnsafeUrlError:
        return False
    try:
        ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return True
    return is_public_ip(host.strip("[]"))


def ensure_public_url(url: str, resolver: Resolver = resolve_host) -> None:
    """Valida esquema, porta e se todos os IPs do domínio são públicos (antes de cada acesso)."""
    host = _parse(url)
    addresses = resolver(host.strip("[]"))
    if not addresses or not all(is_public_ip(address) for address in addresses):
        raise UnsafeUrlError("Endereço privado ou reservado não permitido")
