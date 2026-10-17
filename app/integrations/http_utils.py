"""Utilitários HTTP comuns aos clientes de sites externos."""

import httpx


class ResponseTooLargeError(Exception):
    pass


def read_limited(response: httpx.Response, max_bytes: int) -> str:
    # Lê em partes para interromper antes de carregar uma resposta grande demais na memória
    chunks, size = [], 0
    for chunk in response.iter_bytes():
        size += len(chunk)
        if size > max_bytes:
            raise ResponseTooLargeError
        chunks.append(chunk)
    return b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")
