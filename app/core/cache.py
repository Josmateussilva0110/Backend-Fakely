import hashlib
from threading import Lock

from cachetools import TTLCache


class ThreadSafeTTLCache:
    """TTLCache com lock, pois rotas síncronas rodam em várias threads."""

    def __init__(self, max_entries: int, ttl_seconds: int):
        self._cache: TTLCache = TTLCache(maxsize=max_entries, ttl=ttl_seconds)
        self._lock = Lock()

    def get(self, key: str):
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, value) -> None:
        with self._lock:
            self._cache[key] = value

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


def build_cache_key(*parts: str | None) -> str:
    raw = "\x1f".join(part or "" for part in parts)
    return hashlib.sha256(raw.encode()).hexdigest()
