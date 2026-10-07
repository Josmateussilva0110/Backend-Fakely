"""Cadastro de fontes: domínio, categoria e área de atuação (avaliação das fontes)."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, field_validator

SourceCategory = Literal["official", "scientific", "fact_checker", "news_outlet"]

CATEGORY_LABELS = {
    "official": "fonte oficial",
    "scientific": "instituição científica",
    "fact_checker": "agência de checagem",
    "news_outlet": "veículo jornalístico",
}


class RegisteredSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    name: str
    category: SourceCategory
    areas: tuple[str, ...] = ()
    # Domínio, ou domínio + caminho para seções (ex.: "g1.globo.com/fato-ou-fake")
    domains: tuple[str, ...]

    @field_validator("domains", mode="before")
    @classmethod
    def normalize_domains(cls, values: list[str]) -> tuple[str, ...]:
        return tuple(v.strip().lower().removeprefix("www.").rstrip("/") for v in values)


class SourceRegistryData(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sources: tuple[RegisteredSource, ...]


@dataclass(frozen=True)
class SourceAssessment:
    domain: str
    recognized: bool
    key: str | None
    name: str | None
    category: SourceCategory | None


def split_host_path(value: str) -> tuple[str, str] | None:
    target = value if "://" in value else f"http://{value}"
    try:
        parsed = urlparse(target)
        host = parsed.hostname
    except ValueError:
        return None
    if not host or "." not in host or " " in value.strip():
        return None
    return host.lower().removeprefix("www."), parsed.path.lower().rstrip("/")


class SourceRegistry:
    def __init__(self, data: SourceRegistryData):
        # (host, caminho, fonte), do padrão mais específico para o mais genérico
        entries = []
        for source in data.sources:
            for domain in source.domains:
                host, _, path = domain.partition("/")
                entries.append((host, f"/{path}" if path else "", source))
        self._entries = sorted(entries, key=lambda e: (len(e[0]) + len(e[1])), reverse=True)

    def find(self, url_or_domain: str | None) -> RegisteredSource | None:
        parts = split_host_path(url_or_domain) if url_or_domain else None
        if parts is None:
            return None
        host, path = parts
        for domain, prefix, source in self._entries:
            host_matches = host == domain or host.endswith("." + domain)
            if host_matches and (not prefix or path == prefix or path.startswith(prefix + "/")):
                return source
        return None

    def assess(self, url: str | None, source: str | None) -> SourceAssessment | None:
        """Avalia o domínio informado pelo usuário; reconhecer o site não confirma a notícia."""
        for candidate in (url, source):
            parts = split_host_path(candidate) if candidate else None
            if parts is None:
                continue
            registered = self.find(candidate)
            return SourceAssessment(
                domain=parts[0],
                recognized=registered is not None,
                key=registered.key if registered else None,
                name=registered.name if registered else None,
                category=registered.category if registered else None,
            )
        return None


@lru_cache(maxsize=4)
def load_source_registry(path: Path) -> SourceRegistry:
    with path.open(encoding="utf-8") as file:
        return SourceRegistry(SourceRegistryData.model_validate(json.load(file)))
