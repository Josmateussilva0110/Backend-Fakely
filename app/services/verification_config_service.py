"""Carrega a configuração dos sites consultados na verificação."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, HttpUrl, field_validator

from app.services.text_processing_service import normalize_text

SourceType = Literal["fact_checker", "news_outlet"]
Verdict = Literal["false", "true"]


class VerificationSourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    name: str
    type: SourceType
    client: Literal["wordpress", "folha"]
    search_url: HttpUrl
    allowed_domains: tuple[str, ...]
    default_verdict: Verdict | None = None


class VerdictPatterns(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    false: tuple[str, ...]
    true: tuple[str, ...]

    # Padrões comparados com o texto normalizado (minúsculo e sem acento)
    @field_validator("false", "true", mode="before")
    @classmethod
    def normalize_patterns(cls, values: list[str]) -> tuple[str, ...]:
        return tuple(normalize_text(v) for v in values)


class VerificationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sources: tuple[VerificationSourceConfig, ...]
    verdict_patterns: VerdictPatterns


@lru_cache(maxsize=4)
def load_verification_config(path: Path) -> VerificationConfig:
    with path.open(encoding="utf-8") as file:
        return VerificationConfig.model_validate(json.load(file))
