"""Carrega léxicos e limiares dos indicadores de estilo a partir do arquivo de dados."""

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

from app.services.text_processing_service import normalize_text


class StyleThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    excess_exclamation_min: int
    uppercase_ratio_min: float


class AnalysisRules(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stopwords: frozenset[str]
    sensational_terms: frozenset[str]
    alarmist_terms: frozenset[str]
    style_thresholds: StyleThresholds

    # Termos são comparados com o texto normalizado (minúsculo e sem acento)
    @field_validator("stopwords", "sensational_terms", "alarmist_terms", mode="before")
    @classmethod
    def normalize_terms(cls, values: list[str]) -> frozenset[str]:
        return frozenset(normalize_text(v) for v in values)


@lru_cache(maxsize=4)
def load_analysis_rules(path: Path) -> AnalysisRules:
    with path.open(encoding="utf-8") as file:
        return AnalysisRules.model_validate(json.load(file))
