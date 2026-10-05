"""Carrega as regras da análise heurística a partir do arquivo de dados."""

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

from app.services.text_processing_service import normalize_text


class AnalysisWeights(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline: float
    sensational_per_term: float
    sensational_max: float
    alarmist_per_term: float
    alarmist_max: float
    repeated_punctuation: float
    excess_exclamation: float
    excess_exclamation_min: int
    uppercase: float
    uppercase_ratio_min: float
    trusted_source: float
    no_text_signals: float
    min_probability: float
    max_probability: float
    fact_check_false_min_probability: float
    fact_check_true_max_probability: float
    news_outlet_match: float


class AnalysisRules(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stopwords: frozenset[str]
    sensational_terms: frozenset[str]
    alarmist_terms: frozenset[str]
    trusted_sources: frozenset[str]
    weights: AnalysisWeights

    # Termos são comparados com o texto normalizado (minúsculo e sem acento)
    @field_validator("stopwords", "sensational_terms", "alarmist_terms", mode="before")
    @classmethod
    def normalize_terms(cls, values: list[str]) -> frozenset[str]:
        return frozenset(normalize_text(v) for v in values)

    @field_validator("trusted_sources", mode="before")
    @classmethod
    def normalize_domains(cls, values: list[str]) -> frozenset[str]:
        return frozenset(v.strip().lower().removeprefix("www.") for v in values)


@lru_cache(maxsize=4)
def load_analysis_rules(path: Path) -> AnalysisRules:
    with path.open(encoding="utf-8") as file:
        return AnalysisRules.model_validate(json.load(file))
