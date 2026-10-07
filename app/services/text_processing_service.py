"""Pré-processamento do texto (RN04) e extração de características de estilo."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.analysis_rules_service import AnalysisRules

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_REPEATED_PUNCTUATION_PATTERN = re.compile(r"[!?]{2,}")


@dataclass(frozen=True)
class ProcessedText:
    original: str
    normalized: str
    tokens: tuple[str, ...]
    content_tokens: tuple[str, ...]


@dataclass(frozen=True)
class TextFeatures:
    word_count: int
    uppercase_ratio: float
    exclamation_count: int
    question_count: int
    repeated_punctuation_count: int
    sensational_terms: tuple[str, ...]
    alarmist_terms: tuple[str, ...]


def remove_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def normalize_text(text: str) -> str:
    text = _URL_PATTERN.sub(" ", text)
    text = remove_accents(text.lower())
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def tokenize(normalized_text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(normalized_text)


def preprocess_text(text: str, stopwords: frozenset[str]) -> ProcessedText:
    normalized = normalize_text(text)
    tokens = tokenize(normalized)
    return ProcessedText(
        original=text,
        normalized=normalized,
        tokens=tuple(tokens),
        content_tokens=tuple(t for t in tokens if t not in stopwords),
    )


def find_terms(normalized_text: str, terms: frozenset[str]) -> tuple[str, ...]:
    return tuple(
        term for term in sorted(terms) if re.search(rf"\b{re.escape(term)}\b", normalized_text)
    )


def extract_features(processed: ProcessedText, rules: AnalysisRules, title: str | None = None) -> TextFeatures:
    raw = f"{title}\n{processed.original}" if title else processed.original
    normalized = normalize_text(raw)

    letters = [char for char in raw if char.isalpha()]
    uppercase = sum(1 for char in letters if char.isupper())

    return TextFeatures(
        word_count=len(processed.tokens),
        uppercase_ratio=round(uppercase / len(letters), 3) if letters else 0.0,
        exclamation_count=raw.count("!"),
        question_count=raw.count("?"),
        repeated_punctuation_count=len(_REPEATED_PUNCTUATION_PATTERN.findall(raw)),
        sensational_terms=find_terms(normalized, rules.sensational_terms),
        alarmist_terms=find_terms(normalized, rules.alarmist_terms),
    )


def describe_style_signals(features: TextFeatures, rules: AnalysisRules) -> list[str]:
    """Sinais de estilo comuns em desinformação; são alertas, não provas de falsidade."""
    thresholds = rules.style_thresholds
    signals = []
    if features.sensational_terms:
        signals.append(f"Termos sensacionalistas: {', '.join(features.sensational_terms)}")
    if features.alarmist_terms:
        signals.append(f"Afirmações alarmistas: {', '.join(features.alarmist_terms)}")
    if features.repeated_punctuation_count:
        signals.append("Uso de pontuação repetida (ex.: '!!!', '?!')")
    elif features.exclamation_count >= thresholds.excess_exclamation_min:
        signals.append("Excesso de pontos de exclamação")
    if features.uppercase_ratio > thresholds.uppercase_ratio_min:
        signals.append("Uso excessivo de letras maiúsculas")
    return signals
