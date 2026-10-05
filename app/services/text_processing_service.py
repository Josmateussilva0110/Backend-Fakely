"""Pré-processamento do texto (RN04) e extração de características."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlparse

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
    domain: str | None
    trusted_source: bool | None


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


def extract_domain(url: str | None, source: str | None) -> str | None:
    for candidate in (url, source):
        if not candidate:
            continue
        target = candidate if "://" in candidate else f"http://{candidate}"
        try:
            host = urlparse(target).hostname
        except ValueError:
            continue
        if host and "." in host and " " not in host:
            return host.removeprefix("www.")
    return None


def is_trusted_domain(domain: str | None, trusted_sources: frozenset[str]) -> bool | None:
    if domain is None:
        return None
    return any(domain == d or domain.endswith("." + d) for d in trusted_sources)


def find_terms(normalized_text: str, terms: frozenset[str]) -> tuple[str, ...]:
    return tuple(
        term for term in sorted(terms) if re.search(rf"\b{re.escape(term)}\b", normalized_text)
    )


def extract_features(
    processed: ProcessedText,
    rules: AnalysisRules,
    title: str | None = None,
    url: str | None = None,
    source: str | None = None,
) -> TextFeatures:
    raw = f"{title}\n{processed.original}" if title else processed.original
    normalized = normalize_text(raw)

    letters = [char for char in raw if char.isalpha()]
    uppercase = sum(1 for char in letters if char.isupper())
    domain = extract_domain(url, source)

    return TextFeatures(
        word_count=len(processed.tokens),
        uppercase_ratio=round(uppercase / len(letters), 3) if letters else 0.0,
        exclamation_count=raw.count("!"),
        question_count=raw.count("?"),
        repeated_punctuation_count=len(_REPEATED_PUNCTUATION_PATTERN.findall(raw)),
        sensational_terms=find_terms(normalized, rules.sensational_terms),
        alarmist_terms=find_terms(normalized, rules.alarmist_terms),
        domain=domain,
        trusted_source=is_trusted_domain(domain, rules.trusted_sources),
    )
