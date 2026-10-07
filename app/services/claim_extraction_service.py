"""Identifica a afirmação principal (e outras verificáveis) de uma notícia."""

import re
from dataclasses import dataclass

from app.services.analysis_rules_service import AnalysisRules
from app.services.text_processing_service import find_terms, normalize_text, tokenize

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_WORD_PATTERN = re.compile(r"\w+(?:[-']\w+)*")
MIN_CONTENT_WORD_LENGTH = 4
# Título com ao menos esse número de palavras de conteúdo já costuma ser a própria alegação
MIN_TITLE_CONTENT_WORDS = 3
PROPER_NOUN_BONUS = 2
NUMBER_BONUS = 1
CALL_TO_ACTION_PENALTY = 3


@dataclass(frozen=True)
class ClaimExtraction:
    main_claim: str | None
    candidates: tuple[str, ...]


class ClaimExtractionService:
    def __init__(self, rules: AnalysisRules, min_words: int, max_candidates: int, max_length: int):
        self.rules = rules
        self.min_words = min_words
        self.max_candidates = max_candidates
        self.max_length = max_length

    def extract(self, text: str, title: str | None = None) -> ClaimExtraction:
        sentences = [s for s in self.split_sentences(text) if self._word_count(s) >= self.min_words]
        ranked = sorted(sentences, key=self.score_sentence, reverse=True)

        main_claim = None
        if title and self._content_words(title) >= MIN_TITLE_CONTENT_WORDS:
            main_claim = self._clean(title)
        elif ranked:
            main_claim = ranked[0]

        candidates = [main_claim] if main_claim else []
        for sentence in ranked:
            if len(candidates) >= self.max_candidates:
                break
            if sentence not in candidates:
                candidates.append(sentence)
        return ClaimExtraction(
            main_claim=main_claim[: self.max_length] if main_claim else None,
            candidates=tuple(c[: self.max_length] for c in candidates),
        )

    def split_sentences(self, text: str) -> list[str]:
        return [self._clean(part) for part in _SENTENCE_SPLIT_PATTERN.split(text) if part.strip()]

    def score_sentence(self, sentence: str) -> int:
        """Frases com nomes próprios, números e termos específicos tendem a ser verificáveis."""
        words = _WORD_PATTERN.findall(sentence)
        proper_nouns = sum(1 for word in words[1:] if word[0].isupper() and not word.isupper())
        has_number = any(word.isdigit() for word in words)
        # Chamadas como "compartilhe antes que apaguem" não são afirmações verificáveis
        calls_to_action = len(find_terms(normalize_text(sentence), self.rules.sensational_terms))
        return (
            self._content_words(sentence)
            + PROPER_NOUN_BONUS * proper_nouns
            + NUMBER_BONUS * has_number
            - CALL_TO_ACTION_PENALTY * calls_to_action
        )

    def _content_words(self, text: str) -> int:
        return sum(
            1 for token in tokenize(normalize_text(text))
            if len(token) >= MIN_CONTENT_WORD_LENGTH and token not in self.rules.stopwords
        )

    @staticmethod
    def _word_count(text: str) -> int:
        return len(tokenize(normalize_text(text)))

    @staticmethod
    def _clean(text: str) -> str:
        return " ".join(_URL_PATTERN.sub(" ", text).split())
