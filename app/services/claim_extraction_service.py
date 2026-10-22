import re

from app.services.analysis_rules_service import AnalysisRules
from app.services.text_processing_service import URL_PATTERN, WORD_PATTERN, find_terms, normalize_text, tokenize

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
MIN_CONTENT_WORD_LENGTH = 4
# Título com ao menos esse número de palavras de conteúdo já costuma ser a própria alegação
MIN_TITLE_CONTENT_WORDS = 3
PROPER_NOUN_BONUS = 2
NUMBER_BONUS = 1
CALL_TO_ACTION_PENALTY = 3


class ClaimExtractionService:
    def __init__(self, rules: AnalysisRules, min_words: int, max_length: int):
        self.rules = rules
        self.min_words = min_words
        self.max_length = max_length

    def extract(self, text: str, title: str | None = None) -> str | None:
        if title and self._content_words(title) >= MIN_TITLE_CONTENT_WORDS:
            claim = self._clean(title)
        else:
            sentences = [s for s in self.split_sentences(text) if self._word_count(s) >= self.min_words]
            claim = max(sentences, key=self.score_sentence, default=None)
        return claim[: self.max_length] if claim else None

    def split_sentences(self, text: str) -> list[str]:
        return [self._clean(part) for part in _SENTENCE_SPLIT_PATTERN.split(text) if part.strip()]

    def score_sentence(self, sentence: str) -> int:
        """Frases com nomes próprios, números e termos específicos tendem a ser verificáveis."""
        words = WORD_PATTERN.findall(sentence)
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
        return " ".join(URL_PATTERN.sub(" ", text).split())
