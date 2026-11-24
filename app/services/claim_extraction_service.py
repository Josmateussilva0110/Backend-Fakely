import re

from app.services.analysis_rules_service import AnalysisRules
from app.services.text_processing_service import (
    URL_PATTERN,
    WORD_PATTERN,
    find_terms,
    is_acronym,
    is_shouting,
    normalize_text,
    tokenize,
)

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
# Emojis fecham um pensamento na mensagem ("caiu agora 😱 quem tiver parente avisa")
_EMOJI_PATTERN = re.compile("[\U0001f000-\U0001faff\u2600-\u27bf\ufe0f\u200d]+")
# Vírgula entre orações, não a decimal ("5,35 metros")
_CLAUSE_SPLIT_PATTERN = re.compile(r"(?<!\d)\s*,\s*|\s*,\s*(?!\d)")
# Em perguntas como "era da Voepass ou da Latam?", a alternativa depois do "ou" não é a afirmação
_ALTERNATIVE_PATTERN = re.compile(r"\s+ou\s+", re.IGNORECASE)
_LEADING_PUNCTUATION = " ,:;!-–—"
_TRAILING_QUESTION_WORD = " mesmo"
MIN_CONTENT_WORD_LENGTH = 4
# Título com ao menos esse número de palavras de conteúdo já costuma ser a própria alegação
MIN_TITLE_CONTENT_WORDS = 3
# Rótulo antes de dois-pontos ("Atenção aposentados:") com até esse número de palavras
LABEL_MAX_WORDS = 4
# Rótulo de fala antes de dois-pontos ("Minha cunhada é médica e falou:") com até esse número de palavras
SPEECH_LABEL_MAX_WORDS = 8
PROPER_NOUN_BONUS = 2
NUMBER_BONUS = 1
CALL_TO_ACTION_PENALTY = 3


class ClaimExtractionService:
    def __init__(self, rules: AnalysisRules, min_words: int, max_length: int):
        self.rules = rules
        self.min_words = min_words
        self.max_length = max_length
        # Prefixos mais longos primeiro: "é verdade mesmo que" antes de "é verdade que"
        self.prefixes = sorted((tuple(tokenize(p)) for p in rules.claim_prefixes), key=len, reverse=True)
        self.label_terms = rules.sensational_terms | rules.alarmist_terms | rules.claim_prefixes

    def extract(self, text: str, title: str | None = None) -> str | None:
        sentences = self._candidate_sentences(text)
        if title and self.is_specific_title(title):
            claim = self.to_claim(title)
        elif sentences:
            claim = self.to_claim(max(sentences, key=self.score_sentence))
        elif title and self._content_words(title) >= MIN_TITLE_CONTENT_WORDS and not self._is_opinion(title):
            # Título vago, mas é o único trecho com tamanho suficiente
            claim = self.to_claim(title)
        else:
            claim = None
        return claim[: self.max_length] if claim else None

    def is_long_enough(self, sentence: str) -> bool:
        return self._word_count(sentence) >= self.min_words

    def is_opinion(self, text: str) -> bool:
        """Todas as frases com tamanho suficiente expressam opinião, sem fato verificável."""
        sentences = [s for s in self.split_sentences(text) if self._word_count(s) >= self.min_words]
        return bool(sentences) and all(self._is_opinion(s) for s in sentences)

    def is_specific_title(self, title: str) -> bool:
        """Título que diz o fato: cita nome, sigla ou número e não é chamariz ("você não vai acreditar")."""
        if self._content_words(title) < MIN_TITLE_CONTENT_WORDS or self._is_opinion(title):
            return False
        if find_terms(normalize_text(title), self.rules.vague_title_terms):
            return False
        has_number = any(word.isdigit() for word in WORD_PATTERN.findall(title))
        return has_number or self._specific_words(title) > 0

    def split_sentences(self, text: str) -> list[str]:
        text = _EMOJI_PATTERN.sub("\n", text)
        parts = (self._clean(part) for part in _SENTENCE_SPLIT_PATTERN.split(text))
        return [part for part in parts if part]

    def score_sentence(self, sentence: str) -> int:
        """Frases com nomes próprios, siglas, números e termos específicos tendem a ser verificáveis."""
        has_number = any(word.isdigit() for word in WORD_PATTERN.findall(sentence))
        # Chamadas como "compartilhe antes que apaguem" não são afirmações verificáveis
        calls_to_action = len(find_terms(normalize_text(sentence), self.rules.sensational_terms))
        return (
            self._content_words(sentence)
            + PROPER_NOUN_BONUS * self._specific_words(sentence)
            + NUMBER_BONUS * has_number
            - CALL_TO_ACTION_PENALTY * calls_to_action
        )

    def to_claim(self, sentence: str) -> str:
        """Transforma a frase do usuário na afirmação: sem aberturas de conversa e sem forma de pergunta."""
        original = self._clean(sentence)
        claim = self._strip_openings(original)
        if original.endswith("?"):
            claim = _ALTERNATIVE_PATTERN.split(claim)[0].rstrip(" ?!.")
            if claim.lower().endswith(_TRAILING_QUESTION_WORD):
                claim = claim[: -len(_TRAILING_QUESTION_WORD)]
        claim = self._cut_tail(claim, is_message=claim != original)
        if not claim:
            return original
        return claim[0].upper() + claim[1:] if claim != original else claim

    def _cut_tail(self, claim: str, is_message: bool) -> str:
        """Tira o que vem depois do fato: pedidos ao leitor e, em mensagem de conversa, o comentário.

        Em mensagem ("Graças a Deus o Faustão fez o transplante, deu tudo certo") o fato é a primeira
        oração; em texto de notícia as vírgulas costumam ser do próprio fato ("A barragem da mina
        Córrego do Feijão, da Vale, rompeu"), então só saem os pedidos ("..., saquem o dinheiro").
        """
        parts = [part for part in _CLAUSE_SPLIT_PATTERN.split(claim) if part]
        if len(parts) < 2:
            return claim
        if is_message and self.is_long_enough(parts[0]):
            return parts[0].rstrip(" .!")
        while len(parts) > 1 and find_terms(normalize_text(parts[-1]), self.rules.call_to_action_terms):
            parts.pop()
        head = ", ".join(parts)
        return head.rstrip(" .!") if head != claim and self.is_long_enough(head) else claim

    def _candidate_sentences(self, text: str) -> list[str]:
        return [
            s for s in self.split_sentences(text) if self._word_count(s) >= self.min_words and not self._is_opinion(s)
        ]

    def _strip_openings(self, sentence: str) -> str:
        while True:
            words = list(WORD_PATTERN.finditer(sentence))
            cut = self._label_length(sentence) or self._prefix_length(words)
            if not cut or cut >= len(words):
                return sentence
            sentence = sentence[words[cut - 1].end() :].lstrip(_LEADING_PUNCTUATION)

    def _prefix_length(self, words: list[re.Match[str]]) -> int:
        tokens = tuple(normalize_text(word.group()) for word in words)
        return next((len(prefix) for prefix in self.prefixes if tokens[: len(prefix)] == prefix), 0)

    def _label_length(self, sentence: str) -> int:
        head, separator, _ = sentence.partition(":")
        label_words = WORD_PATTERN.findall(head)
        if not separator or not label_words:
            return 0
        # "Minha cunhada é médica e falou: a cloroquina cura..." : quem fala não é a afirmação
        if len(label_words) <= SPEECH_LABEL_MAX_WORDS and normalize_text(label_words[-1]) in self.rules.speech_verbs:
            return len(label_words)
        if len(label_words) > LABEL_MAX_WORDS:
            return 0
        return len(label_words) if find_terms(normalize_text(head), self.label_terms) else 0

    def _specific_words(self, text: str) -> int:
        """Siglas e nomes próprios; num texto todo em maiúsculas a caixa não indica nome próprio.

        Nomes amplos ("Brasil") e datas com maiúscula ("Sexta") não contam: não dizem de que fato se trata.
        Letra maiúscula sozinha no meio da frase é nome ("o X foi suspenso"), não sigla.
        """
        words = WORD_PATTERN.findall(text)
        shouting = is_shouting(text)
        acronyms = sum(1 for word in words if is_acronym(word, self.rules.acronyms, shouting))
        if shouting:
            return acronyms
        ignored = self.rules.broad_names | self.rules.generic_terms
        proper_nouns = sum(
            1
            for word in words[1:]
            if word[0].isupper() and (len(word) == 1 or not word.isupper()) and normalize_text(word) not in ignored
        )
        return acronyms + proper_nouns

    def _is_opinion(self, text: str) -> bool:
        return bool(find_terms(normalize_text(text), self.rules.opinion_terms))

    def _content_words(self, text: str) -> int:
        return sum(
            1
            for token in tokenize(normalize_text(text))
            if token not in self.rules.stopwords
            and (len(token) >= MIN_CONTENT_WORD_LENGTH or token in self.rules.acronyms)
        )

    @staticmethod
    def _word_count(text: str) -> int:
        return len(tokenize(normalize_text(text)))

    @staticmethod
    def _clean(text: str) -> str:
        return " ".join(URL_PATTERN.sub(" ", text).split())
