"""Intenção do texto: o que o usuário quer (conferir uma afirmação, perguntar algo, prever, opinar).

Só regras, sem modelo: o modelo NLI acertou 28% das intenções e levou ~380 ms por texto; as regras,
88%. Pergunta e opinião são reconhecidas com segurança. Previsão e várias afirmações são sinais
mais fracos ("o governo vai taxar o Pix" é boato, não previsão), por isso nunca impedem a verificação
nem mudam a frase verificada (veja VerificationService).
"""

import re
from dataclasses import dataclass

from app.ml.part_of_speech_tagger import WordTagger
from app.models.news_analysis_model import Intent
from app.services.analysis_rules_service import AnalysisRules
from app.services.claim_extraction_service import ClaimExtractionService
from app.services.text_processing_service import normalize_text, tokenize

# Separadores de orações coordenadas: "A vacina tem chip, o 5G espalha o vírus e a máscara..."
_CLAUSE_SPLIT = re.compile(r"\s*[,;]\s*|\s+e\s+")
# Futuro sintético ("chegará", "cairão"): com acento, para não confundir com "para" ou "cara"
_SYNTHETIC_FUTURE = re.compile(r"\b\w+(?:ará|erá|irá|arão|erão|irão)\b", re.IGNORECASE)
VERB_CLASSES = frozenset({"VERB", "AUX"})
# Começo que não abre uma afirmação nova: "para drenar...", "que foi...", "quando..."
DEPENDENT_START_CLASSES = frozenset({"ADP", "SCONJ", "CCONJ", "PRON"})
MIN_CLAUSE_CONTENT_WORDS = 2
# Com 2 orações, frases normais ("A ponte que liga Tocantins e Maranhão desabou") viravam várias
# afirmações (14 alarmes falsos em 139 casos); a partir de 3, foram 3
MIN_CLAIMS = 3


@dataclass(frozen=True)
class IntentResult:
    intent: Intent
    # Afirmação a verificar; vazia na pergunta aberta e na opinião
    claim: str | None


class IntentService:
    def __init__(self, rules: AnalysisRules, claim_service: ClaimExtractionService, tagger: WordTagger | None = None):
        self.rules = rules
        self.claim_service = claim_service
        self.tagger = tagger
        self.open_terms = sorted((tuple(tokenize(t)) for t in rules.open_question_terms), key=len, reverse=True)

    def detect(self, text: str, claim: str | None) -> IntentResult:
        if claim is None and self.claim_service.is_opinion(text):
            return IntentResult(Intent.OPINION, None)
        # "Vocês sabiam?" no começo da mensagem não faz dela uma pergunta
        questions = [
            s
            for s in self.claim_service.split_sentences(text)
            if s.rstrip().endswith("?") and self.claim_service.is_long_enough(s)
        ]
        if any(self.is_open_question(question) for question in questions):
            return IntentResult(Intent.OPEN_QUESTION, None)
        if claim is None:
            return IntentResult(Intent.CLAIM, None)
        if questions:
            return IntentResult(Intent.YES_NO_QUESTION, claim)
        # A frase verificada não muda: dividir sem análise sintática corta frases normais
        # ("A ponte que liga Tocantins e Maranhão desabou")
        if len(self.split_claims(claim)) > 1:
            return IntentResult(Intent.MULTIPLE_CLAIMS, claim)
        if self.is_about_the_future(claim):
            return IntentResult(Intent.PREDICTION, claim)
        return IntentResult(Intent.CLAIM, claim)

    def is_open_question(self, question: str) -> bool:
        """Começa com quem, qual, quanto, onde, como, por que, o que..."""
        tokens = tuple(tokenize(normalize_text(question)))
        return any(tokens[: len(term)] == term for term in self.open_terms)

    def is_about_the_future(self, claim: str) -> bool:
        tokens = set(tokenize(normalize_text(claim)))
        return bool(tokens & self.rules.future_terms) or bool(_SYNTHETIC_FUTURE.search(claim))

    def split_claims(self, claim: str) -> list[str]:
        """Afirmações independentes na mesma frase; sem a classe gramatical, a frase inteira é uma só."""
        if self.tagger is None:
            return [claim]
        parts = [part for part in _CLAUSE_SPLIT.split(claim.rstrip(" .!")) if part]
        clauses: list[str] = []
        for part in parts:
            tagged = self.tagger.tag(part)
            if clauses and self._starts_new_claim(tagged):
                clauses.append(part)
            elif clauses:
                clauses[-1] = f"{clauses[-1]}, {part}"
            else:
                clauses.append(part)
        return clauses if len(clauses) >= MIN_CLAIMS else [claim]

    def _starts_new_claim(self, tagged: list[tuple[str, str]]) -> bool:
        """Trecho com verbo próprio e conteúdo, que não depende do anterior ("para drenar...")."""
        if not tagged or tagged[0][1] in DEPENDENT_START_CLASSES:
            return False
        has_verb = any(word_class in VERB_CLASSES for _, word_class in tagged)
        content = [w for w, _ in tagged if normalize_text(w) not in self.rules.stopwords and len(w) > 2]
        return has_verb and len(content) >= MIN_CLAUSE_CONTENT_WORDS
