"""Transforma resultados de busca em evidências: relevância, posição, deduplicação e independência."""

import math
import re
import threading
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Literal
from urllib.parse import urlparse

from app.core.config import Settings
from app.core.exceptions import ServiceUnavailableError
from app.integrations.search_client import SearchResult
from app.ml.entailment_model import EntailmentScorer
from app.ml.part_of_speech_tagger import WordTagger
from app.services.evidence_search_service import SearchOutcome
from app.services.source_registry_service import RegisteredSource, SourceRegistry
from app.services.text_processing_service import (
    WORD_PATTERN,
    extract_quantities,
    find_terms,
    is_shouting,
    normalize_text,
    quantities_conflict,
    remove_accents,
    tokenize,
)
from app.services.verification_config_service import (
    Verdict,
    VerdictPatterns,
    VerificationConfig,
    VerificationSourceConfig,
)

Stance = Literal["supports", "contradicts", "neutral"]
EvidenceCategory = Literal["official", "scientific", "fact_checker", "news_outlet", "unknown"]

# Prefixo usado como "radical" simples (ex.: eficacia/eficaz, vacina/vacinas)
STEM_LENGTH = 5
MIN_TOKEN_LENGTH = 3
VERDICT_SCAN_LENGTH = 200
# Frases do resumo com menos palavras que isso (restos cortados, "...") não vão para o modelo
MIN_PREMISE_SENTENCE_WORDS = 4
_PREMISE_SPLIT_PATTERN = re.compile(r"(?<=[.!?;])\s+")
# Datas da afirmação ("em 14/11/2026") ficam fora do modelo: a janela de publicação já as confere
_DATE_PATTERN = re.compile(r"\s*\b(?:em|no dia|dia)?\s*\d{1,2}/\d{1,2}(?:/\d{2,4})?\b", re.IGNORECASE)
# "3x2" vira "3 a 2", como as reportagens escrevem
_SCORE_PATTERN = re.compile(r"\b(\d{1,2})\s*[xX×]\s*(\d{1,2})\b")
# Restos do veredito retirado da manchete: "É falso que X" vira " que X"; "X #boato" vira "X #"
_RUMOR_LEADING = re.compile(r"^[\W_]*(?:que\b)?[\W_]*", re.IGNORECASE)
_RUMOR_TRAILING = re.compile(r"[\s#:;,\-–—]+$")
# Checagem escrita como negação do boato: "Caminhões não estão sendo retidos"
_NEGATION = re.compile(r"(?<!\w)não(?!\w)\s*", re.IGNORECASE)
# Categorias que publicam apuração própria e podem sustentar uma afirmação
REPORTING_CATEGORIES = ("official", "scientific", "news_outlet")
PROPER_NOUN_CLASS = "PROPN"
NEGATION_WORDS = frozenset({"nao", "nunca", "jamais"})
NAME_CONNECTORS = frozenset({"da", "de", "do", "das", "dos"})
_CLAUSE_END = re.compile(r"[,;:.!?]")
# Vírgula entre orações, não a decimal ("5,35 metros")
_CLAUSE_SPLIT = re.compile(r"(?<!\d)\s*[,;]\s*|\s*[,;]\s*(?!\d)")
FACT_CLAUSE_VERBS = frozenset({"VERB", "AUX"})


@dataclass(frozen=True)
class EvidenceItem:
    position: int
    source_name: str
    organization: str
    category: EvidenceCategory
    retrieved_via: str
    title: str
    url: str
    excerpt: str
    published_at: date | None
    accessed_at: datetime
    relevance: float
    stance: Stance
    verdict: Verdict | None
    # Falso quando é republicação ou outra publicação da mesma organização
    independent: bool
    # Probabilidade de a manchete implicar a afirmação (modelo NLI); None sem o modelo
    entailment: float | None = None
    # Reportagem distante da data da notícia: serve só de contexto, não sustenta
    out_of_window: bool = False


@dataclass(frozen=True)
class RequiredNames:
    """Nomes da afirmação que a manchete precisa citar: o sujeito sempre, e uma parte dos demais.

    Cada nome é o radical das suas palavras ("Minas Gerais" é um nome só). O sujeito vale com metade
    das palavras ("Marília" basta para "Marília Mendonça"; só "Costa" não basta para "Gal Costa");
    os demais, com qualquer uma.
    """

    subject: frozenset[str]
    others: tuple[frozenset[str], ...] = ()
    # Todos os nomes, inclusive os amplos: o que uma checagem precisa citar para desmentir a afirmação
    every: frozenset[str] = frozenset()

    @property
    def all(self) -> frozenset[str]:
        return self.subject.union(*self.others)

    def covered_by(self, title_stems: set[str], others_share: float) -> bool:
        subject_needed = math.ceil(len(self.subject) / 2)
        others_needed = math.floor(len(self.others) * others_share)
        others_cited = sum(1 for name in self.others if name & title_stems)
        return len(self.subject & title_stems) >= subject_needed and others_cited >= others_needed

    def covered_strictly_by(self, title_stems: set[str]) -> bool:
        return self.every <= title_stems


@dataclass(frozen=True)
class Candidate:
    provider: VerificationSourceConfig
    result: SearchResult
    relevance: float
    # A manchete cita os nomes próprios da afirmação (condição para o modelo sustentar)
    names_covered: bool
    # Manchete ou resumo citam ao menos uma palavra do fato, além dos nomes (condição para o modelo)
    fact_covered: bool = True
    registered: RegisteredSource | None = None
    verdict: Verdict | None = None
    # A manchete da checagem cita o fato: é sobre ela que a agência dá o veredito
    fact_in_title: bool = True
    # Idade, porcentagem e valores batem com os da afirmação (quando os dois textos os citam)
    same_numbers: bool = True
    # O boato julgado e a afirmação dizem o fato no mesmo sentido (os dois com "não", ou nenhum)
    same_polarity: bool = True

    @property
    def is_same_fact(self) -> bool:
        # "Silvio Santos morreu aos 86 anos #boato" é um boato antigo, não a morte aos 93;
        # "Marília Mendonça não morreu e está bem #boato" desmente o contrário da morte, não a morte
        return self.fact_in_title and self.same_numbers and self.same_polarity

    @property
    def category(self) -> "EvidenceCategory":
        return self.registered.category if self.registered else "unknown"


class EvidenceService:
    def __init__(
        self,
        settings: Settings,
        config: VerificationConfig,
        registry: SourceRegistry,
        stopwords: frozenset[str],
        entailment_model: EntailmentScorer | None = None,
        tagger: WordTagger | None = None,
        broad_names: frozenset[str] = frozenset(),
    ):
        self.settings = settings
        self.verdict_patterns: VerdictPatterns = config.verdict_patterns
        self.denial_terms = config.denial_terms
        self.roundup_terms = config.roundup_terms
        self.rumor_markers = config.rumor_markers
        self.registry = registry
        self.stopwords = stopwords
        self.entailment_model = entailment_model
        self.tagger = tagger
        # Nomes e datas que não identificam o assunto ("Brasil", "Janeiro", "Deus"): não são exigidos
        self.broad_names = broad_names
        # Compartilhado entre as requisições (o serviço é criado uma vez, no lifespan)
        self.inference_slots = threading.BoundedSemaphore(settings.nli_max_concurrent)

    def build_evidences(
        self, outcome: SearchOutcome, claim: str, title: str | None = None, reference_date: date | None = None
    ) -> list[EvidenceItem]:
        """reference_date: data da notícia; sem ela, vale o dia da análise."""
        reference = reference_date or outcome.accessed_at.date()
        claim_text = normalize_text(f"{title or ''} {claim}")
        required = self.required_names([claim, title or ""], outcome.keywords)
        claim_stems = self.stems(claim_text)
        quantities = extract_quantities(claim_text)
        candidates = self._candidates(
            outcome, claim_stems, required, claim_stems - required.all, quantities, is_negated(main_clause(claim))
        )
        entailments = self.score_entailment(candidates, claim, outcome.keywords)
        evidences = [
            evidence
            for candidate, entailment in zip(candidates, entailments, strict=True)
            if (evidence := self._to_evidence(candidate, claim_text, entailment, reference, outcome.accessed_at))
        ]
        unique = self.deduplicate(evidences)
        # Mais relevantes primeiro; com veredito explícito antes, em caso de empate
        unique.sort(key=lambda e: (e.relevance, e.verdict is not None), reverse=True)
        return self.mark_independence(unique)

    def required_names(self, texts: list[str], keywords: tuple[str, ...]) -> RequiredNames:
        """Nomes próprios da afirmação (a frase antes do título); sem nenhum, as palavras-chave mais específicas.

        O modelo dá nota alta a manchetes com as mesmas entidades e outro protagonista
        ("Courtois renova com o Real Madrid" para "Neymar foi contratado pelo Real Madrid"): o
        sujeito, primeiro nome da frase, é sempre exigido. Exigir todos os nomes barrava a reportagem
        certa quando a frase cita lugares e detalhes ("Voepass", "Vinhedo", "Cascavel", "Guarulhos").
        """
        specific = keywords[: self.settings.verification_min_keywords]
        every = self.stems(" ".join(w for text in texts for w, is_name in self.name_flags(text) if is_name))
        names = [name for text in texts for name in self.name_stems(text)]
        if not names:
            # Sem nomes (texto em minúsculas), as palavras-chave mais específicas fazem o papel deles
            names = [name for name in (frozenset(self.stems(k)) for k in specific) if name]
        if not names:
            return RequiredNames(frozenset(), every=frozenset(every))
        others = tuple(dict.fromkeys(name for name in names[1:] if name != names[0]))
        return RequiredNames(names[0], others, frozenset(every or self.stems(" ".join(specific))))

    def name_stems(self, text: str) -> list[frozenset[str]]:
        """Radicais de cada nome, na ordem da frase; nomes compostos juntos e sem os nomes amplos ("Brasil")."""
        units, current = [], []
        words = self.name_flags(text)
        for i, (word, is_name) in enumerate(words):
            if is_name:
                current.append(word)
                continue
            # "Rio Grande do Sul", "Banco Central do Brasil": o conector fica dentro do nome
            if current and normalize_text(word) in NAME_CONNECTORS and i + 1 < len(words) and words[i + 1][1]:
                continue
            units.append(current)
            current = []
        units.append(current)
        stems = [
            frozenset(self.stems(" ".join(w for w in unit if normalize_text(w) not in self.broad_names)))
            for unit in units
        ]
        return [unit for unit in stems if unit]

    def name_flags(self, text: str) -> list[tuple[str, bool]]:
        """Cada palavra e se é nome: pela classe gramatical; sem o modelo, pela inicial maiúscula.

        Só a maiúscula confunde o começo da frase com nome ("Depois que tomei a vacina..." exigia
        "Depois" na manchete); só o modelo marca como nome placares e palavras em minúsculas ("3x2",
        "x"). Texto todo em maiúsculas não indica nome nenhum.
        """
        if is_shouting(text):
            return []
        if self.tagger is None:
            return [(word, word[0].isupper()) for word in WORD_PATTERN.findall(text)]
        return [
            (word, word_class == PROPER_NOUN_CLASS and word[0].isupper()) for word, word_class in self.tagger.tag(text)
        ]

    def _candidates(
        self,
        outcome: SearchOutcome,
        claim_stems: set[str],
        required: RequiredNames,
        fact_stems: set[str],
        quantities: dict[str, set[float]],
        claim_negated: bool = False,
    ) -> list[Candidate]:
        """Resultados com palavras suficientes em comum com a afirmação (filtro antes do modelo).

        Só os nomes não bastam para o modelo: para "Brasil perde para o Japão", ele chegou a ver
        implicação (0,7) em "É falso que Brasil virou desenho animado no Japão".
        """
        candidates = []
        for source_results in outcome.results:
            for result in source_results.results:
                relevance, overlap = self.compute_relevance(result.title, claim_stems)
                if overlap < self.settings.evidence_min_overlap:
                    continue
                registered = self.registry.find(result.url)
                category: EvidenceCategory = registered.category if registered else "unknown"
                verdict = self.detect_verdict(source_results.source, result, category)
                rumor = self.rumor(result.title, verdict) if verdict is not None else None
                candidates.append(
                    Candidate(
                        source_results.source,
                        result,
                        relevance,
                        names_covered=self.names_covered(required, result.title, verdict),
                        fact_covered=not fact_stems or bool(fact_stems & self.stems(self.premise(result))),
                        registered=registered,
                        verdict=verdict,
                        fact_in_title=not fact_stems or bool(fact_stems & self.stems(result.title)),
                        same_numbers=not quantities_conflict(
                            quantities, extract_quantities(normalize_text(result.title))
                        ),
                        same_polarity=self.same_polarity(result.title, rumor, claim_negated),
                    )
                )
        return candidates

    def score_entailment(
        self, candidates: list[Candidate], claim: str, keywords: tuple[str, ...] = ()
    ) -> list[float | None]:
        """Implicação de cada resultado; None sem o modelo ou quando faltam os nomes ou o fato da afirmação.

        Reportagem: a manchete ou o resumo implicam a afirmação. Checagem com veredito: o boato
        julgado é a mesma afirmação (veja score_rumors).
        """
        scores: list[float | None] = [None] * len(candidates)
        if self.entailment_model is None:
            return scores
        hypothesis = to_hypothesis(claim)
        reports = [i for i, c in enumerate(candidates) if c.verdict is None and c.names_covered and c.fact_covered]
        checks = [i for i, c in enumerate(candidates) if c.verdict is not None and c.is_same_fact and c.names_covered]
        # Reportagem: só a oração do fato; o boato julgado continua comparado com a frase inteira
        report_hypothesis = to_hypothesis(self.fact_clause(claim, keywords))
        report_scores = self.score_reports([candidates[i] for i in reports], report_hypothesis)
        rumor_scores = self.score_rumors([candidates[i] for i in checks], hypothesis)
        for index, score in zip([*reports, *checks], [*report_scores, *rumor_scores], strict=True):
            scores[index] = score
        return scores

    def fact_clause(self, claim: str, keywords: tuple[str, ...] = ()) -> str:
        """Começo da frase até a primeira oração com verbo ligada ao fato (cita um nome ou palavra-chave).

        Detalhes que a manchete não traz derrubam a nota do modelo: "Uma operação policial nos
        complexos do Alemão e da Penha, na zona norte do Rio, deixou mais de cem mortos" deu 0,03
        para "Megaoperação tenta prender 100 membros do CV no Alemão e Penha"; só a primeira
        oração, 0,85. O aposto entre vírgulas fica junto ("O presidente dos EUA, Donald Trump,
        anunciou...") e o que vem depois sai ("..., não esqueçam de adiantar o relógio").
        """
        if self.tagger is None:
            return claim
        parts = [part for part in _CLAUSE_SPLIT.split(claim.rstrip(" .!?")) if part]
        linked = False
        for end, part in enumerate(parts, start=1):
            tagged = self.tagger.tag(part)
            # O nome pode estar no aposto antes do verbo ("Donald Trump, anunciou...")
            tokens = set(tokenize(normalize_text(part)))
            linked = linked or bool(tokens & set(keywords)) or any(c == PROPER_NOUN_CLASS for _, c in tagged)
            if linked and any(word_class in FACT_CLAUSE_VERBS for _, word_class in tagged):
                return ", ".join(parts[:end])
        return claim

    def score_reports(self, candidates: list[Candidate], hypothesis: str) -> list[float]:
        if not candidates:
            return []
        # A manchete e cada frase do resumo vão separadas e vale a maior nota: num bloco único o modelo
        # oscila (0,10 para a reportagem certa do jogo, 0,71 para uma checagem sobre outro assunto)
        groups = [self.premise_sentences(candidate.result) for candidate in candidates]
        flat = [sentence for group in groups for sentence in group]
        results = iter(self.run_model(flat, hypothesis))
        return [round(max(next(results) for _ in group), 3) for group in groups]

    def score_rumors(self, candidates: list[Candidate], hypothesis: str) -> list[float]:
        """A afirmação implica o boato que a checagem julgou falso (ou verdadeiro)?

        Só nesse sentido o veredito passa para a afirmação: se ela implica um boato falso, ela é falsa.
        O contrário não vale: "É falso que o papa enviou uma carta antes de morrer" implica que ele
        morreu, mas não desmente a morte. A manchete inteira também não serve: "É falso que X" nega X
        e o modelo dá nota baixa justamente quando a checagem é sobre a afirmação.
        """
        rumors = [self.rumor(c.result.title, c.verdict) for c in candidates]
        return [round(self.run_model([hypothesis], rumor)[0], 3) for rumor in rumors]

    def rumor(self, title: str, verdict: Verdict | None) -> str:
        """Afirmação julgada pela checagem: "É falso que X" vira "X"; "X não Y" (desmentido) vira "X Y"."""
        text = remove_spans(title, self.verdict_spans(title))
        if text == title and verdict == "false":
            # Sem a marca do veredito, a manchete nega o boato; tirar o "não" devolve o boato
            text = _NEGATION.sub("", text, count=1)
        text = _RUMOR_TRAILING.sub("", _RUMOR_LEADING.sub("", text))
        return " ".join(text.split()) or title

    def names_covered(self, required: RequiredNames, title: str, verdict: Verdict | None) -> bool:
        """Reportagem: o sujeito e parte dos outros nomes. Checagem com veredito: todos os nomes.

        Só o veredito pode chamar um fato verdadeiro de falso, e o modelo dá nota alta a checagens
        de outro fato com os mesmos nomes ("É falso que OMS tenha decretado lockdown por causa de
        Mpox" contra "OMS declarou emergência por causa da mpox", 0,92). Para elas, a regra estrita.
        """
        title_stems = self.stems(title)
        if verdict is not None:
            return required.covered_strictly_by(title_stems)
        return required.covered_by(title_stems, self.settings.evidence_min_name_share)

    def same_polarity(self, title: str, rumor: str | None, claim_negated: bool) -> bool:
        """O boato julgado diz o fato no mesmo sentido da afirmação; só se sabe quando a manchete o apresenta.

        "Marília Mendonça não morreu e está bem #boato" julga o "não morreu". Já em "Receita não está
        taxando o Pix; cobrança é golpe" o "não" é da agência, e o boato é a taxação.
        """
        if rumor is None or not find_terms(normalize_text(title), self.rumor_markers):
            return True
        return is_negated(normalize_text(rumor)) == claim_negated

    def run_model(self, premises: list[str], claim: str) -> list[float]:
        if not self.inference_slots.acquire(timeout=self.settings.nli_queue_timeout_seconds):
            raise ServiceUnavailableError("Muitas análises em andamento. Tente novamente em instantes.")
        try:
            return self.entailment_model.entailment(premises, claim)
        finally:
            self.inference_slots.release()

    def premise_sentences(self, result: SearchResult) -> list[str]:
        excerpt = result.excerpt[: self.settings.evidence_excerpt_max_length]
        sentences = [part.strip() for part in _PREMISE_SPLIT_PATTERN.split(excerpt)]
        return [result.title, *(s for s in sentences if len(s.split()) >= MIN_PREMISE_SENTENCE_WORDS)]

    def premise(self, result: SearchResult) -> str:
        # O resumo completa a manchete: "Paquetá aprova empate do Flamengo com o Estudiantes"
        # só diz que foi "pela Libertadores" no resumo
        excerpt = result.excerpt[: self.settings.evidence_excerpt_max_length]
        return f"{result.title}. {excerpt}" if excerpt else result.title

    def verdict_spans(self, title: str) -> list[tuple[int, int]]:
        """Posições dos padrões de veredito na manchete (comparados sem acento e em minúsculas)."""
        folded = fold_letters(title)
        spans = []
        for pattern in (*self.verdict_patterns.false, *self.verdict_patterns.true):
            spans += [match.span() for match in re.finditer(rf"(?<!\w){re.escape(pattern)}", folded)]
        return spans

    def is_relevant(self, relevance: float, entailment: float | None) -> bool:
        # O modelo recupera manchetes que dizem o mesmo com outras palavras ("anunciado" × "contratado")
        if entailment is not None and entailment >= self.settings.nli_min_entailment:
            return True
        return relevance >= self.settings.evidence_min_relevance

    def _to_evidence(
        self, candidate: Candidate, claim_text: str, entailment: float | None, reference: date, accessed_at: datetime
    ) -> EvidenceItem | None:
        provider, result, relevance = candidate.provider, candidate.result, candidate.relevance
        if not self.is_relevant(relevance, entailment):
            return None
        out_of_window = not self.is_within_window(result.published_at, reference)

        registered, category, verdict = candidate.registered, candidate.category, candidate.verdict
        return EvidenceItem(
            position=0,
            source_name=registered.name if registered else provider.name,
            organization=registered.key if registered else (urlparse(result.url).hostname or "").removeprefix("www."),
            category=category,
            retrieved_via=provider.name,
            title=result.title,
            url=result.url,
            excerpt=result.excerpt[: self.settings.evidence_excerpt_max_length],
            published_at=result.published_at,
            accessed_at=accessed_at,
            relevance=round(relevance, 3),
            stance=self.detect_stance(
                verdict,
                category,
                relevance,
                entailment,
                denial=self.is_denial(result.title, claim_text),
                out_of_window=out_of_window,
                same_fact=candidate.is_same_fact,
            ),
            verdict=verdict,
            independent=True,
            entailment=entailment,
            out_of_window=out_of_window,
        )

    def is_within_window(self, published_at: date | None, reference: date) -> bool:
        # Sem data não há como comparar; a limitação "data não identificada" avisa o usuário
        if published_at is None:
            return True
        return abs((published_at - reference).days) <= self.settings.evidence_support_window_days

    def compute_relevance(self, result_title: str, claim_stems: set[str]) -> tuple[float, int]:
        """Semelhança entre o título do resultado e a afirmação, de 0 a 1.

        Usa o maior valor entre a fração do título coberta pela afirmação (boa para afirmações
        longas) e o coeficiente de Dice (não pune manchetes que trazem a afirmação inteira mais
        detalhes, como "Corinthians anuncia Clarence Seedorf como novo técnico").
        """
        title = normalize_text(result_title)
        # Remove o veredito ("é falso que...") para comparar só o assunto
        for pattern in (*self.verdict_patterns.false, *self.verdict_patterns.true):
            title = title.replace(pattern, " ")
        title_stems = self.stems(title)
        if not title_stems:
            return 0.0, 0
        overlap = len(title_stems & claim_stems)
        title_coverage = overlap / len(title_stems)
        dice = 2 * overlap / (len(title_stems) + len(claim_stems))
        return max(title_coverage, dice), overlap

    def detect_verdict(
        self, provider: VerificationSourceConfig, result: SearchResult, category: EvidenceCategory
    ) -> Verdict | None:
        # Manchete em forma de pergunta ("É verdade que...?") ou que reúne várias checagens não traz veredito
        if category != "fact_checker" or "?" in result.title or self.is_roundup(result.title):
            return None
        content = normalize_text(f"{result.title} {result.excerpt[:VERDICT_SCAN_LENGTH]}")
        return self._match_verdict(content, self.verdict_patterns) or provider.default_verdict

    def is_roundup(self, title: str) -> bool:
        return bool(find_terms(normalize_text(title), self.roundup_terms))

    @staticmethod
    def _match_verdict(content: str, patterns: VerdictPatterns) -> Verdict | None:
        # "não é verdade" contém "é verdade": padrões de falso são verificados primeiro
        if any(re.search(rf"(?<!\w){re.escape(p)}", content) for p in patterns.false):
            return "false"
        if any(re.search(rf"(?<!\w){re.escape(p)}", content) for p in patterns.true):
            return "true"
        return None

    def is_denial(self, result_title: str, claim_text: str) -> bool:
        """Título com desmentido ("nega", "descarta") que a própria afirmação não contém.

        Também conta o veredito de falso em qualquer veículo: o modelo chega a ver implicação
        em "É falso que Japão declarou..." para a afirmação "Japão declarou...".
        """
        title = normalize_text(result_title)
        if self._match_verdict(title, self.verdict_patterns) == "false":
            return True
        found = find_terms(title, self.denial_terms)
        return any(term not in find_terms(claim_text, self.denial_terms) for term in found)

    def detect_stance(
        self,
        verdict: Verdict | None,
        category: EvidenceCategory,
        relevance: float,
        entailment: float | None = None,
        denial: bool = False,
        out_of_window: bool = False,
        same_fact: bool = True,
    ) -> Stance:
        # Veredito sobre outro assunto não decide; assim a posição exibida é a mesma que pesa na decisão.
        # same_fact: a manchete da checagem cita o fato, não só os nomes ("No STF, Bolsonaro repete
        # fakes sobre Pix" não desmente "STF condena Bolsonaro"), e com os mesmos números
        if verdict is not None and same_fact and verdict_matches_claim(relevance, entailment, self.settings):
            return "contradicts" if verdict == "false" else "supports"
        # Só reportagens de fontes com apuração própria sustentam; manchete que desmente algo ou
        # publicada longe da data da notícia, nunca (o veredito acima vale sempre: boatos voltam a circular)
        if category not in REPORTING_CATEGORIES or denial or out_of_window:
            return "neutral"
        # Manchete praticamente igual à afirmação: o modelo às vezes não reconhece nem o texto idêntico
        if relevance >= self.settings.evidence_identical_relevance:
            return "supports"
        # Com o modelo, a manchete precisa implicar a afirmação; sem ele, vale a semelhança de palavras.
        # O modelo nunca gera "contradiz": contradição só vem do veredito das agências de checagem
        if entailment is not None:
            supported = entailment >= self.settings.nli_min_entailment
        else:
            supported = relevance >= self.settings.decision_min_relevance
        return "supports" if supported else "neutral"

    @staticmethod
    def deduplicate(evidences: list[EvidenceItem]) -> list[EvidenceItem]:
        """Mesma página vinda de provedores diferentes vira uma evidência só."""
        best: dict[str, EvidenceItem] = {}
        for evidence in evidences:
            key = canonical_url(evidence.url)
            current = best.get(key)
            if current is None or (evidence.verdict is not None, evidence.relevance) > (
                current.verdict is not None,
                current.relevance,
            ):
                best[key] = evidence
        return list(best.values())

    def mark_independence(self, evidences: list[EvidenceItem]) -> list[EvidenceItem]:
        """Cada organização conta uma vez por posição (apoio/contradição); republicações não contam."""
        seen_organizations: set[tuple[str, Stance]] = set()
        seen_titles: set[frozenset[str]] = set()
        marked = []
        for position, evidence in enumerate(evidences, start=1):
            organization_key = (evidence.organization, evidence.stance)
            independent = organization_key not in seen_organizations
            seen_organizations.add(organization_key)
            # Mesmo título em outro veículo indica republicação; checagens distintas da mesma
            # alegação compartilham o texto da alegação e continuam independentes
            if evidence.category != "fact_checker":
                title_key = frozenset(self.stems(evidence.title))
                independent = independent and title_key not in seen_titles
                seen_titles.add(title_key)
            marked.append(replace(evidence, position=position, independent=independent))
        return marked

    def stems(self, text: str) -> set[str]:
        return {
            token[:STEM_LENGTH]
            for token in tokenize(normalize_text(text))
            if len(token) >= MIN_TOKEN_LENGTH and token not in self.stopwords
        }


def to_hypothesis(claim: str) -> str:
    """Afirmação como o modelo compara melhor: sem datas e com placar por extenso."""
    without_dates = _DATE_PATTERN.sub("", claim)
    return " ".join(_SCORE_PATTERN.sub(r"\1 a \2", without_dates).split()) or claim


def fold_letters(text: str) -> str:
    """Minúsculas e sem acento, letra a letra: as posições continuam as mesmas do texto original."""
    folded = []
    for char in text:
        plain = remove_accents(char.lower())
        folded.append(plain if len(plain) == 1 else char)
    return "".join(folded)


def remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    keep = [True] * len(text)
    for start, end in spans:
        keep[start:end] = [False] * (end - start)
    return "".join(char if kept else " " for char, kept in zip(text, keep, strict=True))


def is_negated(normalized_text: str) -> bool:
    return bool(set(tokenize(normalized_text)) & NEGATION_WORDS)


def main_clause(claim: str) -> str:
    """Primeira oração, normalizada: em "confirmou a volta do horário de verão, não esqueçam..." o "não" é
    um pedido ao leitor, não parte do fato."""
    return normalize_text(_CLAUSE_END.split(claim, maxsplit=1)[0])


def canonical_url(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    return f"{host}{parsed.path.rstrip('/')}"


def verdict_matches_claim(relevance: float, entailment: float | None, settings: Settings) -> bool:
    """A checagem trata da mesma afirmação: palavras suficientes em comum ou implicação pelo modelo.

    "É falso que Tony Ramos morreu aos 89 anos em setembro de 2026 e morte foi anunciada na TV"
    tem palavras demais para a relevância, mas o modelo reconhece a afirmação embutida. Só o modelo
    não basta: ele deu 0,80 para "Notícia sobre fábricas da Coca-Cola fechadas..." contra "A notícia
    que parou o Brasil", e um veredito sobre outro assunto chamaria um fato verdadeiro de falso.
    """
    if relevance >= settings.decision_min_relevance:
        return True
    return (
        entailment is not None
        and entailment >= settings.nli_min_entailment
        and relevance >= settings.verdict_min_relevance
    )
