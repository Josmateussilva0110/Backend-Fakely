import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from app.services.claim_extraction_service import ClaimExtractionService
from app.services.evidence_search_service import EvidenceSearchService
from app.services.intent_service import IntentService
from app.services.text_processing_service import normalize_text, tokenize

CASES_PATH = Path(__file__).resolve().parent / "data" / "claim_extraction_cases.json"
CATEGORIES = (
    "news",
    "whatsapp",
    "question",
    "uppercase",
    "lowercase",
    "clickbait_title",
    "opinion",
    "prediction",
    "open_question",
    "multiple_claims",
)
# not_checkable: previsão ou pergunta aberta; não há afirmação para julgar como verdadeira ou falsa
VERACITIES = ("true", "false", "opinion", "not_checkable")
CHECKABLE_VERACITIES = ("true", "false")
# O que o usuário quer com o texto
INTENTS = ("claim", "yes_no_question", "open_question", "multiple_claims", "prediction", "opinion")
# train: casos olhados ao ajustar regras e parâmetros. validation: confirma que a melhora é real; só os
# totais aparecem. test: medido raramente, ao fechar uma versão. Erros caso a caso só do treinamento,
# senão as regras acabam feitas sob medida para os outros grupos também
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
TEST_SPLIT = "test"
SPLITS = (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT)
MIN_CLAIM_F1 = 0.6
# "cirurgia|operado": qualquer uma das formas conta (sinônimos e flexões que o radical não junta)
KEYWORD_ALTERNATIVE_SEPARATOR = "|"


@dataclass(frozen=True)
class ClaimCase:
    id: str
    category: str
    event: str
    veracity: str
    split: str
    intent: str
    title: str | None
    text: str
    expected_claim: str | None
    expected_keywords: tuple[str, ...]

    @property
    def checkable(self) -> bool:
        return self.expected_claim is not None


@dataclass(frozen=True)
class CaseResult:
    case: ClaimCase
    claim: str | None
    keywords: tuple[str, ...]
    claim_hit: bool
    keyword_recall: float | None
    intent: str = ""

    @property
    def intent_hit(self) -> bool:
        return self.intent == self.case.intent


def keyword_alternatives(expected: str) -> list[str]:
    return expected.split(KEYWORD_ALTERNATIVE_SEPARATOR)


def has_keyword(expected: str, keywords: tuple[str, ...]) -> bool:
    return any(alternative in keywords for alternative in keyword_alternatives(expected))


def load_cases(split: str | None = None) -> list[ClaimCase]:
    """Todos os casos, ou só os de um grupo (train, validation ou test)."""
    with CASES_PATH.open(encoding="utf-8") as file:
        raw = json.load(file)
    cases = [ClaimCase(**{**item, "expected_keywords": tuple(item["expected_keywords"])}) for item in raw]
    return [case for case in cases if split is None or case.split == split]


def content_tokens(text: str, stopwords: frozenset[str]) -> set[str]:
    return {token for token in tokenize(normalize_text(text)) if token not in stopwords}


def token_f1(predicted: str, expected: str, stopwords: frozenset[str]) -> float:
    got, want = content_tokens(predicted, stopwords), content_tokens(expected, stopwords)
    common = len(got & want)
    if not common:
        return 0.0
    precision, recall = common / len(got), common / len(want)
    return 2 * precision * recall / (precision + recall)


def is_claim_hit(case: ClaimCase, claim: str | None, intent: str, stopwords: frozenset[str]) -> bool:
    if not case.checkable:
        # Opinião, previsão ou pergunta aberta: o certo é não tratar como afirmação comum. A previsão
        # mantém a frase (ainda é buscada), então vale reconhecer a intenção
        return claim is None or intent == case.intent
    return claim is not None and token_f1(claim, case.expected_claim or "", stopwords) >= MIN_CLAIM_F1


def evaluate_case(
    case: ClaimCase,
    claim_service: ClaimExtractionService,
    intent_service: IntentService,
    search_service: EvidenceSearchService,
    stopwords: frozenset[str],
) -> CaseResult:
    detected = intent_service.detect(case.text, claim_service.extract(case.text, case.title))
    claim, intent = detected.claim, detected.intent.value
    claim_hit = is_claim_hit(case, claim, intent, stopwords)
    if not case.checkable:
        return CaseResult(case, claim, (), claim_hit, None, intent)

    keywords = tuple(search_service.extract_keywords(claim, case.title)) if claim else ()
    found = sum(1 for keyword in case.expected_keywords if has_keyword(keyword, keywords))
    return CaseResult(case, claim, keywords, claim_hit, found / len(case.expected_keywords), intent)


@dataclass(frozen=True)
class Metrics:
    total: int
    claim_accuracy: float
    # Sem valor quando nenhum caso tem palavra-chave esperada (opiniões)
    keyword_recall: float | None
    intent_accuracy: float = 0.0


def compute_metrics(results: list[CaseResult]) -> Metrics:
    recalls = [r.keyword_recall for r in results if r.keyword_recall is not None]
    return Metrics(
        total=len(results),
        claim_accuracy=sum(r.claim_hit for r in results) / len(results),
        keyword_recall=sum(recalls) / len(recalls) if recalls else None,
        intent_accuracy=sum(r.intent_hit for r in results) / len(results),
    )


def in_split(results: list[CaseResult], split: str) -> list[CaseResult]:
    return [result for result in results if result.case.split == split]


def format_summary(results: list[CaseResult]) -> str:
    """Tabela por categoria do treinamento e uma linha com o total da validação; o teste não aparece."""
    train = in_split(results, TRAIN_SPLIT)
    validation = in_split(results, VALIDATION_SPLIT)
    by_category: dict[str, list[CaseResult]] = defaultdict(list)
    for result in train:
        by_category[result.case.category].append(result)

    def row(name: str, metrics: Metrics) -> str:
        recall = "-" if metrics.keyword_recall is None else f"{metrics.keyword_recall:.0%}"
        return (
            f"{name:<16} {metrics.total:>5} {metrics.claim_accuracy:>10.0%} {recall:>9} {metrics.intent_accuracy:>9.0%}"
        )

    lines = [f"{'categoria':<16} {'casos':>5} {'afirmação':>10} {'palavras':>9} {'intenção':>9}"]
    lines += [row(category, compute_metrics(by_category[category])) for category in CATEGORIES if by_category[category]]
    lines.append(row("treinamento", compute_metrics(train)))
    if validation:
        lines.append(row("validação", compute_metrics(validation)))
    return "\n".join(lines)


def has_error(result: CaseResult) -> bool:
    missed_keyword = result.keyword_recall is not None and result.keyword_recall < 1
    return not result.claim_hit or missed_keyword


def format_report(results: list[CaseResult]) -> str:
    """Tabela mais os erros caso a caso, para investigar uma queda."""
    train = in_split(results, TRAIN_SPLIT)
    hidden = sum(has_error(r) for r in in_split(results, VALIDATION_SPLIT))
    lines = [format_summary(results), "", f"Casos de validação com erro: {hidden} (não listados)", ""]
    lines.append("Erros de afirmação:")
    for result in train:
        if not result.claim_hit:
            lines.append(f"- {result.case.id}: esperado={result.case.expected_claim!r} obtido={result.claim!r}")
    lines.append("")
    lines.append("Intenção errada:")
    for result in train:
        if not result.intent_hit:
            lines.append(f"- {result.case.id}: esperado={result.case.intent} obtido={result.intent}")
    lines.append("")
    lines.append("Palavras-chave faltando:")
    for result in train:
        if result.keyword_recall is not None and result.keyword_recall < 1:
            missing = [k for k in result.case.expected_keywords if not has_keyword(k, result.keywords)]
            lines.append(f"- {result.case.id}: faltou={missing} obtido={list(result.keywords)}")
    return "\n".join(lines)
