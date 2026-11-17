import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from app.services.claim_extraction_service import ClaimExtractionService
from app.services.evidence_search_service import EvidenceSearchService
from app.services.text_processing_service import normalize_text, tokenize

CASES_PATH = Path(__file__).resolve().parent / "data" / "claim_extraction_cases.json"
CATEGORIES = ("news", "whatsapp", "question", "uppercase", "lowercase", "clickbait_title", "opinion")
VERACITIES = ("true", "false", "opinion")
MIN_CLAIM_F1 = 0.6


@dataclass(frozen=True)
class ClaimCase:
    id: str
    category: str
    event: str
    veracity: str
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


def load_cases() -> list[ClaimCase]:
    with CASES_PATH.open(encoding="utf-8") as file:
        raw = json.load(file)
    return [ClaimCase(**{**item, "expected_keywords": tuple(item["expected_keywords"])}) for item in raw]


def content_tokens(text: str, stopwords: frozenset[str]) -> set[str]:
    return {token for token in tokenize(normalize_text(text)) if token not in stopwords}


def token_f1(predicted: str, expected: str, stopwords: frozenset[str]) -> float:
    got, want = content_tokens(predicted, stopwords), content_tokens(expected, stopwords)
    common = len(got & want)
    if not common:
        return 0.0
    precision, recall = common / len(got), common / len(want)
    return 2 * precision * recall / (precision + recall)


def evaluate_case(
    case: ClaimCase,
    claim_service: ClaimExtractionService,
    search_service: EvidenceSearchService,
    stopwords: frozenset[str],
) -> CaseResult:
    claim = claim_service.extract(case.text, case.title)
    if not case.checkable:
        # Opinião não tem afirmação verificável: o certo é não extrair nada
        return CaseResult(case, claim, (), claim is None, None)

    keywords = tuple(search_service.extract_keywords(claim, case.title)) if claim else ()
    claim_hit = claim is not None and token_f1(claim, case.expected_claim, stopwords) >= MIN_CLAIM_F1
    found = sum(1 for keyword in case.expected_keywords if keyword in keywords)
    return CaseResult(case, claim, keywords, claim_hit, found / len(case.expected_keywords))


@dataclass(frozen=True)
class Metrics:
    total: int
    claim_accuracy: float
    # Sem valor quando nenhum caso tem palavra-chave esperada (opiniões)
    keyword_recall: float | None


def compute_metrics(results: list[CaseResult]) -> Metrics:
    recalls = [r.keyword_recall for r in results if r.keyword_recall is not None]
    return Metrics(
        total=len(results),
        claim_accuracy=sum(r.claim_hit for r in results) / len(results),
        keyword_recall=sum(recalls) / len(recalls) if recalls else None,
    )


def format_summary(results: list[CaseResult]) -> str:
    """Tabela com o resultado por categoria."""
    by_category: dict[str, list[CaseResult]] = defaultdict(list)
    for result in results:
        by_category[result.case.category].append(result)

    def row(name: str, metrics: Metrics) -> str:
        recall = "-" if metrics.keyword_recall is None else f"{metrics.keyword_recall:.0%}"
        return f"{name:<16} {metrics.total:>5} {metrics.claim_accuracy:>10.0%} {recall:>9}"

    lines = [f"{'categoria':<16} {'casos':>5} {'afirmação':>10} {'palavras':>9}"]
    lines += [row(category, compute_metrics(by_category[category])) for category in CATEGORIES if by_category[category]]
    lines.append(row("total", compute_metrics(results)))
    return "\n".join(lines)


def format_report(results: list[CaseResult]) -> str:
    """Tabela mais os erros caso a caso, para investigar uma queda."""
    lines = [format_summary(results), "", "Erros de afirmação:"]
    for result in results:
        if not result.claim_hit:
            lines.append(f"- {result.case.id}: esperado={result.case.expected_claim!r} obtido={result.claim!r}")
    lines.append("")
    lines.append("Palavras-chave faltando:")
    for result in results:
        if result.keyword_recall is not None and result.keyword_recall < 1:
            missing = [k for k in result.case.expected_keywords if k not in result.keywords]
            lines.append(f"- {result.case.id}: faltou={missing} obtido={list(result.keywords)}")
    return "\n".join(lines)
