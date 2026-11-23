"""Avaliação completa: passa cada caso do conjunto pela verificação inteira e mostra onde ela falha.

Compara a classificação final com a veracidade conhecida de cada caso e aponta, para cada erro,
a etapa em que a verificação se perdeu (frase, palavras-chave, busca, evidências ou decisão).

A seção "Busca" mede se o resultado certo veio entre os encontrados: um resultado cuja manchete
cita todas as palavras-chave esperadas do caso (expected_keywords, pelo radical ou por uma das
alternativas, como "cirurgia|operado"). Assim
separa erro de busca (palavras-chave ruins ou buscador que não achou) de erro de evidência (o
resultado certo veio, mas não virou evidência ou não decidiu).

Usa a internet: as buscas vão aos sites de checagem e jornais, como na API. As respostas ficam
gravadas em training/cache/search_results.json e são reaproveitadas nas execuções seguintes,
então a medição é repetível e só buscas novas vão à internet.

    python -m training.evaluate_verification                    # como o usuário usa (sem data)
    python -m training.evaluate_verification --event-date       # data do evento como data da notícia
    python -m training.evaluate_verification --offline          # só respostas gravadas
    python -m training.evaluate_verification --refresh          # busca tudo de novo
    python -m training.evaluate_verification --category whatsapp --limit 10
    python -m training.evaluate_verification --split validation # validação: só números agregados
    python -m training.evaluate_verification --split test       # teste: só ao fechar uma versão

Por padrão avalia só o treinamento (split "train"). Validação e teste servem para conferir se as
regras e os parâmetros não ficaram sob medida: para eles o relatório nunca lista os erros caso a
caso, só os totais.
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from threading import Lock
from typing import Literal

import httpx

from app.core.config import get_settings
from app.integrations.search_client import SearchClient, SearchError, SearchResult
from app.main import build_verification_service, load_entailment_model
from app.models.news_analysis_model import Classification, VerificationStatus
from app.services.decision_service import DecisionService
from app.services.evidence_search_service import SearchProvider
from app.services.evidence_service import EvidenceItem, canonical_url
from app.services.source_registry_service import load_source_registry
from app.services.text_processing_service import normalize_text, tokenize
from app.services.verification_service import VerificationReport, VerificationService
from tests.claim_dataset import (
    CATEGORIES,
    SPLITS,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    ClaimCase,
    keyword_alternatives,
    load_cases,
)
from tests.claim_dataset import is_claim_hit as dataset_claim_hit
from training.entailment_cache import CachedEntailmentModel, EntailmentStore

ALL_SPLITS = "all"

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_PATH = CACHE_DIR / "search_results.json"
ENTAILMENT_CACHE_PATH = CACHE_DIR / "entailment_scores.json"
# Meio do mês quando o evento só diz "maio de 2024": fica dentro da janela de 60 dias
MID_MONTH_DAY = 15
MONTHS = {
    "janeiro": 1,
    "fevereiro": 2,
    "março": 3,
    "abril": 4,
    "maio": 5,
    "junho": 6,
    "julho": 7,
    "agosto": 8,
    "setembro": 9,
    "outubro": 10,
    "novembro": 11,
    "dezembro": 12,
}
_FULL_DATE = re.compile(r"\b(\d{2})/(\d{2})/(\d{4})\b")
_MONTH_YEAR = re.compile(r"\b(" + "|".join(MONTHS) + r")(?:/\w+)? de (\d{4})\b", re.IGNORECASE)

EXPECTED = {
    "true": Classification.LIKELY_TRUE,
    "false": Classification.LIKELY_FALSE,
    "opinion": Classification.INCONCLUSIVE,
    # Previsão ou pergunta aberta: não há afirmação para julgar
    "not_checkable": Classification.INCONCLUSIVE,
}
# Fato marcado como falso ou boato marcado como verdadeiro: o pior resultado possível
OPPOSITE = {
    Classification.LIKELY_TRUE: Classification.LIKELY_FALSE,
    Classification.LIKELY_FALSE: Classification.LIKELY_TRUE,
}

CacheMode = Literal["reuse", "offline", "refresh"]

# Etapa em que a verificação se perdeu, na ordem do pipeline
STAGE_OK = "acertou"
STAGE_CLAIM_MISSING = "1. frase não encontrada"
STAGE_CLAIM_WRONG = "1. frase errada"
STAGE_NO_KEYWORDS = "2. sem palavras-chave"
STAGE_SOURCES_DOWN = "3. fontes fora do ar"
STAGE_NO_RESULTS = "3. busca sem resultados"
STAGE_RIGHT_RESULT_MISSING = "3. busca sem o resultado certo"
STAGE_UNRELATED = "4. resultados sem relação"
STAGE_OUT_OF_WINDOW = "4. reportagem fora da janela de data"
STAGE_VERDICT_UNMATCHED = "4. checagem com veredito, mas não ligada à frase"
STAGE_VERDICT_MISSING = "4. checagem sem veredito reconhecido"
STAGE_WEAK = "4. evidências fracas ou neutras"
STAGE_CONFLICT = "5. evidências conflitantes"
STAGE_INVERTED = "5. decisão invertida"
STAGE_OPINION_CHECKED = "1. texto sem fato verificável tratado como fato"


class SearchStore:
    """Respostas gravadas das buscas, por fonte e consulta; seguro para as buscas em paralelo."""

    def __init__(self, path: Path, mode: CacheMode):
        self.path = path
        self.mode = mode
        self.lock = Lock()
        self.data: dict[str, list[dict]] = {}
        if path.exists() and mode != "refresh":
            self.data = json.loads(path.read_text(encoding="utf-8"))
        # Resultados devolvidos no caso em andamento (as buscas de um caso rodam em paralelo)
        self.case_results: list[SearchResult] = []
        self.fetched = 0

    @staticmethod
    def key(source: str, query: str, limit: int) -> str:
        return f"{source}|{limit}|{query}"

    def get(self, key: str) -> list[SearchResult] | None:
        with self.lock:
            raw = self.data.get(key)
        return None if raw is None else [result_from_dict(item) for item in raw]

    def put(self, key: str, results: list[SearchResult]) -> None:
        with self.lock:
            self.data[key] = [result_to_dict(result) for result in results]
            self.fetched += 1

    @property
    def results_seen(self) -> int:
        return len(self.case_results)

    def count(self, results: list[SearchResult]) -> None:
        with self.lock:
            self.case_results.extend(results)

    def start_case(self) -> None:
        with self.lock:
            self.case_results = []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            content = json.dumps(self.data, ensure_ascii=False, indent=1, sort_keys=True)
        self.path.write_text(content + "\n", encoding="utf-8")


class RecordingSearchClient:
    """Cliente de busca que reaproveita a resposta gravada e grava as novas."""

    def __init__(self, source: str, client: SearchClient, store: SearchStore):
        self.source = source
        self.client = client
        self.store = store

    def search(self, query: str, limit: int) -> list[SearchResult]:
        key = SearchStore.key(self.source, query, limit)
        results = self.store.get(key)
        if results is None:
            if self.store.mode == "offline":
                raise SearchError(f"Sem resposta gravada para {self.source}: {query!r}")
            # Falhas não são gravadas, para tentar de novo na próxima execução
            results = self.client.search(query, limit)
            self.store.put(key, results)
        self.store.count(results)
        return results


def result_to_dict(result: SearchResult) -> dict:
    published = result.published_at.isoformat() if result.published_at else None
    return {"title": result.title, "url": result.url, "excerpt": result.excerpt, "published_at": published}


def result_from_dict(item: dict) -> SearchResult:
    published = date.fromisoformat(item["published_at"]) if item["published_at"] else None
    return SearchResult(item["title"], item["url"], item["excerpt"], published)


def parse_event_date(event: str) -> date | None:
    """Data do evento escrita no caso: "(11/09/2025)" ou "(maio de 2024)"; períodos longos ficam sem."""
    if match := _FULL_DATE.search(event):
        day, month, year = (int(part) for part in match.groups())
        return date(year, month, day)
    if match := _MONTH_YEAR.search(event):
        return date(int(match.group(2)), MONTHS[match.group(1).lower()], MID_MONTH_DAY)
    return None


@dataclass(frozen=True)
class CaseOutcome:
    case: ClaimCase
    report: VerificationReport
    expected: Classification
    claim_hit: bool
    raw_results: int
    stage: str
    # None quando a busca não se aplica: caso sem afirmação, frase errada ou busca não feita
    search_hit: bool | None = None
    hit_as_evidence: bool | None = None

    @property
    def correct(self) -> bool:
        return self.report.classification == self.expected

    @property
    def serious_error(self) -> bool:
        return OPPOSITE.get(self.expected) == self.report.classification


def is_claim_hit(case: ClaimCase, report: VerificationReport, stopwords: frozenset[str]) -> bool:
    return dataset_claim_hit(case, report.claim, report.intent.value, stopwords)


def diagnose(
    case: ClaimCase,
    report: VerificationReport,
    expected: Classification,
    claim_hit: bool,
    raw_results: int,
    decisions: DecisionService,
    search_hit: bool | None = None,
) -> str:
    """Primeira etapa do pipeline em que o caso se perdeu."""
    if report.classification == expected:
        return STAGE_OK
    if not case.checkable:
        return STAGE_OPINION_CHECKED
    if report.claim is None:
        return STAGE_CLAIM_MISSING
    if not claim_hit:
        return STAGE_CLAIM_WRONG
    if report.verification_status == VerificationStatus.SKIPPED:
        return STAGE_NO_KEYWORDS
    if not report.sources_checked:
        return STAGE_SOURCES_DOWN
    return diagnose_evidences(report, expected, raw_results, decisions, search_hit)


def diagnose_evidences(
    report: VerificationReport,
    expected: Classification,
    raw_results: int,
    decisions: DecisionService,
    search_hit: bool | None = None,
) -> str:
    if raw_results == 0:
        return STAGE_NO_RESULTS
    # Sem o resultado certo, o problema está antes das evidências: palavras-chave ou buscador
    if search_hit is False:
        return STAGE_RIGHT_RESULT_MISSING
    if not report.evidences:
        return STAGE_UNRELATED
    decision = decisions.decide(list(report.evidences), report.sources_checked)
    if decision.reason == "conflicting_evidence":
        return STAGE_CONFLICT
    if report.classification != Classification.INCONCLUSIVE:
        return STAGE_INVERTED
    return diagnose_weak_evidences(report.evidences, expected, decisions)


def diagnose_weak_evidences(
    evidences: tuple[EvidenceItem, ...], expected: Classification, decisions: DecisionService
) -> str:
    """Por que nenhuma evidência encontrada decidiu o caso."""
    fact_checks = [e for e in evidences if e.category == "fact_checker"]
    # Reportagem que sustentaria a frase, não fosse a data distante
    out_of_window = any(e.out_of_window and e.verdict is None and decisions.is_strong(e) for e in evidences)
    stages = [
        (any(e.verdict is not None for e in fact_checks), STAGE_VERDICT_UNMATCHED),
        (any(e.verdict is None for e in fact_checks), STAGE_VERDICT_MISSING),
    ]
    window_stage = [(out_of_window, STAGE_OUT_OF_WINDOW)]
    # Boato depende das agências de checagem; fato verdadeiro, das reportagens
    ordered = stages + window_stage if expected == Classification.LIKELY_FALSE else window_stage + stages
    return next((stage for found, stage in ordered if found), STAGE_WEAK)


def is_right_result(result: SearchResult, expected: tuple[str, ...], stems: Callable[[str], set[str]]) -> bool:
    """A manchete cita todas as palavras-chave esperadas do caso (cada uma em alguma das formas aceitas).

    Só a manchete: o resumo cita nomes de passagem ("Neymar vira escudo da seleção na Copa" não é
    sobre o ligamento do joelho). Sinônimos só contam quando listados como alternativa
    ("morreu|morte"), então a medida depende das expected_keywords escolhidas no conjunto.
    """
    title_tokens = set(tokenize(normalize_text(result.title)))
    title_stems = stems(result.title)

    def cited(alternative: str) -> bool:
        # Palavra curta ("5g") não gera radical: vale a palavra inteira
        alternative_stems = stems(alternative)
        return alternative in title_tokens or bool(alternative_stems) and alternative_stems <= title_stems

    return bool(expected) and all(any(cited(a) for a in keyword_alternatives(k)) for k in expected)


def measure_search(
    case: ClaimCase,
    report: VerificationReport,
    claim_hit: bool,
    results: list[SearchResult],
    stems: Callable[[str], set[str]],
) -> tuple[bool | None, bool | None]:
    """(o resultado certo veio na busca, algum deles virou evidência); None quando não se aplica."""
    if not case.checkable or not claim_hit or not report.sources_checked:
        return None, None
    right = {canonical_url(r.url) for r in results if is_right_result(r, case.expected_keywords, stems)}
    as_evidence = any(canonical_url(e.url) in right for e in report.evidences)
    return bool(right), as_evidence


def evaluate(
    service: VerificationService,
    store: SearchStore,
    cases: list[ClaimCase],
    use_event_date: bool,
    quiet: bool = False,
) -> list[CaseOutcome]:
    assert service.evidence_service is not None
    stems = service.evidence_service.stems
    outcomes = []
    for number, case in enumerate(cases, start=1):
        store.start_case()
        published_at = parse_event_date(case.event) if use_event_date else None
        report = service.verify(case.text, case.title, published_at=published_at, refresh=True)
        expected = EXPECTED[case.veracity]
        claim_hit = is_claim_hit(case, report, service.rules.stopwords)
        search_hit, hit_as_evidence = measure_search(case, report, claim_hit, store.case_results, stems)
        stage = diagnose(case, report, expected, claim_hit, store.results_seen, service.decision_service, search_hit)
        outcomes.append(
            CaseOutcome(case, report, expected, claim_hit, store.results_seen, stage, search_hit, hit_as_evidence)
        )
        if quiet:
            continue
        # Caso de validação ou teste: nem o progresso diz qual errou
        detail = f"{'ok' if stage == STAGE_OK else 'x '} {case.id}" if case.split == TRAIN_SPLIT else ""
        print(f"[{number:>3}/{len(cases)}] {detail}", file=sys.stderr)
    return outcomes


def percent(part: int, total: int) -> str:
    return f"{part / total:.0%}" if total else "-"


def format_confusion(outcomes: list[CaseOutcome]) -> list[str]:
    columns = list(Classification)
    lines = ["Esperado x obtido", f"{'esperado':<14}" + "".join(f"{c.value:>15}" for c in columns)]
    for veracity in EXPECTED:
        row = [o for o in outcomes if o.case.veracity == veracity]
        counts = Counter(o.report.classification for o in row)
        lines.append(f"{veracity:<14}" + "".join(f"{counts[c]:>15}" for c in columns))
    return lines


def format_categories(outcomes: list[CaseOutcome]) -> list[str]:
    by_category: dict[str, list[CaseOutcome]] = defaultdict(list)
    for outcome in outcomes:
        by_category[outcome.case.category].append(outcome)
    header = f"{'categoria':<16} {'casos':>5} {'acerto':>7} {'inconcl.':>9} {'erro grave':>11} {'intenção':>9}"
    lines = ["Por categoria", header]
    groups = [(c, by_category[c]) for c in CATEGORIES if by_category[c]] + [("total", outcomes)]
    for name, group in groups:
        correct = sum(o.correct for o in group)
        inconclusive = sum(o.report.classification == Classification.INCONCLUSIVE for o in group)
        serious = sum(o.serious_error for o in group)
        intents = sum(o.report.intent.value == o.case.intent for o in group)
        lines.append(
            f"{name:<16} {len(group):>5} {percent(correct, len(group)):>7} "
            f"{percent(inconclusive, len(group)):>9} {serious:>11} {percent(intents, len(group)):>9}"
        )
    return lines


def format_search(outcomes: list[CaseOutcome]) -> list[str]:
    """Funil dos casos com a frase certa: o resultado certo veio, virou evidência, o caso acertou."""
    measured = [o for o in outcomes if o.search_hit is not None]
    by_category: dict[str, list[CaseOutcome]] = defaultdict(list)
    for outcome in measured:
        by_category[outcome.case.category].append(outcome)
    header = f"{'categoria':<16} {'casos':>5} {'busca':>7} {'evidência':>10} {'acerto':>7} {'sorte':>6}"
    lines = ["Busca: o resultado certo veio entre os encontrados? (casos com a frase certa)", header]
    groups = [(c, by_category[c]) for c in CATEGORIES if by_category[c]] + [("total", measured)]
    for name, group in groups:
        found = sum(bool(o.search_hit) for o in group)
        as_evidence = sum(bool(o.hit_as_evidence) for o in group)
        correct = sum(o.correct for o in group)
        # Acertou sem o resultado certo na busca: evidência de outro assunto que calhou de concordar
        lucky = sum(o.correct and not o.search_hit for o in group)
        lines.append(
            f"{name:<16} {len(group):>5} {percent(found, len(group)):>7} "
            f"{percent(as_evidence, len(group)):>10} {percent(correct, len(group)):>7} {lucky:>6}"
        )
    return lines


def format_stages(outcomes: list[CaseOutcome]) -> list[str]:
    errors = [o for o in outcomes if not o.correct]
    lines = [f"Onde os {len(errors)} erros aconteceram (primeira etapa que falhou)"]
    for stage, count in sorted(Counter(o.stage for o in errors).items()):
        lines.append(f"{stage:<50} {count:>4}  {percent(count, len(errors)):>5}")
    return lines


def format_errors(outcomes: list[CaseOutcome]) -> list[str]:
    hidden = sum(not o.correct for o in outcomes if o.case.split != TRAIN_SPLIT)
    lines = ["Erros caso a caso" + (f" ({hidden} de validação ou teste não listados)" if hidden else "")]
    for outcome in outcomes:
        if outcome.correct or outcome.case.split != TRAIN_SPLIT:
            continue
        report = outcome.report
        lines.append(f"- {outcome.case.id} [{outcome.stage}]")
        lines.append(f"    esperado={outcome.expected.value} obtido={report.classification.value}")
        lines.append(f"    frase={report.claim!r}")
        lines.append(
            f"    palavras={list(report.keywords)} resultados={outcome.raw_results} evidências={len(report.evidences)}"
            f" resultado certo={SEARCH_LABELS[outcome.search_hit, outcome.hit_as_evidence]}"
        )
        for evidence in report.evidences[:2]:
            lines.append(f"    · {evidence.stance:<11} {evidence.source_name}: {evidence.title[:90]!r}")
    return lines


SEARCH_LABELS = {
    (None, None): "-",
    (False, False): "não veio",
    (True, False): "veio, não virou evidência",
    (True, True): "virou evidência",
}


def format_report(outcomes: list[CaseOutcome], use_event_date: bool, split: str) -> str:
    mode = "com a data do evento" if use_event_date else "sem data (como o usuário usa)"
    names = {TRAIN_SPLIT: "treinamento", VALIDATION_SPLIT: "validação", TEST_SPLIT: "teste", ALL_SPLITS: "todos"}
    sections = [
        [f"Avaliação completa: {len(outcomes)} casos ({names[split]}), {mode}"],
        format_categories(outcomes),
        format_confusion(outcomes),
        format_search(outcomes),
        format_stages(outcomes),
        format_errors(outcomes),
    ]
    return "\n\n".join("\n".join(section) for section in sections)


def build_service(
    store: SearchStore, http_client: httpx.Client, entailments: EntailmentStore | None = None
) -> VerificationService:
    """Mesmo serviço da API, com as buscas e as notas do modelo passando pela gravação."""
    settings = get_settings()
    if not settings.verification_enabled:
        raise SystemExit("VERIFICATION_ENABLED=false no .env: a avaliação completa precisa da busca.")
    registry = load_source_registry(settings.source_registry_path)
    # O modelo não é carregado aqui: entra o substituto com as notas gravadas, que só o carrega se precisar
    lazy_model = settings.nli_enabled and entailments is not None
    build_settings = settings.model_copy(update={"nli_enabled": False}) if lazy_model else settings
    service = build_verification_service(build_settings, registry, http_client)
    search, evidence = service.search_service, service.evidence_service
    assert search is not None and evidence is not None
    search.providers = [
        SearchProvider(p.config, RecordingSearchClient(p.config.key, p.client, store)) for p in search.providers
    ]
    if lazy_model and entailments is not None:
        evidence.entailment_model = CachedEntailmentModel(entailments, lambda: load_entailment_model(settings))
    return service


def entailment_model_id() -> str:
    settings = get_settings()
    return f"{settings.nli_model_repo}@{settings.nli_model_revision}/{settings.nli_model_file}"


def select_cases(split: str, category: str | None, limit: int | None) -> list[ClaimCase]:
    cases = load_cases(None if split == ALL_SPLITS else split)
    cases = [case for case in cases if category is None or case.category == category]
    return cases[:limit] if limit else cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avaliação completa da verificação com o conjunto de casos.")
    parser.add_argument("--event-date", action="store_true", help="usa a data do evento como data da notícia")
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--offline", action="store_true", help="só usa respostas gravadas, sem internet")
    cache.add_argument("--refresh", action="store_true", help="ignora as respostas gravadas e busca de novo")
    parser.add_argument("--category", choices=CATEGORIES, help="avalia só uma categoria")
    parser.add_argument("--limit", type=int, help="avalia só os primeiros N casos")
    parser.add_argument(
        "--split",
        choices=(*SPLITS, ALL_SPLITS),
        default=TRAIN_SPLIT,
        help="treinamento (padrão), validação, teste ou todos",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode: CacheMode = "offline" if args.offline else "refresh" if args.refresh else "reuse"
    store = SearchStore(CACHE_PATH, mode)
    settings = get_settings()
    http_client = httpx.Client(
        timeout=settings.verification_timeout_seconds,
        follow_redirects=True,
        max_redirects=settings.verification_max_redirects,
        headers={"User-Agent": settings.verification_user_agent},
    )
    entailments = EntailmentStore(ENTAILMENT_CACHE_PATH, entailment_model_id())
    service = build_service(store, http_client, entailments)
    try:
        outcomes = evaluate(service, store, select_cases(args.split, args.category, args.limit), args.event_date)
    finally:
        # Grava o que já foi buscado mesmo se a execução for interrompida
        store.save()
        entailments.save()
        if service.search_service is not None:
            service.search_service.close()
        http_client.close()
    print(format_report(outcomes, args.event_date, args.split))
    print(f"\nBuscas novas: {store.fetched} (gravadas em {CACHE_PATH})", file=sys.stderr)
    print(f"Notas novas do modelo: {entailments.computed} (gravadas em {ENTAILMENT_CACHE_PATH})", file=sys.stderr)


if __name__ == "__main__":
    main()
