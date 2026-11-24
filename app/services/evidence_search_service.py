"""Pesquisa a afirmação nos provedores de busca e checagem, em paralelo."""

import logging
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import httpx

from app.core.config import Settings
from app.integrations.folha_search_client import FolhaSearchClient
from app.integrations.search_client import HttpFetcher, Period, SearchClient, SearchError, SearchResult
from app.integrations.wordpress_search_client import WordpressSearchClient
from app.ml.part_of_speech_tagger import WordTagger
from app.services.text_processing_service import WORD_PATTERN, is_acronym, is_shouting, normalize_text, tokenize
from app.services.verification_config_service import VerificationConfig, VerificationSourceConfig

logger = logging.getLogger(__name__)

MIN_KEYWORD_LENGTH = 4
# Ordem de especificidade das palavras-chave: siglas, depois substantivos e nomes, depois verbos
ACRONYM_RANK = 3
NOUN_RANK = 2
VERB_RANK = 1
OTHER_WORD_RANK = 0
NOUN_CLASSES = frozenset({"NOUN", "PROPN"})
VERB_CLASSES = frozenset({"VERB"})
# Particípios ("eleito", "condenado") às vezes saem como adjetivo
FACT_VERB_CLASSES = frozenset({"VERB", "ADJ"})
PROPER_NOUN_CLASS = "PROPN"
# Ligam as partes de um nome composto ("Luiz Inácio Lula da Silva", "Banco Central do Brasil")
NAME_CONNECTORS = frozenset({"da", "de", "do", "das", "dos"})


@dataclass(frozen=True)
class SourceResults:
    source: VerificationSourceConfig
    results: tuple[SearchResult, ...]


@dataclass(frozen=True)
class SearchOutcome:
    keywords: tuple[str, ...]
    results: tuple[SourceResults, ...]
    sources_checked: tuple[str, ...]
    sources_failed: tuple[str, ...]
    accessed_at: datetime


@dataclass(frozen=True)
class SearchProvider:
    config: VerificationSourceConfig
    client: SearchClient


class EvidenceSearchService:
    def __init__(
        self,
        settings: Settings,
        stopwords: frozenset[str],
        providers: list[SearchProvider],
        acronyms: frozenset[str] = frozenset(),
        tagger: WordTagger | None = None,
        generic_terms: frozenset[str] = frozenset(),
        fact_verb_stems: frozenset[str] = frozenset(),
    ):
        self.settings = settings
        self.stopwords = stopwords
        self.providers = providers
        self.acronyms = acronyms
        self.tagger = tagger
        self.ignored = stopwords | generic_terms
        self.fact_verb_stems = tuple(sorted(fact_verb_stems))
        self.executor = ThreadPoolExecutor(max_workers=max(len(providers), 1), thread_name_prefix="search")

    @classmethod
    def build(
        cls,
        settings: Settings,
        config: VerificationConfig,
        stopwords: frozenset[str],
        http_client: httpx.Client,
        acronyms: frozenset[str] = frozenset(),
        tagger: WordTagger | None = None,
        generic_terms: frozenset[str] = frozenset(),
        fact_verb_stems: frozenset[str] = frozenset(),
    ) -> "EvidenceSearchService":
        fetcher = HttpFetcher(http_client, settings.verification_max_response_bytes)
        providers = [SearchProvider(source, cls._build_client(source, fetcher)) for source in config.sources]
        return cls(settings, stopwords, providers, acronyms, tagger, generic_terms, fact_verb_stems)

    @staticmethod
    def _build_client(source: VerificationSourceConfig, fetcher: HttpFetcher) -> SearchClient:
        url, domains = str(source.search_url), source.allowed_domains
        if source.client == "wordpress":
            return WordpressSearchClient(fetcher, url, domains)
        return FolhaSearchClient(fetcher, url, domains)

    @property
    def provider_names(self) -> tuple[str, ...]:
        return tuple(provider.config.name for provider in self.providers)

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

    def search(self, claim: str, title: str | None = None, reference_date: date | None = None) -> SearchOutcome | None:
        """reference_date: data da notícia; com ela, a busca também procura dentro da janela de publicação."""
        keywords = self.extract_keywords(claim, title)
        if len(keywords) < self.settings.verification_min_keywords:
            return None

        period = self.period_around(reference_date) if reference_date else None
        extra = self.extra_query(claim, title, keywords)
        futures = {self.executor.submit(self._search_provider, p, keywords, period, extra): p for p in self.providers}
        done, _ = wait(futures, timeout=self.settings.verification_total_timeout_seconds)

        results, checked, failed = [], [], []
        for future, provider in futures.items():
            name = provider.config.name
            if future not in done:
                future.cancel()
                failed.append(name)
                logger.warning("Tempo esgotado ao consultar %s", name)
                continue
            try:
                found = future.result()
            except SearchError as exc:
                failed.append(name)
                logger.warning("Falha ao consultar %s: %s", name, exc)
                continue
            checked.append(name)
            results.append(SourceResults(provider.config, tuple(found)))

        return SearchOutcome(
            keywords=tuple(keywords),
            results=tuple(results),
            sources_checked=tuple(checked),
            sources_failed=tuple(failed),
            accessed_at=datetime.now(UTC),
        )

    def extract_keywords(self, claim: str, title: str | None = None) -> list[str]:
        """Palavras mais específicas da afirmação, completadas pelo título.

        Retorna em ordem de especificidade: siglas (STF, Pix), substantivos e nomes próprios,
        verbos e depois as demais; dentro de cada grupo, na ordem em que aparecem.
        """
        limit = self.settings.verification_max_keywords
        return self._ranked_candidates(claim, title, limit)[:limit]

    def extra_query(self, claim: str, title: str | None, keywords: list[str]) -> list[str] | None:
        """Consulta curta com as palavras que não couberam: a principal mais as seguintes.

        A busca dos sites exige todas as palavras, então mais vagas na mesma consulta filtram demais
        (com 5 ou 6 a busca caiu de 75% para 69% na validação); numa consulta à parte, somam resultados.
        """
        size = self.settings.verification_extra_query_keywords
        if not size or not keywords:
            return None
        needed = self.settings.verification_max_keywords + size
        leftovers = [t for t in self._ranked_candidates(claim, title, needed) if t not in keywords][:size]
        # Com uma palavra só sobrando, a consulta repetiria quase a principal
        return [keywords[0], *leftovers] if len(leftovers) >= self.settings.verification_min_keywords else None

    def _ranked_candidates(self, claim: str, title: str | None, needed: int) -> list[str]:
        # Com a classe gramatical, desempata pela posição: o assunto vem no começo da frase
        # ("Donald Trump venceu...") e as palavras longas costumam ser genéricas ("presidente",
        # "categoria"). Sem ela, todas ficam no mesmo grupo e o tamanho é o único sinal que resta
        by_position = self.tagger is not None
        candidates: dict[str, tuple[int, int]] = {}
        for chunk in (claim, title):
            for word, rank in self._ranked_words(chunk or ""):
                token = normalize_text(word)
                if token and not is_number_like(token) and token not in self.ignored and token not in candidates:
                    candidates[token] = (rank, -len(candidates) if by_position else len(token))
            if len(candidates) >= needed:
                break
        return sorted(candidates, key=lambda t: candidates[t], reverse=True)

    def _ranked_words(self, text: str) -> list[tuple[str, int]]:
        shouting = is_shouting(text)
        if self.tagger is not None:
            # Texto todo em maiúsculas engana o modelo ("TRUMP ANUNCIA" vira organização): vai em
            # minúsculas, que basta para separar substantivo de verbo; as siglas vêm da lista conhecida
            words = self._drop_name_middles(self.tagger.tag(text.lower() if shouting else text))
            tagged = [(word, self._word_rank(word, word_class)) for word, word_class in words]
        else:
            tagged = [
                (word, NOUN_RANK if word[0].isupper() and not shouting else OTHER_WORD_RANK)
                for word in WORD_PATTERN.findall(text)
            ]
        ranked = []
        for word, rank in tagged:
            if is_acronym(word, self.acronyms, shouting):
                ranked.append((word, ACRONYM_RANK))
            elif len(normalize_text(word)) >= MIN_KEYWORD_LENGTH:
                ranked.append((word, rank))
        return ranked

    def _drop_name_middles(self, tagged: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Num nome composto ficam só a primeira e a última palavra útil.

        Sem isso "Donald John Trump" ou "Supremo Tribunal Federal" ocupam quase todas as vagas
        da busca e tiram o fato ("tarifa", "condena"). Siglas dentro do nome ficam sempre.
        """
        dropped: set[int] = set()
        for chain in self._name_chains(tagged):
            useful = [i for i in chain if self._is_name_part_candidate(tagged[i][0])]
            dropped.update(useful[1:-1])
        return [pair for i, pair in enumerate(tagged) if i not in dropped]

    @staticmethod
    def _name_chains(tagged: list[tuple[str, str]]) -> list[list[int]]:
        """Posições de cada sequência de nomes próprios, aceitando "da", "do"... entre eles."""
        chains: list[list[int]] = []
        current: list[int] = []
        for i, (word, word_class) in enumerate(tagged):
            if word_class == PROPER_NOUN_CLASS:
                current.append(i)
                continue
            next_is_name = i + 1 < len(tagged) and tagged[i + 1][1] == PROPER_NOUN_CLASS
            if current and next_is_name and normalize_text(word) in NAME_CONNECTORS:
                continue
            if current:
                chains.append(current)
            current = []
        if current:
            chains.append(current)
        return chains

    def _is_name_part_candidate(self, word: str) -> bool:
        token = normalize_text(word)
        return (
            len(token) >= MIN_KEYWORD_LENGTH
            and token not in self.ignored
            and not is_acronym(word, self.acronyms, False)
        )

    def _word_rank(self, word: str, word_class: str) -> int:
        # Verbo do fato vale como substantivo e entra pela posição: "STF condena Bolsonaro" busca
        # "condena" no lugar de "prisao" ou "trama", que aparecem em qualquer notícia do caso
        if word_class in FACT_VERB_CLASSES and normalize_text(word).startswith(self.fact_verb_stems):
            return NOUN_RANK
        return self._class_rank(word_class)

    @staticmethod
    def _class_rank(word_class: str) -> int:
        if word_class in NOUN_CLASSES:
            return NOUN_RANK
        return VERB_RANK if word_class in VERB_CLASSES else OTHER_WORD_RANK

    def period_around(self, reference_date: date) -> Period:
        # A mesma janela em que uma reportagem pode sustentar a afirmação
        window = timedelta(days=self.settings.evidence_support_window_days)
        return reference_date - window, reference_date + window

    def _search_provider(
        self,
        provider: SearchProvider,
        keywords: list[str],
        period: Period | None = None,
        extra: list[str] | None = None,
    ) -> list[SearchResult]:
        results = self._search_keywords(provider, keywords)
        if extra:
            known = {r.url for r in results}
            limit = self.settings.verification_max_results
            results += [r for r in provider.client.search(" ".join(extra), limit) if r.url not in known]
        # Eventos antigos ficam enterrados por matérias recentes com os mesmos nomes ("Croácia" na Copa
        # de 2026 tomava o lugar da eliminação de 2022). A busca sem data continua: checagens de boatos
        # que voltam a circular são de anos antes da notícia
        if period is not None and provider.client.supports_period:
            known = {r.url for r in results}
            query = " ".join(keywords)
            limit = self.settings.verification_max_results
            results += [r for r in provider.client.search(query, limit, period) if r.url not in known]
        return results

    def _search_keywords(self, provider: SearchProvider, keywords: list[str]) -> list[SearchResult]:
        limit = self.settings.verification_max_results
        results = provider.client.search(" ".join(keywords), limit)
        # A busca com todas as palavras pode não trazer nada ou trazer só outro assunto
        # (ex.: um ato em Curitiba para "rafael camara contratado formula", que só cita os
        # nomes no resumo): tenta de novo com as palavras mais específicas e junta os resultados
        specific = keywords[: self.settings.verification_min_keywords]
        if len(keywords) > len(specific) and not any(mentions_all(r, specific) for r in results):
            known = {r.url for r in results}
            results += [r for r in provider.client.search(" ".join(specific), limit) if r.url not in known]
        return results


def is_number_like(token: str) -> bool:
    """Datas, placares e valores ("14/11/2026", "3x2") não ajudam a busca; "cop30" e "covid-19" ajudam."""
    digits = sum(char.isdigit() for char in token)
    return digits > 0 and digits >= sum(char.isalpha() for char in token)


def mentions_all(result: SearchResult, keywords: list[str]) -> bool:
    # Só o título: o resumo costuma citar nomes de passagem
    tokens = set(tokenize(normalize_text(result.title)))
    return all(keyword in tokens for keyword in keywords)
