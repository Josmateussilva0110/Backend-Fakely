"""Pesquisa a afirmação nos provedores de busca e checagem, em paralelo."""

import logging
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from app.core.config import Settings
from app.integrations.folha_search_client import FolhaSearchClient
from app.integrations.search_client import HttpFetcher, SearchClient, SearchError, SearchResult
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
    ):
        self.settings = settings
        self.stopwords = stopwords
        self.providers = providers
        self.acronyms = acronyms
        self.tagger = tagger
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
    ) -> "EvidenceSearchService":
        fetcher = HttpFetcher(http_client, settings.verification_max_response_bytes)
        providers = [SearchProvider(source, cls._build_client(source, fetcher)) for source in config.sources]
        return cls(settings, stopwords, providers, acronyms, tagger)

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

    def search(self, claim: str, title: str | None = None) -> SearchOutcome | None:
        keywords = self.extract_keywords(claim, title)
        if len(keywords) < self.settings.verification_min_keywords:
            return None

        futures = {self.executor.submit(self._search_provider, p, keywords): p for p in self.providers}
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
        verbos e depois as demais; dentro de cada grupo, as palavras mais longas.
        """
        candidates: dict[str, int] = {}
        for chunk in (claim, title):
            for word, rank in self._ranked_words(chunk or ""):
                token = normalize_text(word)
                if token and not is_number_like(token) and token not in self.stopwords and token not in candidates:
                    candidates[token] = rank
            if len(candidates) >= self.settings.verification_max_keywords:
                break

        ranked = sorted(candidates, key=lambda t: (candidates[t], len(t)), reverse=True)
        return ranked[: self.settings.verification_max_keywords]

    def _ranked_words(self, text: str) -> list[tuple[str, int]]:
        # Texto todo em maiúsculas engana o modelo ("TRUMP ANUNCIA" vira organização) e a regra da caixa
        shouting = is_shouting(text)
        if self.tagger is not None and not shouting:
            tagged = [(word, self._class_rank(word_class)) for word, word_class in self.tagger.tag(text)]
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

    @staticmethod
    def _class_rank(word_class: str) -> int:
        if word_class in NOUN_CLASSES:
            return NOUN_RANK
        return VERB_RANK if word_class in VERB_CLASSES else OTHER_WORD_RANK

    def _search_provider(self, provider: SearchProvider, keywords: list[str]) -> list[SearchResult]:
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
