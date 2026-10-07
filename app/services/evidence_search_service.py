"""Pesquisa a afirmação nos provedores de busca e checagem, em paralelo."""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from app.core.config import Settings
from app.integrations.folha_search_client import FolhaSearchClient
from app.integrations.google_fact_check_client import GoogleFactCheckClient
from app.integrations.search_client import HttpFetcher, SearchClient, SearchError, SearchResult
from app.integrations.wordpress_search_client import WordpressSearchClient
from app.services.text_processing_service import normalize_text
from app.services.verification_config_service import VerificationConfig, VerificationSourceConfig

logger = logging.getLogger(__name__)

MIN_KEYWORD_LENGTH = 4
_WORD_PATTERN = re.compile(r"\w+(?:[-']\w+)*")


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
    def __init__(self, settings: Settings, stopwords: frozenset[str], providers: list[SearchProvider]):
        self.settings = settings
        self.stopwords = stopwords
        self.providers = providers
        self.executor = ThreadPoolExecutor(max_workers=max(len(providers), 1), thread_name_prefix="search")

    @classmethod
    def build(
        cls, settings: Settings, config: VerificationConfig, stopwords: frozenset[str], http_client: httpx.Client
    ) -> "EvidenceSearchService":
        fetcher = HttpFetcher(http_client, settings.verification_max_response_bytes)
        providers = []
        for source in config.sources:
            client = cls._build_client(settings, source, fetcher)
            if client is not None:
                providers.append(SearchProvider(source, client))
        return cls(settings, stopwords, providers)

    @staticmethod
    def _build_client(
        settings: Settings, source: VerificationSourceConfig, fetcher: HttpFetcher
    ) -> SearchClient | None:
        url, domains = str(source.search_url), source.allowed_domains
        if source.client == "wordpress":
            return WordpressSearchClient(fetcher, url, domains)
        if source.client == "folha":
            return FolhaSearchClient(fetcher, url, domains)
        # Google Fact Check só é usado quando há chave configurada
        if settings.google_fact_check_api_key is None:
            return None
        return GoogleFactCheckClient(
            fetcher, url, domains, settings.google_fact_check_api_key, settings.google_fact_check_language_code
        )

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
            accessed_at=datetime.now(timezone.utc),
        )

    def extract_keywords(self, claim: str, title: str | None = None) -> list[str]:
        """Palavras mais específicas da afirmação, completadas pelo título.

        Retorna em ordem de especificidade: nomes próprios (inicial maiúscula) primeiro,
        depois as palavras mais longas.
        """
        candidates: dict[str, bool] = {}
        for chunk in (claim, title):
            for word in _WORD_PATTERN.findall(chunk or ""):
                token = normalize_text(word)
                if (
                    len(token) < MIN_KEYWORD_LENGTH or token.isdigit()
                    or token in self.stopwords or token in candidates
                ):
                    continue
                candidates[token] = word[0].isupper()
            if len(candidates) >= self.settings.verification_max_keywords:
                break

        ranked = sorted(candidates, key=lambda t: (candidates[t], len(t)), reverse=True)
        return ranked[: self.settings.verification_max_keywords]

    def _search_provider(self, provider: SearchProvider, keywords: list[str]) -> list[SearchResult]:
        limit = self.settings.verification_max_results
        results = provider.client.search(" ".join(keywords), limit)
        # Buscas que exigem todas as palavras: tenta de novo com as mais específicas
        fallback_size = self.settings.verification_min_keywords
        if not results and len(keywords) > fallback_size:
            results = provider.client.search(" ".join(keywords[:fallback_size]), limit)
        return results
