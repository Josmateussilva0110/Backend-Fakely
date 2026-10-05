"""Busca a notícia nos sites confiáveis e identifica checagens correspondentes."""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass

import httpx

from app.core.config import Settings
from app.integrations.folha_search_client import FolhaSearchClient
from app.integrations.search_client import HttpFetcher, SearchClient, SearchError, SearchResult
from app.integrations.wordpress_search_client import WordpressSearchClient
from app.services.text_processing_service import normalize_text, tokenize
from app.services.verification_config_service import (
    SourceType, VerdictPatterns, Verdict, VerificationConfig, VerificationSourceConfig,
)

logger = logging.getLogger(__name__)

# Prefixo usado como "radical" simples (ex.: eficacia/eficaz, vacina/vacinas)
STEM_LENGTH = 5
MIN_KEYWORD_LENGTH = 4
MIN_TOKEN_LENGTH = 3
VERDICT_SCAN_LENGTH = 200
_WORD_PATTERN = re.compile(r"\w+(?:[-']\w+)*")


@dataclass(frozen=True)
class VerificationMatch:
    source_key: str
    source_name: str
    source_type: SourceType
    title: str
    url: str
    excerpt: str
    published_at: str | None
    similarity: float
    verdict: Verdict | None


@dataclass(frozen=True)
class VerificationResult:
    keywords: tuple[str, ...]
    matches: tuple[VerificationMatch, ...]
    sources_checked: tuple[str, ...]
    sources_failed: tuple[str, ...]

    @property
    def has_failures(self) -> bool:
        return bool(self.sources_failed)


@dataclass(frozen=True)
class _Source:
    config: VerificationSourceConfig
    client: SearchClient


class SourceVerificationService:
    def __init__(
        self,
        settings: Settings,
        config: VerificationConfig,
        stopwords: frozenset[str],
        sources: list[_Source],
    ):
        self.settings = settings
        self.verdict_patterns: VerdictPatterns = config.verdict_patterns
        self.stopwords = stopwords
        self.sources = sources
        self.executor = ThreadPoolExecutor(max_workers=max(len(sources), 1), thread_name_prefix="verification")

    @classmethod
    def build(
        cls, settings: Settings, config: VerificationConfig, stopwords: frozenset[str], http_client: httpx.Client
    ) -> "SourceVerificationService":
        fetcher = HttpFetcher(http_client, settings.verification_max_response_bytes)
        client_types = {"wordpress": WordpressSearchClient, "folha": FolhaSearchClient}
        sources = [
            _Source(source, client_types[source.client](fetcher, str(source.search_url), source.allowed_domains))
            for source in config.sources
        ]
        return cls(settings, config, stopwords, sources)

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

    def verify(self, text: str, title: str | None = None) -> VerificationResult | None:
        keywords = self.extract_keywords(text, title)
        if len(keywords) < self.settings.verification_min_keywords:
            return None

        user_stems = self._stems(f"{title or ''} {text}")
        futures = {self.executor.submit(self._search_source, source, keywords): source for source in self.sources}
        done, _ = wait(futures, timeout=self.settings.verification_total_timeout_seconds)

        matches: list[VerificationMatch] = []
        checked, failed = [], []
        for future, source in futures.items():
            if future not in done:
                future.cancel()
                failed.append(source.config.name)
                logger.warning("Tempo esgotado ao consultar %s", source.config.name)
                continue
            try:
                results = future.result()
            except SearchError as exc:
                failed.append(source.config.name)
                logger.warning("Falha ao consultar %s: %s", source.config.name, exc)
                continue
            checked.append(source.config.name)
            matches.extend(self._match_results(source.config, results, user_stems))

        matches.sort(key=lambda m: m.similarity, reverse=True)
        return VerificationResult(
            keywords=tuple(keywords),
            matches=tuple(matches),
            sources_checked=tuple(checked),
            sources_failed=tuple(failed),
        )

    def extract_keywords(self, text: str, title: str | None = None) -> list[str]:
        """Palavras mais específicas do título, completadas pelo texto.

        Retorna em ordem de especificidade: nomes próprios (inicial maiúscula) primeiro,
        depois as palavras mais longas.
        """
        candidates: dict[str, bool] = {}
        for chunk in (title, text):
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

    def _search_source(self, source: _Source, keywords: list[str]) -> list[SearchResult]:
        limit = self.settings.verification_max_results
        results = source.client.search(" ".join(keywords), limit)
        # A busca do WordPress exige todas as palavras; tenta de novo com as mais específicas
        fallback_size = self.settings.verification_min_keywords
        if not results and len(keywords) > fallback_size:
            results = source.client.search(" ".join(keywords[:fallback_size]), limit)
        return results

    def _match_results(
        self, source: VerificationSourceConfig, results: list[SearchResult], user_stems: set[str]
    ) -> list[VerificationMatch]:
        matches = []
        for result in results:
            similarity, overlap = self.compute_similarity(result.title, user_stems)
            if overlap < self.settings.verification_min_overlap or similarity < self.settings.verification_min_similarity:
                continue
            matches.append(VerificationMatch(
                source_key=source.key,
                source_name=source.name,
                source_type=source.type,
                title=result.title,
                url=result.url,
                excerpt=result.excerpt[: self.settings.verification_excerpt_max_length],
                published_at=result.published_at.isoformat() if result.published_at else None,
                similarity=round(similarity, 3),
                verdict=self.detect_verdict(source, result) if source.type == "fact_checker" else None,
            ))
        return matches

    def compute_similarity(self, result_title: str, user_stems: set[str]) -> tuple[float, int]:
        """Fração do título do resultado coberta pelo texto do usuário."""
        title = normalize_text(result_title)
        # Remove o veredito ("é falso que...") para comparar só o assunto
        for pattern in (*self.verdict_patterns.false, *self.verdict_patterns.true):
            title = title.replace(pattern, " ")
        title_stems = self._stems(title)
        if not title_stems:
            return 0.0, 0
        overlap = len(title_stems & user_stems)
        return overlap / len(title_stems), overlap

    def detect_verdict(self, source: VerificationSourceConfig, result: SearchResult) -> Verdict | None:
        content = normalize_text(f"{result.title} {result.excerpt[:VERDICT_SCAN_LENGTH]}")
        # "não é verdade" contém "é verdade": padrões de falso são verificados primeiro
        if any(re.search(rf"(?<!\w){re.escape(p)}", content) for p in self.verdict_patterns.false):
            return "false"
        if any(re.search(rf"(?<!\w){re.escape(p)}", content) for p in self.verdict_patterns.true):
            return "true"
        return source.default_verdict

    def _stems(self, text: str) -> set[str]:
        return {
            token[:STEM_LENGTH]
            for token in tokenize(normalize_text(text))
            if len(token) >= MIN_TOKEN_LENGTH and token not in self.stopwords
        }
