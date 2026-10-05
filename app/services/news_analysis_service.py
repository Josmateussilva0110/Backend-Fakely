"""Pipeline: pré-processamento → características → modelo/heurística → verificação → classificação."""

from dataclasses import dataclass

from app.core.cache import ThreadSafeTTLCache, build_cache_key
from app.core.config import Settings
from app.ml.news_classifier import NewsClassifier
from app.models.news_model import Classification
from app.services.analysis_rules_service import AnalysisRules
from app.services.source_verification_service import SourceVerificationService, VerificationResult
from app.services.text_processing_service import TextFeatures, extract_features, preprocess_text

METHOD_NONE = "none"
METHOD_HEURISTIC = "heuristic"
METHOD_MODEL = "model"
METHOD_VERIFICATION = "verification"


@dataclass(frozen=True)
class AnalysisResult:
    classification: Classification
    confidence: float | None
    probability_fake: float | None
    features: TextFeatures
    evidence: tuple[str, ...]
    method: str
    verification: VerificationResult | None = None


class NewsAnalysisService:
    def __init__(
        self,
        settings: Settings,
        rules: AnalysisRules,
        classifier: NewsClassifier,
        verification_service: SourceVerificationService | None = None,
    ):
        self.settings = settings
        self.rules = rules
        self.classifier = classifier
        self.verification_service = verification_service
        # Mesmo conteúdo gera a mesma análise; evita reprocessar e repetir buscas externas
        self.cache = ThreadSafeTTLCache(
            settings.analysis_cache_max_entries, settings.analysis_cache_ttl_seconds
        )

    def analyze(
        self, text: str, title: str | None = None, url: str | None = None, source: str | None = None
    ) -> AnalysisResult:
        cache_key = build_cache_key(text, title, url, source)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        result = self._run_analysis(text, title, url, source)
        # Não guarda resultado incompleto (site fora do ar), para tentar de novo depois
        if result.verification is None or not result.verification.has_failures:
            self.cache.set(cache_key, result)
        return result

    def _run_analysis(
        self, text: str, title: str | None, url: str | None, source: str | None
    ) -> AnalysisResult:
        processed = preprocess_text(text, self.rules.stopwords)
        features = extract_features(processed, self.rules, title=title, url=url, source=source)

        # RN07: texto curto demais não permite classificação confiável
        if features.word_count < self.settings.min_words:
            return AnalysisResult(
                classification=Classification.INCONCLUSIVE,
                confidence=None,
                probability_fake=None,
                features=features,
                evidence=(f"Texto muito curto (mínimo de {self.settings.min_words} palavras)",),
                method=METHOD_NONE,
            )

        probability_fake, evidence = self.compute_heuristic_score(features)
        methods = [METHOD_HEURISTIC]

        if self.classifier.is_trained:
            model_input = f"{title}\n{text}" if title else text
            model_probability = self.classifier.predict_fake_probability(model_input)
            weight = self.settings.model_weight
            probability_fake = weight * model_probability + (1 - weight) * probability_fake
            methods.insert(0, METHOD_MODEL)

        verification = None
        if self.verification_service is not None:
            verification = self.verification_service.verify(text, title)
            if verification is not None:
                probability_fake = self.apply_verification(probability_fake, verification, evidence)
                if verification.sources_checked:
                    methods.append(METHOD_VERIFICATION)

        w = self.rules.weights
        probability_fake = min(max(probability_fake, w.min_probability), w.max_probability)

        if probability_fake >= 0.5:
            classification, confidence = Classification.LIKELY_FALSE, probability_fake
        else:
            classification, confidence = Classification.LIKELY_TRUE, 1 - probability_fake

        if confidence < self.settings.confidence_threshold:
            classification = Classification.INCONCLUSIVE

        return AnalysisResult(
            classification=classification,
            confidence=round(confidence, 3),
            probability_fake=round(probability_fake, 3),
            features=features,
            evidence=tuple(evidence),
            method="+".join(methods),
            verification=verification,
        )

    def apply_verification(
        self, probability_fake: float, verification: VerificationResult, evidence: list[str]
    ) -> float:
        """Ajusta P(falsa) com as checagens e notícias encontradas nos sites confiáveis."""
        w = self.rules.weights
        min_similarity = self.settings.verification_fact_check_min_similarity
        # Matches já vêm ordenados por similaridade
        decisive_check = next(
            (m for m in verification.matches
             if m.source_type == "fact_checker" and m.verdict and m.similarity >= min_similarity),
            None,
        )
        news_match = next((m for m in verification.matches if m.source_type == "news_outlet"), None)

        # Checagem decide só se for mais parecida com o texto que a matéria encontrada no veículo
        if decisive_check and (news_match is None or decisive_check.similarity >= news_match.similarity):
            if decisive_check.verdict == "false":
                probability_fake = max(probability_fake, w.fact_check_false_min_probability)
                evidence.append(
                    f"Checagem de {decisive_check.source_name} aponta como falso: \"{decisive_check.title}\""
                )
            else:
                probability_fake = min(probability_fake, w.fact_check_true_max_probability)
                evidence.append(
                    f"Checagem de {decisive_check.source_name} aponta como verdadeiro: \"{decisive_check.title}\""
                )
        elif news_match:
            probability_fake += w.news_outlet_match
            evidence.append(f"Notícia semelhante publicada por {news_match.source_name}: \"{news_match.title}\"")

        if verification.sources_checked and not verification.matches:
            evidence.append(
                "Nenhuma publicação correspondente encontrada em: " + ", ".join(verification.sources_checked)
            )
        if verification.sources_failed:
            evidence.append("Não foi possível consultar: " + ", ".join(verification.sources_failed))
        return probability_fake

    def compute_heuristic_score(self, features: TextFeatures) -> tuple[float, list[str]]:
        """Estima P(falsa) a partir das características textuais e da fonte."""
        w = self.rules.weights
        score = w.baseline
        evidence: list[str] = []

        if features.sensational_terms:
            score += min(w.sensational_per_term * len(features.sensational_terms), w.sensational_max)
            evidence.append(f"Termos sensacionalistas: {', '.join(features.sensational_terms)}")
        if features.alarmist_terms:
            score += min(w.alarmist_per_term * len(features.alarmist_terms), w.alarmist_max)
            evidence.append(f"Afirmações alarmistas: {', '.join(features.alarmist_terms)}")

        excess_exclamation = features.exclamation_count >= w.excess_exclamation_min
        if features.repeated_punctuation_count:
            score += w.repeated_punctuation
            evidence.append("Uso de pontuação repetida (ex.: '!!!', '?!')")
        elif excess_exclamation:
            score += w.excess_exclamation
            evidence.append("Excesso de pontos de exclamação")

        excess_uppercase = features.uppercase_ratio > w.uppercase_ratio_min
        if excess_uppercase:
            score += w.uppercase
            evidence.append("Uso excessivo de letras maiúsculas")

        if features.trusted_source:
            score += w.trusted_source
            evidence.append(f"Fonte reconhecida: {features.domain}")
        elif features.domain is None:
            evidence.append("Fonte não informada")
        else:
            evidence.append(f"Fonte não reconhecida: {features.domain}")

        has_text_signals = (
            features.sensational_terms or features.alarmist_terms
            or features.repeated_punctuation_count or excess_exclamation or excess_uppercase
        )
        if not has_text_signals:
            # Ausência de sinais é evidência fraca; sozinha não basta para concluir
            score += w.no_text_signals

        return min(max(score, w.min_probability), w.max_probability), evidence
