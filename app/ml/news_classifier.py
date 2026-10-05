"""Modelo de NLP/ML (TF-IDF + Regressão Logística) que estima P(falsa)."""

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.services.text_processing_service import normalize_text

logger = logging.getLogger(__name__)

# Convenção de rótulos: 1 = falsa, 0 = verdadeira
FAKE_LABEL = 1


class NewsClassifier:
    def __init__(self, pipeline: Pipeline | None = None):
        self.pipeline = pipeline

    @property
    def is_trained(self) -> bool:
        return self.pipeline is not None

    @staticmethod
    def build_pipeline(stopwords: Iterable[str]) -> Pipeline:
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                preprocessor=normalize_text,
                stop_words=sorted(stopwords),
                ngram_range=(1, 2),
                min_df=2,
                max_features=50_000,
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])

    def train(self, texts: Sequence[str], labels: Sequence[int], stopwords: Iterable[str]) -> "NewsClassifier":
        pipeline = self.build_pipeline(stopwords)
        pipeline.fit(list(texts), list(labels))
        self.pipeline = pipeline
        return self

    def predict_fake_probability(self, text: str) -> float:
        if self.pipeline is None:
            raise RuntimeError("Modelo não treinado.")
        classes = list(self.pipeline.classes_)
        probabilities = self.pipeline.predict_proba([text])[0]
        return float(probabilities[classes.index(FAKE_LABEL)])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: Path) -> "NewsClassifier":
        # joblib usa pickle: carregar apenas arquivos gerados pelo próprio projeto
        if not path.exists():
            logger.warning("Modelo não encontrado em %s; usando apenas a análise heurística.", path)
            return cls()
        return cls(joblib.load(path))
