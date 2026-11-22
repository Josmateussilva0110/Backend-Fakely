"""Notas do modelo NLI gravadas por par (premissa, hipótese), para a avaliação levar segundos.

O modelo é determinístico: a mesma manchete comparada com a mesma afirmação dá sempre a mesma nota.
Gravar as notas não muda o resultado, só evita recalcular. O modelo real só é carregado quando
aparece um par ainda sem nota.
"""

import json
from collections.abc import Callable
from pathlib import Path
from threading import Lock

from app.ml.entailment_model import EntailmentScorer

# Arredondamento suficiente: as regras comparam com limiares de duas casas (0,6)
SCORE_DIGITS = 6


class EntailmentStore:
    """Notas gravadas, válidas só para a mesma versão do modelo."""

    def __init__(self, path: Path, model_id: str):
        self.path = path
        self.model_id = model_id
        self.lock = Lock()
        self.scores: dict[str, dict[str, float]] = {}
        self.computed = 0
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            # Outro modelo ou outra revisão dá outras notas: começa do zero
            if saved.get("model") == model_id:
                self.scores = saved["scores"]

    def get(self, premise: str, hypothesis: str) -> float | None:
        with self.lock:
            return self.scores.get(hypothesis, {}).get(premise)

    def put(self, premise: str, hypothesis: str, score: float) -> None:
        with self.lock:
            self.scores.setdefault(hypothesis, {})[premise] = round(score, SCORE_DIGITS)
            self.computed += 1

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            content = json.dumps({"model": self.model_id, "scores": self.scores}, ensure_ascii=False, sort_keys=True)
        self.path.write_text(content + "\n", encoding="utf-8")


class CachedEntailmentModel:
    """Mesma interface do modelo; consulta as notas gravadas e só chama o modelo nos pares novos."""

    def __init__(self, store: EntailmentStore, load_model: Callable[[], EntailmentScorer | None]):
        self.store = store
        self.load_model = load_model
        self.model: EntailmentScorer | None = None
        self.load_lock = Lock()

    def entailment(self, premises: list[str], hypothesis: str) -> list[float]:
        scores = [self.store.get(premise, hypothesis) for premise in premises]
        missing = [premise for premise, score in zip(premises, scores, strict=True) if score is None]
        if missing:
            computed = dict(zip(missing, self._model().entailment(missing, hypothesis), strict=True))
            for premise, score in computed.items():
                self.store.put(premise, hypothesis, score)
            scores = [computed[p] if s is None else s for p, s in zip(premises, scores, strict=True)]
        return [round(score, SCORE_DIGITS) for score in scores]

    def _model(self) -> EntailmentScorer:
        with self.load_lock:
            if self.model is None:
                self.model = self.load_model()
                if self.model is None:
                    raise RuntimeError("Não foi possível carregar o modelo NLI para calcular as notas novas.")
        return self.model
