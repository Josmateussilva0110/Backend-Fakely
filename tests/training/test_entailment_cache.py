import pytest

from tests.fakes import FakeEntailmentModel
from training.entailment_cache import CachedEntailmentModel, EntailmentStore

MODEL_ID = "modelo@rev1"


class CountingModel(FakeEntailmentModel):
    def __init__(self, scores: dict):
        super().__init__(scores)
        self.premises_scored: list[str] = []

    def entailment(self, premises: list[str], hypothesis: str) -> list[float]:
        self.premises_scored += premises
        return super().entailment(premises, hypothesis)


def test_recorded_scores_are_reused_without_loading_the_model(tmp_path):
    path = tmp_path / "scores.json"
    model = CountingModel({"manchete a": 0.9, "manchete b": 0.2})
    store = EntailmentStore(path, MODEL_ID)
    assert CachedEntailmentModel(store, lambda: model).entailment(["manchete a", "manchete b"], "afirmação") == [
        0.9,
        0.2,
    ]
    store.save()

    def no_model():
        raise AssertionError("o modelo não deveria ser carregado")

    replay = CachedEntailmentModel(EntailmentStore(path, MODEL_ID), no_model)
    assert replay.entailment(["manchete b", "manchete a"], "afirmação") == [0.2, 0.9]


def test_only_new_pairs_go_to_the_model(tmp_path):
    model = CountingModel({"nova": 0.7})
    store = EntailmentStore(tmp_path / "scores.json", MODEL_ID)
    store.put("gravada", "afirmação", 0.4)
    scores = CachedEntailmentModel(store, lambda: model).entailment(["gravada", "nova"], "afirmação")
    assert scores == [0.4, 0.7]
    assert model.premises_scored == ["nova"]
    assert store.computed == 2  # a gravada manualmente e a nova


def test_same_premise_with_other_hypothesis_is_a_new_pair(tmp_path):
    model = CountingModel({("manchete", "outra afirmação"): 0.1})
    store = EntailmentStore(tmp_path / "scores.json", MODEL_ID)
    store.put("manchete", "afirmação", 0.9)
    assert CachedEntailmentModel(store, lambda: model).entailment(["manchete"], "outra afirmação") == [0.1]


def test_scores_of_another_model_version_are_discarded(tmp_path):
    path = tmp_path / "scores.json"
    store = EntailmentStore(path, MODEL_ID)
    store.put("manchete", "afirmação", 0.9)
    store.save()
    assert EntailmentStore(path, "modelo@rev2").get("manchete", "afirmação") is None
    assert EntailmentStore(path, MODEL_ID).get("manchete", "afirmação") == 0.9


def test_missing_model_is_reported(tmp_path):
    cached = CachedEntailmentModel(EntailmentStore(tmp_path / "scores.json", MODEL_ID), lambda: None)
    with pytest.raises(RuntimeError):
        cached.entailment(["nova"], "afirmação")
