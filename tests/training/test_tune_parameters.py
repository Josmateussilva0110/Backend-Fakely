import json

from app.services.analysis_rules_service import load_analysis_rules
from app.services.source_registry_service import load_source_registry
from app.services.verification_service import VerificationService
from training.tune_parameters import (
    Score,
    Trial,
    append_history,
    apply_settings,
    coordinate_search,
    env_lines,
    is_better,
    passes_validation,
    save_best,
)


class FakeTuner:
    """Acerto definido por uma função dos parâmetros; validação separada para simular sob medida."""

    def __init__(self, train_fn, validation_fn):
        self.train_fn = train_fn
        self.validation_fn = validation_fn
        self.validations = 0

    def trial(self, params, best):
        train = self.train_fn(params)
        if best is not None and not is_better(train, best.train):
            return Trial(params, train), False
        self.validations += 1
        validation = self.validation_fn(params)
        return Trial(params, train, validation), passes_validation(validation, best.validation if best else None)


def test_more_serious_errors_never_win_even_with_more_hits():
    assert not is_better(Score(100, 90, serious=3), Score(100, 50, serious=2))
    assert is_better(Score(100, 51, 0), Score(100, 50, 0))
    assert not is_better(Score(100, 50, 0), Score(100, 50, 0))


def test_fewer_serious_errors_win_even_with_fewer_hits():
    assert is_better(Score(100, 40, serious=1), Score(100, 50, serious=2))


def test_same_serious_errors_compare_hits():
    # O ponto de partida pode já ter erros graves: a busca precisa conseguir avançar sem piorar
    assert is_better(Score(100, 75, serious=2), Score(100, 63, serious=2))


def test_validation_must_not_get_worse():
    best = Score(60, 30, 0)
    assert passes_validation(Score(60, 30, 0), best)
    assert not passes_validation(Score(60, 29, 0), best)
    assert not passes_validation(Score(60, 40, serious=1), best)


def test_search_keeps_the_best_and_rejects_overfitting():
    # "a" melhora treinamento e validação; "b" melhora só o treinamento (sob medida): fica de fora
    def train(params):
        return Score(100, 50 + 5 * (params["a"] == 2) + 10 * (params["b"] == 2), 0)

    def validation(params):
        return Score(50, 25 + 3 * (params["a"] == 2) - 4 * (params["b"] == 2), 0)

    tuner = FakeTuner(train, validation)
    start = Trial({"a": 1, "b": 1}, train({"a": 1, "b": 1}), validation({"a": 1, "b": 1}))
    accepted = []
    best = coordinate_search(tuner, start, {"a": [1, 2], "b": [1, 2]}, 3, lambda r, t, ok: accepted.append(ok))
    assert best.params == {"a": 2, "b": 1}
    assert accepted.count(True) == 1


def test_validation_is_only_measured_when_training_improves():
    tuner = FakeTuner(lambda p: Score(100, 50, 0), lambda p: Score(50, 25, 0))
    start = Trial({"a": 1}, Score(100, 50, 0), Score(50, 25, 0))
    coordinate_search(tuner, start, {"a": [1, 2, 3]}, 3, lambda *args: None)
    assert tuner.validations == 0


def test_apply_settings_reaches_every_component(settings):
    service = VerificationService(
        settings, load_analysis_rules(settings.analysis_rules_path), load_source_registry(settings.source_registry_path)
    )
    new = settings.model_copy(update={"nli_min_entailment": 0.42})
    apply_settings(service, new)
    components = [service, *vars(service).values()]
    assert all(c.settings is new for c in components if hasattr(c, "settings"))
    assert service.decision_service.settings.nli_min_entailment == 0.42


def test_best_and_history_are_saved(tmp_path):
    best = Trial({"a": 2}, Score(100, 60, 0), Score(50, 30, 0))
    save_best(tmp_path / "best.json", best, Trial({"a": 1}, Score(100, 50, 0), Score(50, 25, 0)))
    saved = json.loads((tmp_path / "best.json").read_text(encoding="utf-8"))
    assert saved["params"] == {"a": 2}
    assert saved["baseline"]["train"]["correct"] == 50

    append_history(tmp_path / "history.jsonl", 1, best, True)
    append_history(tmp_path / "history.jsonl", 1, best, False)
    lines = (tmp_path / "history.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["accepted"] for line in lines] == [True, False]


def test_env_lines_only_list_changed_values(settings):
    params = {"nli_min_entailment": 0.5, "decision_min_relevance": settings.decision_min_relevance}
    assert env_lines(params, settings) == ["NLI_MIN_ENTAILMENT=0.5"]
