"""Busca de parâmetros: testa combinações nos casos de treinamento e guarda sempre a melhor.

Começa da melhor combinação já guardada (ou dos valores atuais do Settings) e testa, um parâmetro
por vez, cada valor da lista; repete as rodadas enquanto alguma mudança melhorar. Uma combinação
só substitui a melhor quando:

1. no treinamento, tem menos erros graves (fato verdadeiro marcado como falso ou boato como
   verdadeiro) ou, com o mesmo número, acerta mais (somando os modos sem data e com a data do evento);
2. na validação, não tem mais erros graves nem acerta menos que a melhor atual.

Mais erros graves nunca são aceitos, por mais que a combinação acerte.

A regra 2 barra a combinação que só melhora no treinamento (feita sob medida para ele). Da validação
só os totais são usados; o teste nunca entra aqui.

    python -m training.tune_parameters                 # sem internet: só parâmetros depois da busca
    python -m training.tune_parameters --rounds 5
    python -m training.tune_parameters --with-search   # inclui o número de palavras-chave (usa internet)

Resultado em training/tuning/best_params.json (a melhor) e training/tuning/history.jsonl (todas as
tentativas). A API não muda sozinha: o script mostra as linhas do .env para aplicar a melhor.
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from app.core.config import Settings, get_settings
from app.services.verification_service import VerificationService
from tests.claim_dataset import TRAIN_SPLIT, VALIDATION_SPLIT, ClaimCase, load_cases
from training.entailment_cache import EntailmentStore
from training.evaluate_verification import (
    CACHE_PATH,
    ENTAILMENT_CACHE_PATH,
    CacheMode,
    CaseOutcome,
    SearchStore,
    build_service,
    entailment_model_id,
    evaluate,
)

TUNING_DIR = Path(__file__).resolve().parent / "tuning"
BEST_PATH = TUNING_DIR / "best_params.json"
HISTORY_PATH = TUNING_DIR / "history.jsonl"
DEFAULT_ROUNDS = 3

# Valores testados para cada parâmetro (nomes iguais aos do Settings). Só etapas depois da busca:
# mudam a leitura das evidências e a decisão, não as consultas, então rodam sem internet
SEARCH_SPACE: dict[str, list[Any]] = {
    "nli_min_entailment": [0.4, 0.5, 0.6, 0.7, 0.8],
    "decision_min_relevance": [0.5, 0.6, 0.7, 0.8],
    "evidence_min_relevance": [0.3, 0.4, 0.5, 0.6],
    "evidence_min_overlap": [1, 2, 3],
    "evidence_identical_relevance": [0.85, 0.9, 0.95, 1.0],
    "evidence_support_window_days": [30, 60, 120, 365, 730],
    "decision_min_supporting_sources": [1, 2],
    "decision_min_contradicting_sources": [1, 2],
}
# Mudam as consultas: na primeira vez cada combinação faz buscas novas na internet
KEYWORD_SPACE: dict[str, list[Any]] = {
    "verification_max_keywords": [3, 4, 5],
    "verification_min_keywords": [2, 3],
}


@dataclass(frozen=True)
class Score:
    cases: int
    correct: int
    serious: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.cases if self.cases else 0.0


@dataclass(frozen=True)
class Trial:
    params: dict[str, Any]
    train: Score
    validation: Score | None = None


def combine(outcomes_by_mode: list[list[CaseOutcome]]) -> Score:
    """Soma os modos (sem data e com data): a combinação precisa servir aos dois."""
    flat = [outcome for outcomes in outcomes_by_mode for outcome in outcomes]
    return Score(len(flat), sum(o.correct for o in flat), sum(o.serious_error for o in flat))


def is_better(candidate: Score, best: Score) -> bool:
    """Segurança primeiro: menos erros graves sempre ganha; com o mesmo número, ganha quem acerta mais.

    Uma combinação com mais erros graves nunca é aceita, por mais que acerte.
    """
    if candidate.serious != best.serious:
        return candidate.serious < best.serious
    return candidate.correct > best.correct


def passes_validation(candidate: Score, best: Score | None) -> bool:
    """Na validação a combinação não pode piorar em nada: nem mais erros graves, nem menos acertos."""
    if best is None:
        return True
    return candidate.serious <= best.serious and candidate.correct >= best.correct


def apply_settings(service: VerificationService, settings: Settings) -> None:
    """Troca os parâmetros do serviço já montado, sem recarregar modelos (é o que deixa cada tentativa rápida)."""
    for component in (service, *vars(service).values()):
        if hasattr(component, "settings"):
            component.settings = settings


class Tuner:
    def __init__(self, service: VerificationService, base: Settings, cases: dict[str, list[ClaimCase]], store):
        self.service = service
        self.base = base
        self.cases = cases
        self.store = store

    def score(self, params: dict[str, Any], split: str) -> Score:
        apply_settings(self.service, self.base.model_copy(update=params))
        modes = [evaluate(self.service, self.store, self.cases[split], dated, quiet=True) for dated in (False, True)]
        return combine(modes)

    def trial(self, params: dict[str, Any], best: Trial | None) -> tuple[Trial, bool]:
        """Mede no treinamento; só olha a validação se o treinamento melhorou."""
        train = self.score(params, TRAIN_SPLIT)
        if best is not None and not is_better(train, best.train):
            return Trial(params, train), False
        validation = self.score(params, VALIDATION_SPLIT)
        accepted = passes_validation(validation, best.validation if best else None)
        return Trial(params, train, validation), accepted


def coordinate_search(tuner: Tuner, start: Trial, space: dict[str, list[Any]], rounds: int, on_trial) -> Trial:
    """Um parâmetro por vez, cada valor da lista; repete enquanto alguma rodada melhorar."""
    best = start
    for round_number in range(1, rounds + 1):
        improved = False
        for name, values in space.items():
            for value in values:
                if best.params.get(name) == value:
                    continue
                trial, accepted = tuner.trial({**best.params, name: value}, best)
                on_trial(round_number, trial, accepted)
                if accepted:
                    best, improved = trial, True
        if not improved:
            break
    return best


def load_best(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_best(path: Path, best: Trial, baseline: Trial) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = {
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "params": best.params,
        "train": asdict(best.train),
        "validation": asdict(best.validation) if best.validation else None,
        "baseline": {
            "params": baseline.params,
            "train": asdict(baseline.train),
            "validation": asdict(baseline.validation) if baseline.validation else None,
        },
    }
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_history(path: Path, round_number: int, trial: Trial, accepted: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "round": round_number,
        "params": trial.params,
        "train": asdict(trial.train),
        "validation": asdict(trial.validation) if trial.validation else None,
        "accepted": accepted,
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def describe(score: Score | None) -> str:
    if score is None:
        return "-"
    return f"{score.correct}/{score.cases} ({score.accuracy:.1%}), graves={score.serious}"


def env_lines(params: dict[str, Any], current: Settings) -> list[str]:
    """Linhas do .env para aplicar a melhor combinação (só o que difere do atual)."""
    return [f"{name.upper()}={value}" for name, value in params.items() if getattr(current, name) != value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Busca de parâmetros com os casos de treinamento e validação.")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS, help="rodadas no máximo")
    parser.add_argument("--with-search", action="store_true", help="inclui o número de palavras-chave (internet)")
    parser.add_argument("--from-settings", action="store_true", help="ignora a melhor guardada e parte do Settings")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    space = {**SEARCH_SPACE, **(KEYWORD_SPACE if args.with_search else {})}
    settings = get_settings()
    mode: CacheMode = "reuse" if args.with_search else "offline"
    store = SearchStore(CACHE_PATH, mode)
    entailments = EntailmentStore(ENTAILMENT_CACHE_PATH, entailment_model_id())
    http_client = httpx.Client(
        timeout=settings.verification_timeout_seconds,
        follow_redirects=True,
        max_redirects=settings.verification_max_redirects,
        headers={"User-Agent": settings.verification_user_agent},
    )
    service = build_service(store, http_client, entailments)
    cases = {split: load_cases(split) for split in (TRAIN_SPLIT, VALIDATION_SPLIT)}
    tuner = Tuner(service, settings, cases, store)

    current = {name: getattr(settings, name) for name in space}
    saved = None if args.from_settings else load_best(BEST_PATH)
    start_params = {**current, **(saved["params"] if saved else {})}
    try:
        baseline = Trial(current, tuner.score(current, TRAIN_SPLIT), tuner.score(current, VALIDATION_SPLIT))
        start = Trial(start_params, tuner.score(start_params, TRAIN_SPLIT), tuner.score(start_params, VALIDATION_SPLIT))
        print(f"Atual (Settings): treinamento {describe(baseline.train)} | validação {describe(baseline.validation)}")
        print(f"Ponto de partida: treinamento {describe(start.train)} | validação {describe(start.validation)}")

        def on_trial(round_number: int, trial: Trial, accepted: bool) -> None:
            append_history(HISTORY_PATH, round_number, trial, accepted)
            changed = {k: v for k, v in trial.params.items() if start_params.get(k) != v}
            mark = "MELHOR" if accepted else "      "
            print(
                f"[rodada {round_number}] {mark} treinamento {describe(trial.train)} "
                f"| validação {describe(trial.validation)} | {changed}",
                file=sys.stderr,
            )
            if accepted:
                save_best(BEST_PATH, trial, baseline)

        best = coordinate_search(tuner, start, space, args.rounds, on_trial)
    finally:
        store.save()
        entailments.save()
        if service.search_service is not None:
            service.search_service.close()
        http_client.close()

    if best is start and saved is None:
        save_best(BEST_PATH, best, baseline)
    print(f"\nMelhor: treinamento {describe(best.train)} | validação {describe(best.validation)}")
    print(f"Guardada em {BEST_PATH}; tentativas em {HISTORY_PATH}")
    lines = env_lines(best.params, settings)
    print("\nPara aplicar na API, coloque no .env:" if lines else "\nA melhor combinação é a atual do Settings.")
    for line in lines:
        print(f"  {line}")


if __name__ == "__main__":
    main()
