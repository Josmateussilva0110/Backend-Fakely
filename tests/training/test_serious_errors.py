"""Trava dos erros graves: nenhuma mudança pode chamar mais fatos verdadeiros de falsos (ou o contrário).

Roda a verificação inteira com as buscas e as notas do modelo gravadas em training/cache, sem internet
e sem carregar o modelo. Os tetos são o resultado atual: ao corrigir um erro grave, baixe o teto.
"""

import httpx
import pytest

from app.core.config import get_settings
from tests.claim_dataset import TRAIN_SPLIT, VALIDATION_SPLIT, load_cases
from training.entailment_cache import CachedEntailmentModel, EntailmentStore
from training.evaluate_verification import (
    CACHE_PATH,
    ENTAILMENT_CACHE_PATH,
    SearchStore,
    build_service,
    entailment_model_id,
    evaluate,
)

# (grupo, com a data do evento): erros graves aceitos. O grupo de teste fica de fora: só ao fechar uma versão
MAX_SERIOUS_ERRORS = {
    (TRAIN_SPLIT, True): 1,
    (TRAIN_SPLIT, False): 1,
    (VALIDATION_SPLIT, True): 2,
    (VALIDATION_SPLIT, False): 3,
}
RECORD_COMMAND = "python -m training.evaluate_verification --split all [--event-date]"


def no_model():
    raise RuntimeError(f"Nota do modelo NLI sem gravação: rode {RECORD_COMMAND} para gravá-la.")


@pytest.fixture(scope="module")
def offline_service():
    store = SearchStore(CACHE_PATH, "offline")
    entailments = EntailmentStore(ENTAILMENT_CACHE_PATH, entailment_model_id())
    # Nenhuma requisição sai do teste: busca sem gravação vira erro, não consulta ao site
    http = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(599)))
    # Os testes desligam a busca e o modelo (conftest); aqui eles voltam, mas só com o que está gravado
    settings = get_settings().model_copy(update={"verification_enabled": True, "nli_enabled": True})
    service = build_service(store, http, entailments, settings)
    assert service.evidence_service is not None
    if service.evidence_service.entailment_model is not None:
        service.evidence_service.entailment_model = CachedEntailmentModel(entailments, no_model)
    yield service, store
    if service.search_service is not None:
        service.search_service.close()
    http.close()


@pytest.mark.parametrize(("split", "use_event_date"), list(MAX_SERIOUS_ERRORS))
def test_serious_errors_do_not_increase(offline_service, split, use_event_date):
    service, store = offline_service
    outcomes = evaluate(service, store, load_cases(split), use_event_date, quiet=True)

    # Busca sem gravação conta como fonte fora do ar e mudaria o resultado sem avisar
    unrecorded = [o.case.id if split == TRAIN_SPLIT else "?" for o in outcomes if o.report.sources_failed]
    assert not unrecorded, f"{len(unrecorded)} casos com busca sem gravação: rode {RECORD_COMMAND}"

    serious = [o for o in outcomes if o.serious_error]
    limit = MAX_SERIOUS_ERRORS[split, use_event_date]
    # Da validação só o total: listar os casos levaria a regras feitas sob medida para eles
    listed = [o.case.id for o in serious] if split == TRAIN_SPLIT else "não listados"
    assert len(serious) <= limit, f"{len(serious)} erros graves em {split} (teto {limit}): {listed}"
