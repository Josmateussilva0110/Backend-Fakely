# Fake News Detection API — Back-end

API REST em Python (FastAPI) que recebe uma notícia, valida, pré-processa o texto, extrai características e retorna uma classificação (`likely_true`, `likely_false` ou `inconclusive`) com nível de confiança.

> O resultado é uma estimativa do sistema, não uma confirmação absoluta da veracidade da informação.

Os padrões de código do projeto estão em [`.claude/skills/backend/SKILL.md`](.claude/skills/backend/SKILL.md).

## Arquitetura (Model-Service-Controller)

```
app/
├── main.py                          # create_app(): middlewares, handlers, controllers
├── core/                            # config, exceptions, security, cache, dependencies
├── controllers/news_controller.py   # endpoints REST
├── integrations/                    # clientes de busca (WordPress e Folha)
├── services/
│   ├── news_service.py              # regras de negócio do recurso news
│   ├── news_analysis_service.py     # pipeline de análise + cache
│   ├── source_verification_service.py  # busca nos sites confiáveis
│   ├── text_processing_service.py   # pré-processamento e extração de características
│   ├── analysis_rules_service.py    # carrega app/data/analysis_rules.json
│   └── verification_config_service.py  # carrega app/data/verification_sources.json
├── models/
│   ├── news_model.py                # entidade ORM (tabela news)
│   └── news_repository.py           # queries
├── schemas/news/                    # form, create, update, filters, response
├── database/session.py
├── ml/news_classifier.py            # TF-IDF + Regressão Logística
└── data/
    ├── analysis_rules.json          # léxicos, fontes reconhecidas, pesos da heurística
    └── verification_sources.json    # sites consultados e padrões de veredito
migrations/                          # Alembic
```

## Como rodar

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # ajuste senhas e gere ADMIN_API_KEY
docker compose up -d          # PostgreSQL em 127.0.0.1:5432
alembic upgrade head          # cria/atualiza as tabelas
python -m app                 # usa HOST, PORT e RELOAD do .env
```

Com `DOCS_ENABLED=true`, a documentação fica em http://localhost:8000/docs.

## Endpoints (`/api/v1`)

| Método | Rota | Descrição | Acesso |
|---|---|---|---|
| POST | `/news` | Cria a notícia e executa a análise | público, com limite de requisições |
| GET | `/news` | Lista com filtros e paginação | público |
| GET | `/news/{id}` | Detalha notícia e análise | público |
| PATCH | `/news/{id}` | Altera campos e reanalisa | `X-API-Key` |
| DELETE | `/news/{id}` | Remove | `X-API-Key` |
| GET | `/health` | Status | público |

Filtros de `GET /news`: `page`, `page_size` (máx. `MAX_PAGE_SIZE`), `classification`, `source`, `analyzed_from`, `analyzed_to`, `order_by` (`analyzed_at`, `published_at` ou `confidence`; prefixo `-` para decrescente).

Exemplo:

```bash
curl -X POST localhost:8000/api/v1/news -H 'content-type: application/json' \
  -d '{"text": "URGENTE!!! Compartilhe antes que apaguem...", "source": "exemplo.net"}'
```

Corpo: `text` (obrigatório), `title`, `url`, `source`, `published_at` (opcionais). Campos desconhecidos, ou que só o servidor pode definir (como `classification`), são rejeitados com 422.

Erros seguem sempre o formato `{"detail": "...", "errors": [{"field": "...", "message": "..."}]}`.

## Critérios de análise

- **Características textuais:** termos sensacionalistas e alarmistas, pontuação repetida, excesso de exclamações, proporção de maiúsculas.
- **Fonte:** domínio extraído da URL ou da fonte e comparado à lista de fontes reconhecidas.
- **Modelo de ML:** quando há modelo treinado, `P(falsa) = MODEL_WEIGHT × modelo + (1 − MODEL_WEIGHT) × heurística`.
- **Inconclusiva (RN07):** texto com menos de `MIN_WORDS` palavras, ou confiança abaixo de `CONFIDENCE_THRESHOLD`.
- **Configuração:** léxicos, fontes e pesos ficam em `app/data/analysis_rules.json`. Os limites ficam no `.env`.
- **Cache:** análises de conteúdo idêntico ficam em cache com tempo de expiração (`ANALYSIS_CACHE_TTL_SECONDS`).

## Verificação nos sites confiáveis

A cada análise, o back-end busca a notícia em três sites brasileiros, em paralelo e sem custo:

| Site | Tipo | Método |
|---|---|---|
| Agência Lupa | agência de checagem | API pública do WordPress |
| Boatos.org | agência de checagem | API pública do WordPress |
| Folha de S.Paulo | veículo jornalístico | scraping da página de busca |

1. Extrai até 4 palavras-chave do título (ou do texto), priorizando nomes próprios. Se não houver resultado, tenta de novo com as 2 mais específicas.
2. Compara o título de cada resultado com o texto do usuário (similaridade por palavras em comum) e detecta o veredito das checagens ("É falso que…", "#boato" etc.).
3. Ajusta a classificação:
   - uma checagem com similaridade ≥ `VERIFICATION_FACT_CHECK_MIN_SIMILARITY` decide como falsa ou verdadeira, desde que não haja matéria mais parecida no veículo jornalístico;
   - uma matéria correspondente na Folha reforça "possivelmente verdadeira";
   - nenhum resultado não altera nada.
4. A resposta traz os links encontrados em `verification.matches`, para o front-end exibir.

Se um site falhar ou demorar, a análise continua e informa isso em `sources_failed`. Resultados incompletos não entram no cache.

**Limitações do MVP:**
- A comparação é por palavras, não por significado. Checagens escritas com outras palavras podem não ser encontradas, e textos com palavras parecidas mas sentido oposto podem ser confundidos. Por isso a resposta mostra os links, para o usuário conferir.
- O scraping da Folha quebra se o site mudar o layout. O teste `tests/integrations/test_folha_search_client.py` documenta o formato esperado.

## Treinando o modelo

```bash
python scripts/train_model.py data.csv --text-column text --label-column label
```

Rótulos aceitos: `1`/`0`, `fake`/`true`, `falsa`/`verdadeira`. O modelo é salvo em `app/ml/news_classifier.joblib` e carregado ao iniciar a API. Dataset sugerido: [Fake.br Corpus](https://github.com/roneysco/Fake.br-Corpus).

## Testando como se fosse o front-end

Com a API rodando, em outro terminal:

```bash
./scripts/playground.sh       # abre em http://localhost:5173
```

É uma página HTML simples (`tools/playground/index.html`) com o formulário (texto, título, URL e fonte), a tela de resultado e o histórico. Ela chama a API pela mesma origem que o React vai usar, então também testa o CORS. Os exemplos prontos preenchem o formulário com um clique.

Para rodar todos os cenários de uma vez: `./scripts/smoke_test.sh`. Para testar requisição por requisição: `docs/api-tests.http`.

## Testes

```bash
pytest
```

Os testes ficam em `tests/controllers`, `tests/services` e `tests/models`. Eles usam SQLite em memória e não precisam do PostgreSQL.
