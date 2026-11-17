---
name: backend
description: Padrões obrigatórios do back-end Python (FastAPI, Pydantic, SQLAlchemy, PostgreSQL, scikit-learn) do detector de fake news. Use em qualquer criação ou alteração de código do back-end.
---

# Padrões do back-end

Stack: Python 3.12, FastAPI, Pydantic v2, pydantic-settings, SQLAlchemy 2 (sessão **síncrona**), PostgreSQL, Alembic, scikit-learn, Pytest.

Antes de escrever código, leia os módulos vizinhos e siga o mesmo estilo. As regras abaixo valem para todo código novo ou alterado.

## 1. Arquitetura Model-Service-Controller (MSC)

Toda funcionalidade segue três camadas, sempre nesta direção: **Controller → Service → Model**. Uma camada só chama a camada imediatamente abaixo, nunca o contrário.

| Camada | Responsabilidade | Não pode |
|---|---|---|
| **Controller** | Expor o endpoint REST: receber a requisição, validar via schema, aplicar dependências (auth, sessão), chamar o service e devolver a resposta com o status HTTP correto. | Conter regra de negócio ou query. |
| **Service** | Regras de negócio (RNxx), orquestração, autorização por recurso, cache, chamadas a ML e a integrações externas. | Conhecer `Request`/`Response`/status HTTP ou montar query. |
| **Model** | Entidades ORM (`*_model.py`) e acesso a dados (`*_repository.py`). Único lugar que fala com o banco. | Conter regra de negócio ou fazer `commit`. |

```
app/
├── main.py                        # cria o app, lifespan, controllers, middlewares e handlers
├── core/
│   ├── config.py                  # Settings (pydantic-settings), única fonte de configuração
│   ├── security.py                # hash, JWT, dependências de autenticação
│   ├── cache.py                   # helpers de cache
│   └── exceptions.py              # exceções de domínio e handlers HTTP
├── controllers/
│   └── news_controller.py
├── services/
│   └── news_service.py
├── models/
│   ├── news_model.py              # entidade ORM
│   └── news_repository.py         # queries da entidade
├── schemas/
│   ├── page.py                    # Page[T] genérico
│   └── news/
│       ├── form.py                # campos e validações compartilhados
│       ├── create.py              # POST, importa o form
│       ├── update.py              # PUT e PATCH, importam o form
│       ├── filters.py             # só se o recurso tiver listagem
│       └── response.py            # o que sai para o cliente
├── integrations/                  # clientes de APIs/sites externos (chamados só pelos services)
│   └── wordpress_search_client.py
├── database/
│   └── session.py                 # engine e dependência get_db
├── data/                          # configurações de domínio em JSON (léxicos, fontes)
└── ml/
    ├── news_classifier.py         # carrega o artefato e faz a inferência
    └── artifacts/                 # modelos treinados versionados (.joblib + metadata.json)
training/                          # scripts de treino e avaliação, fora do app
migrations/                        # Alembic
tests/
├── controllers/
├── services/
├── models/
├── integrations/                  # com httpx.MockTransport, nunca rede real
└── ml/
```

- Um arquivo por recurso em cada camada, com sufixo da camada: `<resource>_controller.py`, `<resource>_service.py`, `<resource>_model.py`, `<resource>_repository.py`.
- Services lançam exceções de domínio (`NotFoundError`, `BusinessRuleError`, `ForbiddenError`, `ConflictError`); handlers em `core/exceptions.py` as convertem em status HTTP.
- Injeção de dependência com `Depends` para sessão, usuário atual e services.
- Integrações externas: timeout, limite de tamanho da resposta, validação do domínio final e dos links devolvidos. A falha de um site externo nunca derruba a requisição: o service registra no log e segue com resultado parcial.
- Funções pequenas e com um propósito; se passar de ~40 linhas, provavelmente dá para dividir.

```python
# controllers/news_controller.py
router = APIRouter(prefix="/news", tags=["news"])

@router.get("/{news_id}", response_model=NewsResponse)
def get_news(news_id: int, service: NewsService = Depends(get_news_service)):
    return service.get_by_id(news_id)

# services/news_service.py
class NewsService:
    def __init__(self, repository: NewsRepository):
        self.repository = repository

    def get_by_id(self, news_id: int) -> News:
        news = self.repository.find_by_id(news_id)
        if news is None:
            raise NotFoundError("Notícia não encontrada.")
        return news

# models/news_repository.py
class NewsRepository:
    def __init__(self, db: Session):
        self.db = db

    def find_by_id(self, news_id: int) -> News | None:
        return self.db.get(News, news_id)

    def add(self, news: News) -> News:
        self.db.add(news)
        self.db.flush()  # gera o id sem fechar a transação
        return news
```

### Transações

Uma requisição é uma transação. A dependência `get_db` faz `commit` no sucesso e `rollback` em qualquer exceção. Repositórios usam só `add`/`flush`; services e repositórios nunca chamam `commit`.

```python
# database/session.py
def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
```

## 2. API REST

- **Recursos são substantivos no plural**, em inglês e `kebab-case`: `/news`, `/news-analyses`, `/sources`. Nada de verbos na URL (`/news/analyze`, `/getNews` ❌).
- **O verbo HTTP define a ação:**

| Ação | Método | Rota | Sucesso |
|---|---|---|---|
| Listar (com filtros/paginação) | `GET` | `/news` | `200` |
| Detalhar | `GET` | `/news/{id}` | `200` |
| Criar | `POST` | `/news` | `201` + header `Location` |
| Substituir (todos os campos) | `PUT` | `/news/{id}` | `200` |
| Atualizar parcialmente | `PATCH` | `/news/{id}` | `200` |
| Remover | `DELETE` | `/news/{id}` | `204` |

- Uma ação que não é CRUD vira a criação de um recurso: "analisar notícia" é `POST /news-analyses`, não `POST /news/analyze`.
- Sub-recursos para relações: `GET /news/{id}/analyses`.
- **Versionamento** no prefixo: `/api/v1/...`.
- **Exceção permitida:** `GET /health` (processo vivo) e `GET /ready` (banco e modelo de ML carregados), fora do `/api/v1`.
- **Status corretos:** `400` requisição malformada, `401` sem autenticação, `403` sem permissão, `404` não encontrado, `409` conflito, `422` validação, `429` limite de requisições, `500` erro interno (sem detalhes).
- **Formato de erro único** em toda a API, inclusive no 422 de validação:

```json
{"detail": "Dados de entrada inválidos.", "errors": [{"field": "text", "message": "..."}]}
```

O 422 padrão do FastAPI tem outro formato, então sobrescreva o handler:

```python
# core/exceptions.py
@app.exception_handler(RequestValidationError)
def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [
        {"field": ".".join(str(part) for part in err["loc"][1:]), "message": err["msg"]}
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"detail": "Dados de entrada inválidos.", "errors": errors},
    )
```

- Filtros, ordenação e paginação via **query string**: `GET /news?classification=fake&order_by=-analyzed_at&page=2&page_size=20`.
- `GET`, `PUT` e `DELETE` são idempotentes; `GET` nunca altera estado.
- Stateless: nenhuma sessão em memória do servidor; o estado vem do token/cookie e do banco.
- JSON com campos em `snake_case` e em inglês; datas em ISO 8601 UTC.

## 3. Nomes em inglês, comentários em português

- **Arquivos e pastas**, funções, variáveis, classes, campos de schema, tabelas e colunas: **inglês**. Arquivos e pastas em `snake_case`, classes em `PascalCase`, tabelas no plural (`news`, `analyses`).
- Mensagens para o usuário final (erros, avisos): **português**.
- Comentários e docstrings: **português**, breves. Comente o *porquê*, não o óbvio. Sem comentários decorativos.

```python
# Ruim
# arquivo: services/analise.py
def analisar_noticia(texto): ...   # nome em português
x = 0.6                            # sem significado, hardcoded

# Bom
# arquivo: services/news_analysis_service.py
def analyze_news(news: News, settings: Settings) -> AnalysisResult:
    # Abaixo do limiar o resultado é inconclusivo (RN07)
    if confidence < settings.confidence_threshold:
        ...
```

## 4. Form compartilhado: create e update importam o mesmo form

Para todo recurso que recebe dados, crie os schemas em `schemas/<resource>/`. As restrições de cada campo ficam **uma vez**, como tipos `Annotated` no `form.py`, e são reaproveitadas pelo create, pelo PUT e pelo PATCH.

```python
# schemas/news/form.py
NewsText = Annotated[str, Field(min_length=1, max_length=100_000)]
NewsTitle = Annotated[str, Field(max_length=500)]

class NewsForm(BaseModel):
    # strip + min_length=1 já rejeita texto em branco
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    text: NewsText
    title: NewsTitle | None = None
    url: HttpUrl | None = None

# schemas/news/create.py
from app.schemas.news.form import NewsForm

class NewsCreate(NewsForm):
    pass

# schemas/news/update.py
from app.schemas.news.form import NewsForm, NewsText

class NewsUpdate(NewsForm):
    # PUT substitui o recurso inteiro: exige os mesmos campos do create
    pass

class NewsPatch(NewsForm):
    # PATCH: tudo opcional, mas com as mesmas restrições do form
    text: NewsText | None = None

    @model_validator(mode="after")
    def reject_null_required(self) -> Self:
        # Omitir é permitido; enviar null num campo obrigatório não
        if "text" in self.model_fields_set and self.text is None:
            raise ValueError("O texto da notícia não pode ser nulo.")
        return self
```

- No service, aplique o PATCH só com os campos enviados: `data.model_dump(exclude_unset=True)`.
- Sempre `extra="forbid"`, para rejeitar campos desconhecidos.
- Nunca escreva validador que assuma valor não nulo num campo que pode ser `None`: isso vira erro 500 em vez de 422.
- O schema de resposta (`response.py`) é separado, lista explicitamente os campos expostos e usa `ConfigDict(from_attributes=True)`.

## 5. Nunca confiar no cliente: regras e validação sempre no back-end

- Toda entrada é validada no servidor (schema no controller + regras no service), mesmo que o front já valide.
- O cliente nunca define `id`, `owner_id`, `role`, `created_at`, status, resultado de análise, confiança ou versão do modelo. Esses valores são calculados no servidor.
- Verifique autorização no service: o usuário só acessa os próprios recursos (evite IDOR).
- Regras de negócio (RNxx) ficam nos services, nunca no front e nunca só no banco.
- Limite tamanhos (strings, listas, uploads) e valide tipos, formatos e intervalos.
- Use o ORM ou parâmetros vinculados; nunca monte SQL com f-string.

## 6. Nada hardcoded

- URLs, credenciais, limites, limiares, TTLs de cache, origens de CORS, caminhos de arquivo e do modelo de ML: tudo em `Settings` (`core/config.py`), lido de variáveis de ambiente/`.env`.
- Mantenha `.env.example` atualizado (sem valores reais) e `.env` no `.gitignore`.
- Listas de domínio (léxicos, fontes) ficam em `app/data/` ou no banco, não espalhadas no código.
- Constantes nomeadas para números mágicos que não são configuração.

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: Literal["development", "test", "production"] = "development"
    database_url: PostgresDsn
    secret_key: SecretStr           # SecretStr evita vazar em logs/repr
    cors_origins: list[str] = []
    docs_enabled: bool = False
    default_page_size: int = 20
    max_page_size: int = 100
    cache_ttl_seconds: int = 300
    cache_max_items: int = 1_000
    model_path: Path
    confidence_threshold: float = Field(0.6, ge=0, le=1)
```

## 7. Filtros e paginação sempre no back-end

- Toda listagem é paginada no banco (`LIMIT/OFFSET` ou cursor). Nunca devolva a tabela inteira para o cliente filtrar.
- Valide os parâmetros em `schemas/<resource>/filters.py`: `page >= 1`, `page_size` entre 1 e `settings.max_page_size`.
- Ordenação e filtros usam uma **lista branca** de campos permitidos (`Literal` ou `Enum`); nunca passe nome de coluna vindo do cliente direto para a query.
- Resposta padrão de listagem:

```python
# schemas/page.py
class Page[T](BaseModel):
    items: list[T]
    total: int
    page: int
    page_size: int
    pages: int
```

## 8. Segurança: nada exposto, nada sensível no cliente

- **Tokens:** nunca devolva segredos (API keys, chaves de serviços, refresh tokens) no corpo da resposta. Sessão/refresh em cookie `HttpOnly`, `Secure`, `SameSite`. Access token de curta duração.
- **Senhas:** hash com `argon2`/`bcrypt`; nunca logar nem retornar.
- **Respostas:** `response_model` sempre, para não vazar colunas internas. Nunca retorne o model ORM sem um schema.
- **Erros:** mensagens genéricas para o cliente; detalhes (stack trace, SQL) só no log. `debug=False` em produção.
- **Docs:** `/docs` e `/openapi.json` desativados em produção via `settings.docs_enabled`.
- **CORS:** origens explícitas vindas de Settings; nunca `"*"` com credenciais.
- **Rate limiting** (ex.: `slowapi`) nos endpoints públicos e nos custosos, como `POST /news-analyses`. Limites em Settings.
- **Logs:** estruturados (JSON), com um `request_id` por requisição; nunca registrar tokens, senhas, dados pessoais ou o `.env`.
- **Dependências:** versões fixadas em lockfile; nada de `eval`/`pickle`/`joblib.load` de conteúdo vindo do usuário (joblib só de artefatos próprios).
- **Cabeçalhos de segurança:** `X-Content-Type-Options`, `X-Frame-Options`, `Strict-Transport-Security` via middleware.

## 9. Machine learning

- **Treino fora do app:** scripts em `training/`, com seed fixa e dados versionados ou referenciados. O app só faz inferência.
- **Pipeline único:** pré-processamento (limpeza, vetorização) e classificador dentro do mesmo `sklearn.pipeline.Pipeline`, salvo num único artefato. Assim treino e inferência usam exatamente a mesma transformação e não há *data leakage*.
- **Divisão dos dados:** split de treino/teste feito antes de qualquer `fit`; validação cruzada estratificada; evitar a mesma notícia (ou quase duplicata) em treino e teste.
- **Versionamento:** cada artefato vem com um `metadata.json` (versão, data, dataset, métricas). Toda análise salva `model_version` no banco.
- **Métricas:** registre precision, recall e F1 por classe e a matriz de confusão, não só a acurácia (as classes costumam ser desbalanceadas).
- **Resultado probabilístico:** a API devolve classificação, confiança e `model_version`. Abaixo de `settings.confidence_threshold` o resultado é "inconclusivo" (RN07). As mensagens nunca afirmam que algo "é falso" de forma absoluta.
- **Carregamento:** o modelo é carregado **uma vez** no `lifespan` e injetado via dependência; se o artefato falhar ao carregar, `/ready` responde 503.

```python
# main.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.classifier = NewsClassifier.load(settings.model_path)
    yield
```

## 10. Desempenho e cache

- Evite N+1: use `selectinload`/`joinedload` quando precisar de relacionamentos.
- Crie índices (via migração) para colunas usadas em filtros e ordenação.
- Inferência é CPU-bound: use rota `def` (threadpool) ou worker, nunca `async def` bloqueando o event loop.
- Cache quando for pertinente, aplicado na camada **service**: leituras frequentes e de baixo custo de invalidação (análise de texto idêntico, listas de referência, configurações).
  - Cache local com TTL: `cachetools.TTLCache(maxsize=settings.cache_max_items, ttl=settings.cache_ttl_seconds)`. Não use `functools.lru_cache` para dados que expiram, porque ele não tem TTL.
  - Cache compartilhado entre instâncias: Redis, com o mesmo TTL de Settings.
  - A chave de cache da análise inclui o hash do texto normalizado **e** o `model_version`, para não servir resultado de um modelo antigo depois de retreinar.
  - Invalide na escrita.
- Nunca coloque em cache dados de um usuário sob uma chave que outro usuário possa acessar.

## 11. Banco de dados e migrações

- Toda mudança de schema vira uma migração Alembic. Revise sempre o arquivo gerado por `--autogenerate` antes de aplicar (ele não detecta tudo, como renomeações).
- Datas em `DateTime(timezone=True)`, geradas com `datetime.now(UTC)`. Nunca use `datetime.utcnow()` (depreciado e sem timezone).
- Restrições importantes também no banco (`NOT NULL`, `UNIQUE`, `FOREIGN KEY`), como segunda linha de defesa; a regra de negócio continua no service.

## 12. Testes

- Todo endpoint ou regra nova vem com teste em `tests/`, espelhando as camadas: `controllers/`, `services/`, `models/`, `integrations/`, `ml/`.
- Services testados isoladamente (repositório falso); controllers com `TestClient` e `app.dependency_overrides`.
- Repositórios e controllers rodam contra **PostgreSQL real** (Docker ou testcontainers), com rollback por teste. Não use SQLite: tipos, `ILIKE`, JSONB e timezone se comportam diferente.
- Integrações externas com `httpx.MockTransport`; nunca rede real.
- ML: teste de fumaça que carrega o artefato e classifica um texto conhecido; teste de que o pipeline aceita texto vazio/longo sem quebrar.
- Cubra: caminho feliz, entrada inválida (422, no formato único), `null` em campo obrigatório no PATCH, acesso não autorizado (401/403), recurso inexistente (404), limites de paginação, ordenação fora da lista branca.

## 13. Qualidade de código

- `ruff` (lint e format) e `mypy` sem erros; ambos no pre-commit e no CI.
- Type hints em todas as funções públicas.

## Checklist antes de concluir

- [ ] Camadas MSC respeitadas (controller → service → model), sem atalhos
- [ ] Commit/rollback só no `get_db`; repositórios usam `flush`
- [ ] Rotas REST: substantivos no plural, verbo HTTP correto, status correto, `/api/v1`
- [ ] Erros (inclusive 422) no formato único
- [ ] Arquivos, funções e variáveis em inglês; comentários em português, curtos
- [ ] `form.py` + `create.py` + `update.py` + `response.py` no recurso; `filters.py` se houver listagem
- [ ] PUT exige todos os campos; PATCH usa `exclude_unset=True` e rejeita `null` em campo obrigatório
- [ ] Validação e regra de negócio no servidor; `extra="forbid"`
- [ ] Nenhum valor hardcoded; `.env.example` atualizado
- [ ] Listagens paginadas e filtradas no back-end, com lista branca
- [ ] Nenhum segredo/campo interno exposto; `response_model` em todas as rotas
- [ ] Análises salvam `model_version`; cache da análise inclui `model_version` na chave
- [ ] Cache com TTL configurável, aplicado onde faz sentido
- [ ] Migração Alembic criada e revisada, se o schema mudou
- [ ] `ruff`, `mypy` e `pytest` passando