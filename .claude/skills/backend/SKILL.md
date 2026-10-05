---
name: backend
description: Padrões obrigatórios para criar ou alterar código do back-end Python (FastAPI, Pydantic, SQLAlchemy, PostgreSQL, scikit-learn). Use sempre que for criar endpoints, controllers, services, models, schemas, módulos de ML, testes ou configuração neste projeto — mesmo em mudanças pequenas — para garantir arquitetura Model-Service-Controller, API REST, nomes (inclusive de arquivos) em inglês, segurança, validação no servidor, paginação/filtros no back-end, cache e nada hardcoded.
---

# Padrões do back-end

Stack: Python 3.12, FastAPI, Pydantic v2, pydantic-settings, SQLAlchemy 2, PostgreSQL, scikit-learn, Pytest.

Antes de escrever código, leia os módulos vizinhos e siga o mesmo estilo. As regras abaixo valem para todo código novo ou alterado.

## 1. Arquitetura Model-Service-Controller (MSC)

Toda funcionalidade segue três camadas, sempre nesta direção: **Controller → Service → Model**. Uma camada só chama a camada imediatamente abaixo; nunca o contrário.

| Camada | Responsabilidade | Não pode |
|---|---|---|
| **Controller** | Expor o endpoint REST: receber a requisição, validar via schema, aplicar dependências (auth, sessão), chamar o service e devolver a resposta com o status HTTP correto. | Conter regra de negócio ou query. |
| **Service** | Regras de negócio (RNxx), orquestração, autorização por recurso, cache, chamadas a ML e serviços externos. | Conhecer `Request`/`Response`/status HTTP ou montar query. |
| **Model** | Entidades ORM e acesso a dados (queries). Único lugar que fala com o banco. | Conter regra de negócio. |

```
app/
├── main.py                        # cria o app, registra controllers, middlewares e handlers
├── core/
│   ├── config.py                  # Settings (pydantic-settings), única fonte de configuração
│   ├── security.py                # hash, JWT, dependências de autenticação
│   ├── cache.py                   # helpers de cache
│   └── exceptions.py              # exceções de domínio e handlers HTTP
├── controllers/
│   └── news_controller.py         # APIRouter do recurso
├── services/
│   └── news_service.py
├── models/
│   ├── news_model.py              # entidade ORM
│   └── news_repository.py         # queries da entidade
├── schemas/
│   └── news/
│       ├── form.py                # campos e validações compartilhados
│       ├── create.py              # importa o form
│       ├── update.py              # importa o form
│       ├── filters.py             # parâmetros de filtro/ordenação permitidos
│       └── response.py            # o que sai para o cliente
├── integrations/                  # clientes de APIs/sites externos (chamados só pelos services)
│   └── wordpress_search_client.py
├── database/
│   └── session.py                 # engine e sessão
├── data/                          # configurações de domínio em JSON (léxicos, fontes)
└── ml/
    └── news_classifier.py
migrations/                        # Alembic: toda mudança de schema vira uma migração
tests/
├── controllers/test_news_controller.py
├── services/test_news_service.py
└── integrations/                  # com httpx.MockTransport, nunca rede real
```

- Um arquivo por recurso em cada camada, com sufixo da camada: `<resource>_controller.py`, `<resource>_service.py`, `<resource>_model.py`, `<resource>_repository.py`.
- Services lançam exceções de domínio (`NotFoundError`, `BusinessRuleError`, `ForbiddenError`); handlers em `core/exceptions.py` as convertem em status HTTP. Assim o service não depende de HTTP.
- Injeção de dependência com `Depends` para sessão, usuário atual e services.
- Integrações externas (`integrations/`): timeout, limite de tamanho da resposta, validação do domínio final e dos links devolvidos; falha de um site externo nunca derruba a requisição.
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
```

## 2. API REST

- **Recursos são substantivos no plural**, em inglês e `kebab-case`: `/news`, `/news-analyses`, `/sources`. Nada de verbos na URL (`/news/analyze`, `/getNews` ❌).
- **O verbo HTTP define a ação:**

| Ação | Método | Rota | Sucesso |
|---|---|---|---|
| Listar (com filtros/paginação) | `GET` | `/news` | `200` |
| Detalhar | `GET` | `/news/{id}` | `200` |
| Criar | `POST` | `/news` | `201` + header `Location` |
| Substituir | `PUT` | `/news/{id}` | `200` |
| Atualizar parcialmente | `PATCH` | `/news/{id}` | `200` |
| Remover | `DELETE` | `/news/{id}` | `204` |

- Uma ação que não é CRUD vira a criação de um recurso: "analisar notícia" é `POST /news-analyses` (cria uma análise), não `POST /news/analyze`.
- Sub-recursos para relações: `GET /news/{id}/analyses`.
- **Versionamento** no prefixo: `/api/v1/...`.
- **Status corretos:** `400` requisição malformada, `401` sem autenticação, `403` sem permissão, `404` não encontrado, `409` conflito, `422` validação, `429` limite de requisições, `500` erro interno (sem detalhes).
- **Formato de erro único** em toda a API:

```json
{"detail": "Dados de entrada inválidos.", "errors": [{"field": "text", "message": "..."}]}
```

- Filtros, ordenação e paginação via **query string**: `GET /news?classification=fake&order_by=-analyzed_at&page=2&page_size=20`.
- `GET`, `PUT` e `DELETE` são idempotentes; `GET` nunca altera estado.
- Stateless: nenhuma sessão em memória do servidor; o estado vem do token/cookie e do banco.
- JSON com campos em `snake_case` e em inglês; datas em ISO 8601 UTC.

## 3. Nomes em inglês, comentários em português

- **Arquivos e pastas**, funções, variáveis, classes, campos de schema, tabelas e colunas: **inglês**. Arquivos e pastas em `snake_case` (`news_service.py`), classes em `PascalCase`, tabelas no plural (`news`, `analyses`).
- Mensagens para o usuário final (erros, avisos): **português**.
- Comentários e docstrings: **português**, breves e objetivos. Comente o *porquê*, não o óbvio. Sem comentários decorativos.

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

Para todo recurso que recebe dados, crie os arquivos em `schemas/<resource>/`. As validações ficam **uma vez** no form.

```python
# schemas/news/form.py
class NewsForm(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    text: str = Field(..., min_length=1, max_length=100_000)
    title: str | None = Field(None, max_length=500)
    url: HttpUrl | None = None

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("O texto da notícia não pode estar vazio.")
        return value

# schemas/news/create.py
from app.schemas.news.form import NewsForm

class NewsCreate(NewsForm):
    pass

# schemas/news/update.py
from app.schemas.news.form import NewsForm

class NewsUpdate(NewsForm):
    # No update todos os campos são opcionais, mas as validações do form continuam valendo
    text: str | None = None
```

- Sempre `extra="forbid"`, para rejeitar campos desconhecidos.
- O schema de resposta (`response.py`) é separado e lista explicitamente os campos expostos.

## 5. Nunca confiar no cliente: regras e validação sempre no back-end

- Toda entrada é validada no servidor (schema no controller + regras no service), mesmo que o front já valide.
- O cliente nunca define `id`, `owner_id`, `role`, `created_at`, preço, status, resultado de análise etc. Esses valores são calculados no servidor.
- Verifique autorização no service: o usuário só acessa os próprios recursos (evite IDOR).
- Regras de negócio (RNxx) ficam nos services, nunca no front e nunca só no banco.
- Limite tamanhos (strings, listas, uploads) e valide tipos, formatos e intervalos.
- Use o ORM ou parâmetros vinculados; nunca monte SQL com f-string.

## 6. Nada hardcoded

- URLs, credenciais, limites, limiares, TTLs de cache, origens de CORS, caminhos de arquivo: tudo em `Settings` (`core/config.py`), lido de variáveis de ambiente/`.env`.
- Mantenha `.env.example` atualizado (sem valores reais) e `.env` no `.gitignore`.
- Listas de domínio (léxicos, fontes) ficam em arquivos de dados ou no banco, não espalhadas no código.
- Constantes nomeadas para números mágicos que não são configuração.

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: PostgresDsn
    secret_key: SecretStr           # SecretStr evita vazar em logs/repr
    cors_origins: list[str] = []
    default_page_size: int = 20
    max_page_size: int = 100
    cache_ttl_seconds: int = 300
```

## 7. Filtros e paginação sempre no back-end

- Toda listagem é paginada no banco (`LIMIT/OFFSET` ou cursor). Nunca devolva a tabela inteira para o cliente filtrar.
- Valide os parâmetros em `schemas/<resource>/filters.py`: `page >= 1`, `page_size` com teto em `settings.max_page_size`.
- Ordenação e filtros usam uma **lista branca** de campos permitidos; nunca passe nome de coluna vindo do cliente direto para a query.
- Resposta padrão de listagem:

```python
class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    page_size: int
    pages: int
```

## 8. Segurança: nada exposto, nada sensível no cliente

- **Tokens:** nunca devolva segredos (API keys, chaves de serviços, refresh tokens) no corpo da resposta para guardar no cliente. Sessão/refresh em cookie `HttpOnly`, `Secure`, `SameSite`. Access token de curta duração.
- **Senhas:** hash com `argon2`/`bcrypt`; nunca logar nem retornar.
- **Respostas:** use `response_model` sempre, para não vazar colunas internas. Nunca retorne o model ORM sem um schema.
- **Erros:** mensagens genéricas para o cliente; detalhes (stack trace, SQL) só no log. `debug=False` em produção.
- **Docs:** `/docs` e `/openapi.json` desativados ou protegidos em produção (via Settings).
- **CORS:** origens explícitas vindas de Settings; nunca `"*"` com credenciais.
- **Rate limiting** nos endpoints públicos e nos custosos (ex.: análise de ML).
- **Logs:** nunca registrar tokens, senhas, dados pessoais ou o `.env`.
- **Dependências:** versões fixadas; nada de `eval`/`pickle` de entrada do usuário (joblib só de arquivos próprios).
- **Cabeçalhos de segurança:** `X-Content-Type-Options`, `X-Frame-Options`, `Strict-Transport-Security` via middleware.

## 9. Desempenho e cache

- Carregue modelos de ML e recursos pesados **uma vez** na inicialização (`lifespan`) e reutilize.
- Evite N+1: use `selectinload`/`joinedload` quando precisar de relacionamentos.
- Crie índices para colunas usadas em filtros e ordenação.
- Operações CPU-bound (inferência) em rota `def` (threadpool) ou worker, nunca bloqueando o event loop em `async def`.
- Cache quando for pertinente, aplicado na camada **service**: leituras frequentes e de baixo custo de invalidação (ex.: resultado de análise de um texto idêntico por hash, listas de referência, configurações). TTL vindo de Settings; invalide na escrita. Use `functools.lru_cache` para dados do processo e Redis para cache compartilhado entre instâncias.
- Nunca coloque em cache dados de um usuário sob uma chave que outro usuário possa acessar.

## 10. Testes

- Todo endpoint ou regra nova vem com teste em `tests/`, espelhando as camadas: `tests/controllers/`, `tests/services/`, `tests/models/`.
- Services testados isoladamente (repositório falso/mock); controllers com `TestClient`.
- Cubra: caminho feliz, entrada inválida (422), acesso não autorizado (401/403), recurso inexistente (404), limites de paginação.
- Testes não dependem de serviços externos: SQLite em memória ou fixtures, e mocks para APIs externas.

## Checklist antes de concluir

- [ ] Camadas MSC respeitadas (controller → service → model), sem atalhos
- [ ] Rotas REST: substantivos no plural, verbo HTTP correto, status correto, `/api/v1`
- [ ] Arquivos, funções e variáveis em inglês; comentários em português, curtos
- [ ] `form.py` + `create.py` + `update.py` + `filters.py` + `response.py` no recurso
- [ ] Validação e regra de negócio no servidor; `extra="forbid"`
- [ ] Nenhum valor hardcoded; `.env.example` atualizado
- [ ] Listagens paginadas e filtradas no back-end, com lista branca
- [ ] Nenhum segredo/campo interno exposto; `response_model` em todas as rotas
- [ ] Cache aplicado onde faz sentido, com TTL configurável
- [ ] Testes passando (`pytest`)
