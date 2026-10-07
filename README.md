# Fake News Detection API — Back-end

API em FastAPI que recebe uma notícia (texto, título e/ou URL), identifica a afirmação principal, busca evidências em agências de checagem e veículos jornalísticos e devolve:
- a classificação (`likely_true`, `likely_false` ou `inconclusive`);
- a justificativa, as evidências com links e as limitações.

> Sem evidências suficientes, o resultado é **inconclusivo**, nunca "falso".

Padrões de código: [`.claude/skills/backend/SKILL.md`](.claude/skills/backend/SKILL.md).

## Como executar

Funciona em **Linux, macOS e Windows**. Quando o comando muda entre os sistemas, há uma versão para cada um. No Windows, use o **PowerShell**.

**Pré-requisitos:** Python 3.12 e Docker (no Windows, [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/)).

### 1. Ambiente Python

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- Ative o venv em **todo terminal novo**: o prompt passa a mostrar `(.venv)`.
- No Windows, se o PowerShell bloquear o `Activate.ps1`, rode uma vez: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- Se a pasta do projeto for renomeada ou movida, o `.venv` quebra. Apague a pasta `.venv` e repita os comandos.

### 2. Configuração

Linux/macOS: `cp .env.example .env` · Windows: `copy .env.example .env`

No `.env`, defina:
- `POSTGRES_PASSWORD`, usando a mesma senha dentro de `DATABASE_URL`;
- `ADMIN_API_KEY`, gerada com `python -c "import secrets; print(secrets.token_urlsafe(32))"`.

### 3. Banco de dados

Os comandos abaixo são iguais nos dois sistemas:

```bash
docker compose up -d db            # PostgreSQL em localhost:5432
alembic upgrade head               # cria/atualiza as tabelas
```

Para acessar o banco:

```bash
docker compose exec db psql -U fakenews -d fakenews
```

Em um cliente gráfico (DBeaver, pgAdmin), use `localhost:5432` com o usuário, a senha e o banco do `.env`.

Tabelas: `news` (notícias), `news_analyses` (resultados) e `evidences` (fontes consultadas).

### 4. Back-end

```bash
python -m app
```

- API: http://localhost:8000/api/v1
- Documentação: http://localhost:8000/docs (com `DOCS_ENABLED=true`)
- Status: http://localhost:8000/health

### 5. Front-end

**Página de teste:** com a API rodando, abra outro terminal (com o venv ativo) e execute:

```bash
python scripts/playground.py       # http://localhost:5173
```

**Projeto React:** use a URL base `http://localhost:8000/api/v1`. A origem do front precisa estar em `CORS_ORIGINS` no `.env`.

### 6. Testes

```bash
pytest                             # automatizados (sem banco e sem internet)
python scripts/smoke_test.py       # ponta a ponta, com a API rodando
```

Para testar requisição por requisição, use [`docs/api-tests.http`](docs/api-tests.http).

### Opcional: tudo em Docker

```bash
docker compose --profile api up -d --build
```

## Endpoints (`/api/v1`)

| Método | Rota | Descrição |
|---|---|---|
| POST | `/news-analyses` | Analisa uma notícia: `text` e/ou `url` (obrigatório um dos dois), `title`, `source`, `published_at` |
| GET | `/news-analyses` | Lista análises (`page`, `page_size`, `classification`, `verification_status`, `order_by`) |
| GET | `/news-analyses/{id}` | Relatório completo |
| GET | `/news` · `/news/{id}` | Notícias |
| GET · POST | `/news/{id}/analyses` | Histórico da notícia · reanalisar |
| PATCH · DELETE | `/news/{id}` | Editar · remover (exige o cabeçalho `X-API-Key`) |

Os erros seguem o formato `{"detail": "...", "errors": [{"field": "...", "message": "..."}]}`.

## Como a classificação é decidida

| Situação | Resultado |
|---|---|
| Fontes confiáveis contradizem a afirmação | `likely_false` |
| Fontes confiáveis sustentam a afirmação | `likely_true` |
| Fontes divergem, nenhuma evidência, fontes fora do ar ou texto curto | `inconclusive` |

- **Fontes consultadas:** Agência Lupa, Boatos.org e Folha de S.Paulo. A Google Fact Check Tools também entra se `GOOGLE_FACT_CHECK_API_KEY` estiver definida.
- **O que conta:** só evidências relevantes, de fontes cadastradas em `app/data/source_registry.json` e independentes entre si.
- **O que não decide:** o estilo do texto só gera alertas. Estar em um site conhecido não torna a notícia verdadeira.

## Problemas comuns

| Problema | Solução |
|---|---|
| `No module named 'pydantic'` ao rodar `alembic` | Ative o venv (passo 1). Se não resolver, recrie o `.venv`. |
| Windows: `python` abre a Microsoft Store ou não é encontrado | Instale o Python 3.12 de python.org marcando "Add to PATH", ou use `py -3.12` |
| Windows: `docker` não é reconhecido ou não conecta | Abra o Docker Desktop e espere ele iniciar |
| `/health` retorna 503 ou `connection refused` | Suba o banco: `docker compose up -d db` |
| `relation ... does not exist` | `alembic upgrade head` |
| Erro de CORS no front | Adicione a origem em `CORS_ORIGINS` e reinicie a API |
| 429 | Limite de análises por minuto (`RATE_LIMIT_CREATE`) |
