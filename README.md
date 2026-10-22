# Fake News Detection API — Back-end

API em FastAPI que recebe uma notícia (texto, título e/ou URL), identifica a afirmação principal, busca evidências em agências de checagem e veículos jornalísticos e devolve:
- a classificação (`likely_true`, `likely_false` ou `inconclusive`);
- a justificativa, as evidências com links e as limitações.

> Sem evidências suficientes, o resultado é **inconclusivo**, nunca "falso".

Padrões de código: [`.claude/skills/backend/SKILL.md`](.claude/skills/backend/SKILL.md).

## Como executar

Funciona em **Linux, macOS e Windows**. Quando o comando muda entre os sistemas, há uma versão para cada um. No Windows, use o **PowerShell**.

**Pré-requisitos:** Python 3.12 e Docker (no Windows, [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/)).

As dependências ficam em dois arquivos:

| Arquivo | Conteúdo | Quando usar |
|---|---|---|
| `requirements.txt` | Só o necessário para a API rodar | Produção (é o que a imagem Docker instala) |
| `requirements-dev.txt` | Tudo do `requirements.txt` + `pytest` e `ruff` | Desenvolvimento |

## Desenvolvimento

### 1. Ambiente Python

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

- Ative o venv em **todo terminal novo**: o prompt passa a mostrar `(.venv)`.
- No Windows, se o PowerShell bloquear o `Activate.ps1`, rode uma vez: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- Se a pasta do projeto for renomeada ou movida, o `.venv` quebra. Apague a pasta `.venv` e repita os comandos.

### 2. Configuração

Linux/macOS: `cp .env.example .env` · Windows: `copy .env.example .env`

No `.env`, defina:
- `POSTGRES_PASSWORD`, usando a mesma senha dentro de `DATABASE_URL`;
- `ADMIN_API_KEY`, gerada com `python -c "import secrets; print(secrets.token_urlsafe(32))"`.

O `.env.example` já vem pronto para desenvolvimento: `RELOAD=true` (reinicia a API ao salvar um arquivo) e `DOCS_ENABLED=true` (libera o `/docs`).

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

Na **primeira execução**, a API baixa o modelo de inferência textual (~1,1 GB) para a pasta `.models/`. Depois disso, ele só é carregado do disco (alguns segundos). Para rodar sem o modelo, defina `NLI_ENABLED=false` no `.env`.

- API: http://localhost:8000/api/v1
- Documentação: http://localhost:8000/docs
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
ruff check .                       # lint (imports sem uso, ordem, estilo)
```

Para testar requisição por requisição, use a documentação interativa em http://localhost:8000/docs.

## Produção

### 1. Configuração

Crie o `.env` a partir do `.env.example` e ajuste:

| Variável | Valor em produção |
|---|---|
| `ENVIRONMENT` | `production` |
| `RELOAD` | `false` |
| `DOCS_ENABLED` | `false` (não expõe `/docs` nem `/openapi.json`) |
| `CORS_ORIGINS` | Só o domínio real do front, ex.: `["https://meusite.com.br"]` |
| `POSTGRES_PASSWORD` | Senha forte, diferente da de desenvolvimento (e igual à de `DATABASE_URL`) |
| `ADMIN_API_KEY` | Chave nova, gerada com o comando do passo 2 de desenvolvimento |
| `NLI_ENABLED` | `true` (exige ~1,5 GB de RAM livre para o modelo) |
| `TRUSTED_PROXIES` | IP(s) do proxy reverso, ex.: `["127.0.0.1"]`. Com Docker, o gateway da rede (veja abaixo) |

Nunca versione o `.env` (ele já está no `.gitignore`).

### 2. Execução com Docker (recomendado)

```bash
docker compose --profile api up -d --build
```

- A imagem instala só o `requirements.txt` (sem `pytest` e `ruff`) e roda sem usuário root.
- Ao iniciar, o container aplica as migrações (`alembic upgrade head`) e depois sobe a API.
- O modelo NLI é baixado na primeira inicialização e fica no volume `models`; os próximos builds não baixam de novo.
- A API fica em `127.0.0.1:8000`, acessível só pela própria máquina. Para publicar, coloque um proxy reverso com HTTPS (Nginx, Caddy) na frente.

Para atualizar depois de um `git pull`, rode o mesmo comando: ele reconstrói a imagem e aplica as migrações novas.

### 3. Execução sem Docker

Com o PostgreSQL já disponível e o `DATABASE_URL` apontando para ele:

```bash
pip install -r requirements.txt    # só as dependências de produção
alembic upgrade head
python -m app
```

### Observações

- **Proxy reverso:** sem `TRUSTED_PROXIES`, a API vê o IP do proxy, e o limite por minuto passa a valer para todos os usuários juntos. Cadastre o IP de quem chega na API: `127.0.0.1` sem Docker; com Docker, o gateway da rede, que aparece em `docker network inspect backend_default` (ex.: `172.18.0.1`). Nunca use `*`: qualquer cliente poderia forjar o IP e fugir do limite.
- O limite de análises por minuto (`RATE_LIMIT_CREATE`) fica na memória do processo (`RATE_LIMIT_STORAGE_URI=memory://`). Ele zera quando a API reinicia e não é compartilhado entre várias instâncias.
- **Análises simultâneas:** o modelo atende uma análise por vez (`NLI_MAX_CONCURRENT=1`), porque já usa todos os núcleos. As outras esperam na fila; depois de `NLI_QUEUE_TIMEOUT_SECONDS`, a API responde `503` e o cliente pode tentar de novo.

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

A análise (`GET /news-analyses/{id}`) vem agrupada por finalidade:

| Bloco | Conteúdo |
|---|---|
| `result` | Classificação, nível de confiança, justificativa e quantas fontes independentes sustentam ou contradizem |
| `evidences` | Publicações em três grupos: `supporting`, `contradicting` e `related` (sobre o assunto, sem posição). Cada uma traz `scores` (relevância e implicação) para auditoria |
| `warnings` | Limitações, sinais de estilo e avaliação do site de origem |
| `verification` | Status da busca, sites consultados e com falha, palavras-chave e versão das regras |

Na listagem, cada item traz só `id`, `analyzed_at`, `news`, `claim` e o resumo de `result`.

## Como a classificação é decidida

| Situação | Resultado |
|---|---|
| Fontes confiáveis contradizem a afirmação | `likely_false` |
| Fontes confiáveis sustentam a afirmação | `likely_true` |
| Fontes divergem, nenhuma evidência, fontes fora do ar ou texto curto | `inconclusive` |

- **Fontes consultadas:** Agência Lupa e Boatos.org (checagem); Folha de S.Paulo, CNN Brasil, Veja e Exame (jornalismo). A lista fica em `app/data/verification_sources.json`.
- **O que conta:** só evidências relevantes, de fontes cadastradas em `app/data/source_registry.json` e independentes entre si.
- **Quando uma reportagem sustenta:** um modelo local de inferência textual ([mDeBERTa multilíngue](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7), licença MIT) estima se a manchete implica a afirmação, mesmo com outras palavras ("anunciado como piloto da Haas na F1" × "contratado para a Fórmula 1"). A manchete também precisa citar os nomes próprios da afirmação, e não pode desmentir o fato ("nega", "descarta").
- **Data:** uma reportagem só sustenta se foi publicada até `EVIDENCE_SUPPORT_WINDOW_DAYS` (60) dias da data da notícia, ou da análise quando a notícia não tem data. Assim, a passagem anterior de um técnico não confirma que ele é "o novo técnico". As mais distantes aparecem como contexto.
- **Quando contradiz:** só pelo veredito das agências de checagem ("é falso que...", "foi gerada por IA", "fora de contexto"). Manchete em forma de pergunta não conta como veredito. O veredito vale sem limite de data, porque boatos antigos voltam a circular. O modelo nunca conclui que algo é falso.
- **Nível de confiança** (não é probabilidade):

  | Nível | Regra |
  |---|---|
  | Alta | Ao menos `CONFIDENCE_HIGH_MIN_SOURCES` (2) fontes independentes concordam, nenhuma discorda e todos os sites responderam |
  | Média | Só 1 fonte independente, ou algum site não respondeu |
  | Baixa | Resultado inconclusivo |
- **O que não decide:** o estilo do texto só gera alertas. Estar em um site conhecido não torna a notícia verdadeira.

## Problemas comuns

| Problema | Solução |
|---|---|
| `No module named 'pydantic'` ao rodar `alembic` | Ative o venv (passo 1). Se não resolver, recrie o `.venv`. |
| Windows: `python` abre a Microsoft Store ou não é encontrado | Instale o Python 3.12 de python.org marcando "Add to PATH", ou use `py -3.12` |
| Windows: `docker` não é reconhecido ou não conecta | Abra o Docker Desktop e espere ele iniciar |
| `/health` retorna 503 ou `connection refused` | Suba o banco: `docker compose up -d db` |
| `relation ... does not exist` ou `column ... does not exist` | `alembic upgrade head` |
| 503 "Muitas análises em andamento" | Fila do modelo cheia. Tente de novo ou aumente `NLI_QUEUE_TIMEOUT_SECONDS` |
| Erro de CORS no front | Adicione a origem em `CORS_ORIGINS` e reinicie a API |
| 429 | Limite de análises por minuto (`RATE_LIMIT_CREATE`) |
| `Não foi possível carregar o modelo NLI` no log | Sem internet na primeira execução ou pouca RAM. A API continua funcionando, só com a comparação por palavras |
