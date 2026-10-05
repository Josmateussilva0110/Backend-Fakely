#!/usr/bin/env bash
# Testa a API rodando de ponta a ponta (inclui consultas reais à Lupa, Boatos.org e Folha).
# Uso: ./scripts/smoke_test.sh [BASE_URL]      (padrão: http://localhost:8000)
# Obs.: faz 7 análises; o limite padrão é 10 por minuto (RATE_LIMIT_CREATE).

set -uo pipefail

BASE_URL="${1:-${BASE_URL:-http://localhost:8000}}"
API="$BASE_URL/api/v1"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Chave admin lida do ambiente ou do .env, sem exibi-la
if [[ -z "${ADMIN_API_KEY:-}" && -f "$ROOT_DIR/.env" ]]; then
  ADMIN_API_KEY="$(grep -E '^ADMIN_API_KEY=' "$ROOT_DIR/.env" | cut -d= -f2-)"
fi

PASSED=0
FAILED=0
BODY_FILE="$(mktemp)"
trap 'rm -f "$BODY_FILE"' EXIT

# request MÉTODO URL [JSON] [HEADER] → define STATUS e grava o corpo em $BODY_FILE
request() {
  local method="$1" url="$2" data="${3:-}" header="${4:-}"
  local args=(-s -o "$BODY_FILE" -w "%{http_code}" -X "$method" "$url" -H "Content-Type: application/json")
  [[ -n "$data" ]] && args+=(-d "$data")
  [[ -n "$header" ]] && args+=(-H "$header")
  STATUS="$(curl "${args[@]}")"
}

# json_get CAMPO → lê um campo do último corpo (ex.: classification, verification.matches.0.source_name)
json_get() {
  python3 - "$BODY_FILE" "$1" <<'EOF'
import json, sys
try:
    value = json.load(open(sys.argv[1]))
    for part in sys.argv[2].split("."):
        value = value[int(part)] if isinstance(value, list) else value.get(part)
    if isinstance(value, list):
        value = ", ".join(map(str, value))
    print("" if value is None else value)
except Exception:
    print("")
EOF
}

check() {
  local name="$1" expected_status="$2" expected_classification="${3:-}"
  local classification="" ok=1
  [[ "$STATUS" == "$expected_status" ]] || ok=0
  if [[ -n "$expected_classification" ]]; then
    classification="$(json_get classification)"
    [[ "$classification" == "$expected_classification" ]] || ok=0
  fi
  if (( ok )); then
    PASSED=$((PASSED + 1)); printf '  \033[32mPASSOU\033[0m  %s\n' "$name"
  else
    FAILED=$((FAILED + 1))
    printf '  \033[31mFALHOU\033[0m  %s (status %s, esperado %s' "$name" "$STATUS" "$expected_status"
    [[ -n "$expected_classification" ]] && printf '; classificação "%s", esperado "%s"' "$classification" "$expected_classification"
    printf ')\n'
  fi
}

show_analysis() {
  local label confidence sources match
  label="$(json_get classification_label)"; confidence="$(json_get confidence)"
  sources="$(json_get verification.sources_checked)"; match="$(json_get verification.matches.0.title)"
  printf '          → %s (confiança %s)\n' "$label" "${confidence:-n/a}"
  [[ -n "$sources" ]] && printf '          → sites consultados: %s\n' "$sources"
  [[ -n "$match" ]] && printf '          → melhor correspondência: %s: %s\n' "$(json_get verification.matches.0.source_name)" "${match:0:90}"
  return 0
}

echo "API: $BASE_URL"
request GET "$BASE_URL/health"
if [[ "$STATUS" == "503" ]]; then
  echo "A API está no ar, mas o banco de dados está indisponível."
  echo "Suba o PostgreSQL com: docker compose up -d   (e depois: alembic upgrade head)"; exit 1
elif [[ "$STATUS" != "200" ]]; then
  echo "A API não respondeu em $BASE_URL/health (status ${STATUS:-sem resposta}). Ela está rodando?"; exit 1
fi

echo; echo "== Análises (consultam os sites reais; o resultado pode variar com o conteúdo deles)"
request POST "$API/news" '{"title":"Ivermectina tem 70% de eficácia contra a Covid","text":"Um estudo mostra que a ivermectina tem 70% de eficácia contra a Covid. Compartilhe com todos!"}'
check "Alegação falsa checada pela Lupa" 201 likely_false; show_analysis
CREATED_ID="$(json_get id)"

request POST "$API/news" '{"title":"Quatro em cada dez brasileiros usaram remédios sem eficácia contra a Covid, diz estudo","text":"Levantamento mostra que 40% dos brasileiros usaram medicamentos sem eficácia comprovada contra a Covid durante a pandemia."}'
check "Notícia real publicada na Folha" 201 likely_true; show_analysis

request POST "$API/news" '{"title":"BOMBA!!!","text":"URGENTE!!! COMPARTILHE ANTES QUE APAGUEM! A mídia esconde a cura definitiva que eles não querem que você saiba. Alerta: vão proibir o remédio milagroso!","source":"blogqualquer.net"}'
check "Texto sensacionalista" 201 likely_false; show_analysis

request POST "$API/news" '{"text":"O Ministério da Saúde divulgou nesta segunda-feira o boletim semanal com os dados de vacinação. Segundo o órgão, a cobertura vacinal aumentou em relação ao mês anterior.","url":"https://g1.globo.com/saude/noticia/boletim.ghtml"}'
check "Texto neutro com fonte reconhecida" 201 likely_true; show_analysis

request POST "$API/news" '{"text":"A prefeitura de uma pequena cidade do interior inaugurou hoje uma nova praça com bancos de madeira e iluminação."}'
check "Texto sem correspondência e sem fonte" 201 inconclusive; show_analysis

request POST "$API/news" '{"text":"Isso é falso"}'
check "Texto curto (RN07)" 201 inconclusive

echo; echo "== Validações (devem retornar 422)"
request POST "$API/news" '{"text":"    "}';                                                    check "Texto vazio (RN02)" 422
request POST "$API/news" '{}';                                                                check "Texto ausente" 422
request POST "$API/news" '{"text":"texto com palavras suficientes aqui","url":"nao-e-url"}';   check "URL inválida" 422
request POST "$API/news" '{"text":"texto com palavras suficientes aqui","published_at":"2999-01-01"}'; check "Data no futuro" 422
request POST "$API/news" '{"text":"texto com palavras suficientes aqui","classification":"likely_true"}'; check "Campo controlado pelo servidor" 422
printf '          → mensagem: %s\n' "$(json_get errors.0.message)"

echo; echo "== Consultas"
request GET "$API/news?page=1&page_size=5";                          check "Listagem paginada" 200
printf '          → total: %s | páginas: %s\n' "$(json_get total)" "$(json_get pages)"
request GET "$API/news?classification=likely_false&order_by=-confidence"; check "Filtro + ordenação" 200
if [[ -n "$CREATED_ID" ]]; then
  request GET "$API/news/$CREATED_ID";                               check "Detalhe da notícia $CREATED_ID" 200
fi
request GET "$API/news/999999999";                                   check "Notícia inexistente" 404
request GET "$API/news?page_size=1000";                              check "page_size acima do máximo" 422
request GET "$API/news?order_by=text";                               check "Ordenação fora da lista branca" 422
request GET "$API/news?campo=x";                                     check "Parâmetro desconhecido" 422

echo; echo "== Segurança"
request GET "$BASE_URL/health"
HEADERS="$(curl -s -D - -o /dev/null "$BASE_URL/health")"
if grep -qi "x-content-type-options: nosniff" <<<"$HEADERS"; then STATUS=ok; else STATUS=ausente; fi
check "Headers de segurança presentes" ok
if [[ -z "$CREATED_ID" ]]; then
  echo "  (nenhuma notícia foi criada: testes de PATCH/DELETE ignorados)"
else
  request PATCH "$API/news/$CREATED_ID" '{"title":"x"}';              check "PATCH sem chave" 403
  request DELETE "$API/news/$CREATED_ID" '' "X-API-Key: chave-errada"; check "DELETE com chave errada" 403
fi

if [[ -n "${ADMIN_API_KEY:-}" && -n "$CREATED_ID" ]]; then
  request POST "$API/news" '{"text":"Notícia temporária criada pelo smoke test para validar edição e remoção."}'
  TEMP_ID="$(json_get id)"
  request PATCH "$API/news/$TEMP_ID" '{"title":"Título editado"}' "X-API-Key: $ADMIN_API_KEY"; check "PATCH com chave (reanalisa)" 200
  request DELETE "$API/news/$TEMP_ID" '' "X-API-Key: $ADMIN_API_KEY";                        check "DELETE com chave" 204
  request GET "$API/news/$TEMP_ID";                                                            check "Removida não existe mais" 404
elif [[ -z "${ADMIN_API_KEY:-}" ]]; then
  echo "  (ADMIN_API_KEY não definida: testes de PATCH/DELETE com chave ignorados)"
fi

echo; printf 'Resultado: \033[32m%d passaram\033[0m, \033[31m%d falharam\033[0m\n' "$PASSED" "$FAILED"
(( FAILED == 0 ))
