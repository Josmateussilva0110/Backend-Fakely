import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


# Mensagens em português para os erros de validação mais comuns do Pydantic
VALIDATION_MESSAGES = {
    "missing": "Campo obrigatório.",
    "extra_forbidden": "Campo não permitido.",
    "string_type": "Deve ser um texto.",
    "string_too_short": "Deve ter pelo menos {min_length} caractere(s).",
    "string_too_long": "Deve ter no máximo {max_length} caracteres.",
    "url_parsing": "URL inválida.",
    "url_scheme": "A URL deve começar com http:// ou https://.",
    "url_too_long": "URL muito longa (máximo de {max_length} caracteres).",
    "date_parsing": "Data inválida (use o formato AAAA-MM-DD).",
    "date_from_datetime_parsing": "Data inválida (use o formato AAAA-MM-DD).",
    "datetime_parsing": "Data e hora inválidas (use o formato ISO 8601).",
    "datetime_from_date_parsing": "Data e hora inválidas (use o formato ISO 8601).",
    "int_parsing": "Deve ser um número inteiro.",
    "int_type": "Deve ser um número inteiro.",
    "greater_than_equal": "Deve ser maior ou igual a {ge}.",
    "less_than_equal": "Deve ser menor ou igual a {le}.",
    "enum": "Valor inválido. Use um destes: {expected}.",
    "literal_error": "Valor inválido. Use um destes: {expected}.",
    "json_invalid": "JSON inválido.",
    "model_attributes_type": "Formato inválido.",
}


def translate_validation_error(error: dict) -> str:
    template = VALIDATION_MESSAGES.get(error["type"])
    if template is None:
        # Erros dos nossos validadores já vêm em português
        return error["msg"].removeprefix("Value error, ")
    try:
        return template.format(**error.get("ctx", {}))
    except (KeyError, IndexError):
        return error["msg"]


class AppError(Exception):
    status_code = 400

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class NotFoundError(AppError):
    status_code = 404


class ForbiddenError(AppError):
    status_code = 403


class BusinessRuleError(AppError):
    status_code = 409


class UnprocessableError(AppError):
    # Entrada bem formada, mas que não pode ser processada (ex.: URL bloqueada ou ilegível)
    status_code = 422


def error_response(status_code: int, detail: str, errors: list[dict] | None = None) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail, "errors": errors or []})


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError):
        return error_response(exc.status_code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError):
        errors = [
            {
                "field": ".".join(str(part) for part in err["loc"] if part not in ("body", "query")),
                "message": translate_validation_error(err),
            }
            for err in exc.errors()
        ]
        return error_response(422, "Dados de entrada inválidos.", errors)

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(_: Request, exc: StarletteHTTPException):
        return error_response(exc.status_code, str(exc.detail))

    @app.exception_handler(RateLimitExceeded)
    async def handle_rate_limit(_: Request, __: RateLimitExceeded):
        return error_response(429, "Muitas requisições. Tente novamente mais tarde.")

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception):
        # Detalhes só no log; o cliente recebe mensagem genérica
        logger.exception("Erro não tratado em %s %s", request.method, request.url.path, exc_info=exc)
        return error_response(500, "Erro interno do servidor.")
