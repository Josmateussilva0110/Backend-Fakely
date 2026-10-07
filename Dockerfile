FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY migrations ./migrations
COPY alembic.ini .

# Executa sem privilégios de root
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000
CMD ["sh", "-c", "python -m alembic upgrade head && python -m app"]
