from datetime import date, datetime


def parse_iso_date(value: str | None) -> date | None:
    """Data de um valor ISO 8601 (com ou sem hora); None se ausente ou inválido."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.strip()).date()
    except ValueError:
        return None
