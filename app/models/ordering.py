from sqlalchemy import ColumnElement
from sqlalchemy.orm import InstrumentedAttribute


def order_clause(columns: dict[str, InstrumentedAttribute], order_by: str) -> ColumnElement:
    """Converte "campo" / "-campo" em ORDER BY; o campo já foi validado contra a lista branca."""
    column = columns[order_by.lstrip("-")]
    return column.desc() if order_by.startswith("-") else column.asc()
