from collections.abc import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


def enable_sqlite_foreign_keys(engine) -> None:
    # SQLite (usado nos testes) só aplica ON DELETE CASCADE com esta opção ligada
    @event.listens_for(engine, "connect")
    def _set_pragma(dbapi_connection, _):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")


def build_engine(url: str):
    options: dict = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        options["connect_args"] = {"check_same_thread": False}
    engine = create_engine(url, **options)
    if url.startswith("sqlite"):
        enable_sqlite_foreign_keys(engine)
    return engine


engine = build_engine(get_settings().database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

