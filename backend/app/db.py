from collections.abc import Iterable

from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import DATABASE_URL

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def upsert(table, rows: Iterable[dict]) -> int:
    """Insert rows, updating non-key columns on primary-key conflict. Works on SQLite and Postgres."""
    rows = list(rows)
    if not rows:
        return 0
    insert = pg_insert if engine.dialect.name == "postgresql" else sqlite_insert
    keys = [c.name for c in table.primary_key.columns]
    chunk = max(1, 30000 // len(rows[0]))  # stay under SQLite's 32,766 bound-parameter limit
    with engine.begin() as conn:
        for i in range(0, len(rows), chunk):
            batch = rows[i : i + chunk]
            stmt = insert(table).values(batch)
            update_cols = {c: stmt.excluded[c] for c in batch[0] if c not in keys}
            stmt = stmt.on_conflict_do_update(index_elements=keys, set_=update_cols) if update_cols else stmt.on_conflict_do_nothing()
            conn.execute(stmt)
    return len(rows)
