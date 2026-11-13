from sqlalchemy import inspect, text

from app import models  # noqa: F401  (registers tables)
from app.db import Base, engine
from app.teams import seed_teams


def init_db() -> None:
    Base.metadata.create_all(engine)
    _add_missing_columns()
    seed_teams()


def _add_missing_columns() -> None:
    """create_all never changes an existing table, so add nullable columns that were added to a model since.

    Only nullable columns are added automatically; anything else needs a real migration.
    """
    insp = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            existing = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name not in existing and col.nullable:
                    conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN "{col.name}" {col.type.compile(engine.dialect)}'))


if __name__ == "__main__":
    init_db()
    print(f"Database ready: {engine.url.render_as_string(hide_password=True)}")
