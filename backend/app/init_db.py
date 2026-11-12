from app import models  # noqa: F401  (registers tables)
from app.db import Base, engine
from app.teams import seed_teams


def init_db() -> None:
    Base.metadata.create_all(engine)
    seed_teams()


if __name__ == "__main__":
    init_db()
    print(f"Database ready: {engine.url.render_as_string(hide_password=True)}")
