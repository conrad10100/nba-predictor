import io
from functools import lru_cache

import joblib
from sqlalchemy import select

from app.db import SessionLocal
from app.models import ModelRun


def dump(bundle: dict) -> bytes:
    buf = io.BytesIO()
    joblib.dump(bundle, buf, compress=3)
    return buf.getvalue()


def active_version(target: str) -> str | None:
    with SessionLocal() as s:
        return s.scalar(
            select(ModelRun.version).where(ModelRun.target == target, ModelRun.promoted).order_by(ModelRun.trained_at.desc()).limit(1)
        )


@lru_cache(maxsize=8)
def _load(version: str, target: str) -> dict:
    with SessionLocal() as s:
        artifact = s.scalar(select(ModelRun.artifact).where(ModelRun.version == version, ModelRun.target == target))
    return joblib.load(io.BytesIO(artifact))


def load_active(target: str) -> tuple[str, dict] | None:
    """(version, {"point": model, "p10": model, "p90": model, "features": [...]}) for the newest promoted model."""
    version = active_version(target)
    return (version, _load(version, target)) if version else None
