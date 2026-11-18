import io
from functools import lru_cache

import joblib
from sqlalchemy import select

from app.db import SessionLocal
from app.models import ModelConfig, ModelRun


def dump(bundle: dict) -> bytes:
    buf = io.BytesIO()
    joblib.dump(bundle, buf, compress=3)
    return buf.getvalue()


def active_version(target: str) -> str | None:
    """The pinned version if one is set (python -m app.ml.modelctl pin), else the newest promoted model."""
    with SessionLocal() as s:
        cfg = s.get(ModelConfig, target)
        if cfg is not None and cfg.pinned_version:
            exists = s.scalar(select(ModelRun.version).where(ModelRun.version == cfg.pinned_version, ModelRun.target == target))
            if exists:
                return exists
        return s.scalar(
            select(ModelRun.version).where(ModelRun.target == target, ModelRun.promoted).order_by(ModelRun.trained_at.desc()).limit(1)
        )


def _load_bytes(artifact: bytes) -> dict:
    return joblib.load(io.BytesIO(artifact))


@lru_cache(maxsize=8)
def _load(version: str, target: str) -> dict:
    with SessionLocal() as s:
        artifact = s.scalar(select(ModelRun.artifact).where(ModelRun.version == version, ModelRun.target == target))
    return joblib.load(io.BytesIO(artifact))


def load_active(target: str) -> tuple[str, dict] | None:
    """(version, {"point": model, "interval": ..., "features": [...]}) for the pinned or newest promoted model."""
    version = active_version(target)
    return (version, _load(version, target)) if version else None
