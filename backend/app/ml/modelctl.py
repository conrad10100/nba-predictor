"""Manual model controls: see saved models, pin one, or force a feature set on or off.

python -m app.ml.modelctl list [target]              recent saved models with their holdout accuracy
python -m app.ml.modelctl status                     what each stat is using right now
python -m app.ml.modelctl pin <version> [target]     predict with this saved model (all stats unless one is named)
python -m app.ml.modelctl unpin [target]             back to the newest promoted model
python -m app.ml.modelctl features on|off|auto [target]
    defender features: force on, force off, or let the daily champion/challenger decide (the default)

Every daily retrain is kept in model_runs, so any version listed can be pinned. A pin holds until unpinned; the
daily job keeps training (and self-tuning) in the background, so unpinning picks up the newest model immediately.
"""

import sys

from sqlalchemy import select

from app.config import TARGETS
from app.db import SessionLocal
from app.init_db import init_db
from app.ml import tuning
from app.ml.registry import active_version
from app.models import ModelConfig, ModelRun

SWITCH = "defender_features"


def _config(s, target: str) -> ModelConfig:
    cfg = s.get(ModelConfig, target)
    if cfg is None:
        cfg = ModelConfig(target=target, params=dict(tuning.DEFAULT), bias_correction=0.0, note="created by modelctl")
        s.add(cfg)
    return cfg


def list_runs(targets: list[str], limit: int = 15) -> None:
    with SessionLocal() as s:
        for t in targets:
            print(f"\n{t}:")
            runs = s.execute(
                select(ModelRun.version, ModelRun.mae, ModelRun.baseline_mae, ModelRun.promoted, ModelRun.params)
                .where(ModelRun.target == t).order_by(ModelRun.trained_at.desc()).limit(limit)
            ).all()
            for r in runs:
                flag = "defender features" if (r.params or {}).get(SWITCH) else "no defender features"
                print(f"  {r.version}  MAE {r.mae:.3f}  (L10 {r.baseline_mae:.3f})  promoted={r.promoted}  {flag}")


def status(targets: list[str]) -> None:
    with SessionLocal() as s:
        for t in targets:
            cfg = s.get(ModelConfig, t)
            pinned = cfg.pinned_version if cfg else None
            fixed = (cfg.locked_params or {}) if cfg else {}
            champ = tuning.champion(t)
            print(f"{t}: predicting with {active_version(t)}{' (pinned)' if pinned else ' (newest promoted)'}; "
                  f"defender features {'on' if champ.get(SWITCH) else 'off'}"
                  f"{' (forced)' if SWITCH in fixed else ' (auto)'}")


def pin(version: str, targets: list[str]) -> None:
    with SessionLocal() as s:
        for t in targets:
            if s.scalar(select(ModelRun.version).where(ModelRun.version == version, ModelRun.target == t)) is None:
                sys.exit(f"No saved {t} model with version {version}; see `python -m app.ml.modelctl list {t}`")
        for t in targets:
            _config(s, t).pinned_version = version
        s.commit()
    print(f"Pinned {', '.join(targets)} to {version}. Re-run predictions (python -m jobs.daily --refresh) to apply now.")


def unpin(targets: list[str]) -> None:
    with SessionLocal() as s:
        for t in targets:
            cfg = s.get(ModelConfig, t)
            if cfg:
                cfg.pinned_version = None
        s.commit()
    print(f"Unpinned {', '.join(targets)}: predictions use the newest promoted model.")


def features(mode: str, targets: list[str]) -> None:
    if mode not in ("on", "off", "auto"):
        sys.exit("features takes on, off or auto")
    with SessionLocal() as s:
        for t in targets:
            cfg = _config(s, t)
            fixed = dict(cfg.locked_params or {})
            fixed.pop(SWITCH, None)
            if mode != "auto":
                fixed[SWITCH] = int(mode == "on")
                cfg.params = {**(cfg.params or {}), SWITCH: int(mode == "on")}
            cfg.locked_params = fixed or None
        s.commit()
    print(f"Defender features {mode} for {', '.join(targets)}; takes effect at the next retrain (python -m app.ml.train).")


def main(argv: list[str]) -> None:
    init_db()  # adds the control columns on first use
    if not argv:
        sys.exit(__doc__)
    cmd, args = argv[0], argv[1:]
    if cmd == "list":
        list_runs(args[:1] or TARGETS)
    elif cmd == "status":
        status(TARGETS)
    elif cmd == "pin" and args:
        pin(args[0], args[1:2] or TARGETS)
    elif cmd == "unpin":
        unpin(args[:1] or TARGETS)
    elif cmd == "features" and args:
        features(args[0], args[1:2] or TARGETS)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
