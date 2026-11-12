from datetime import date, datetime

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, ForeignKey, Integer, LargeBinary, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Team(Base):
    __tablename__ = "teams"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    abbr: Mapped[str] = mapped_column(String(3))
    name: Mapped[str] = mapped_column(String(64))
    conference: Mapped[str] = mapped_column(String(4))
    division: Mapped[str] = mapped_column(String(16))


class Player(Base):
    __tablename__ = "players"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    team_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    position: Mapped[str | None] = mapped_column(String(8), nullable=True)
    pos_group: Mapped[str | None] = mapped_column(String(1), nullable=True)  # G / F / C


class Game(Base):
    __tablename__ = "games"
    id: Mapped[str] = mapped_column(String(10), primary_key=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    season: Mapped[str] = mapped_column(String(7), index=True)
    home_team_id: Mapped[int] = mapped_column(Integer)
    away_team_id: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(10))  # scheduled | final


class PlayerGameStat(Base):
    __tablename__ = "player_game_stats"
    player_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_id: Mapped[str] = mapped_column(String(10), ForeignKey("games.id"), primary_key=True, index=True)
    team_id: Mapped[int] = mapped_column(Integer)
    opp_team_id: Mapped[int] = mapped_column(Integer)
    min: Mapped[float] = mapped_column(Float)
    pts: Mapped[int] = mapped_column(Integer)
    reb: Mapped[int] = mapped_column(Integer)
    oreb: Mapped[int] = mapped_column(Integer)
    dreb: Mapped[int] = mapped_column(Integer)
    ast: Mapped[int] = mapped_column(Integer)
    stl: Mapped[int] = mapped_column(Integer)
    blk: Mapped[int] = mapped_column(Integer)
    tov: Mapped[int] = mapped_column(Integer)
    pf: Mapped[int] = mapped_column(Integer)
    fgm: Mapped[int] = mapped_column(Integer)
    fga: Mapped[int] = mapped_column(Integer)
    fg3m: Mapped[int] = mapped_column(Integer)
    fg3a: Mapped[int] = mapped_column(Integer)
    ftm: Mapped[int] = mapped_column(Integer)
    fta: Mapped[int] = mapped_column(Integer)
    plus_minus: Mapped[float | None] = mapped_column(Float, nullable=True)


class TeamGameStat(Base):
    __tablename__ = "team_game_stats"
    team_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_id: Mapped[str] = mapped_column(String(10), ForeignKey("games.id"), primary_key=True, index=True)
    opp_team_id: Mapped[int] = mapped_column(Integer)
    is_home: Mapped[bool] = mapped_column(Boolean)
    min: Mapped[float] = mapped_column(Float)
    pts: Mapped[int] = mapped_column(Integer)
    reb: Mapped[int] = mapped_column(Integer)
    oreb: Mapped[int] = mapped_column(Integer)
    dreb: Mapped[int] = mapped_column(Integer)
    ast: Mapped[int] = mapped_column(Integer)
    tov: Mapped[int] = mapped_column(Integer)
    fgm: Mapped[int] = mapped_column(Integer)
    fga: Mapped[int] = mapped_column(Integer)
    fg3m: Mapped[int] = mapped_column(Integer)
    fg3a: Mapped[int] = mapped_column(Integer)
    ftm: Mapped[int] = mapped_column(Integer)
    fta: Mapped[int] = mapped_column(Integer)
    possessions: Mapped[float] = mapped_column(Float)
    pace: Mapped[float] = mapped_column(Float)
    off_rtg: Mapped[float] = mapped_column(Float)
    def_rtg: Mapped[float] = mapped_column(Float)
    pts_allowed: Mapped[int] = mapped_column(Integer)
    reb_allowed: Mapped[int] = mapped_column(Integer)
    ast_allowed: Mapped[int] = mapped_column(Integer)


class Injury(Base):
    __tablename__ = "injuries"
    player_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    status: Mapped[str] = mapped_column(String(16))  # Out | Doubtful | Questionable | Probable | Day-To-Day
    detail: Mapped[str | None] = mapped_column(String(512), nullable=True)
    source: Mapped[str] = mapped_column(String(16), default="espn")


class Prediction(Base):
    __tablename__ = "predictions"
    player_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    game_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    target: Mapped[str] = mapped_column(String(4), primary_key=True)  # pts | reb | ast
    game_date: Mapped[date] = mapped_column(Date, index=True)
    model_version: Mapped[str] = mapped_column(String(32))
    predicted: Mapped[float] = mapped_column(Float)
    p10: Mapped[float] = mapped_column(Float)
    p90: Mapped[float] = mapped_column(Float)
    baseline: Mapped[float | None] = mapped_column(Float, nullable=True)  # L10 average, for comparison
    actual: Mapped[float | None] = mapped_column(Float, nullable=True)
    context: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # b2b, same_conference, min_l10, ... for segment analysis
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ModelRun(Base):
    __tablename__ = "model_runs"
    version: Mapped[str] = mapped_column(String(32), primary_key=True)
    target: Mapped[str] = mapped_column(String(4), primary_key=True)
    trained_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    n_train: Mapped[int] = mapped_column(Integer)
    n_holdout: Mapped[int] = mapped_column(Integer)
    mae: Mapped[float] = mapped_column(Float)
    rmse: Mapped[float] = mapped_column(Float)
    baseline_mae: Mapped[float] = mapped_column(Float)
    interval_coverage: Mapped[float] = mapped_column(Float)  # share of holdout actuals inside [p10, p90]
    promoted: Mapped[bool] = mapped_column(Boolean, default=False)
    params: Mapped[dict] = mapped_column(JSON)  # hyperparameters this run used (from self-tuning)
    tuning_note: Mapped[str | None] = mapped_column(String(512), nullable=True)
    feature_importance: Mapped[dict] = mapped_column(JSON)
    artifact: Mapped[bytes] = mapped_column(LargeBinary)  # joblib bundle of point + quantile models


class DailyEvaluation(Base):
    """How the pre-game predictions for one date did once the games were played."""

    __tablename__ = "daily_evaluations"
    game_date: Mapped[date] = mapped_column(Date, primary_key=True)
    target: Mapped[str] = mapped_column(String(4), primary_key=True)
    model_version: Mapped[str] = mapped_column(String(32))
    n_players: Mapped[int] = mapped_column(Integer)
    mae: Mapped[float] = mapped_column(Float)
    rmse: Mapped[float] = mapped_column(Float)
    bias: Mapped[float] = mapped_column(Float)  # mean(predicted - actual); > 0 means we projected too high
    baseline_mae: Mapped[float | None] = mapped_column(Float, nullable=True)
    beat_baseline_pct: Mapped[float | None] = mapped_column(Float, nullable=True)  # share of players we were closer on than L10 avg
    close_threshold: Mapped[float] = mapped_column(Float)
    within_close_pct: Mapped[float] = mapped_column(Float)  # share within e.g. +/-3 pts
    within_wide_pct: Mapped[float] = mapped_column(Float)  # share within e.g. +/-5 pts
    interval_coverage: Mapped[float] = mapped_column(Float)  # share inside [p10, p90]; target ~0.8
    segments: Mapped[dict] = mapped_column(JSON)  # {"b2b": {"n":.., "mae":..}, "same_conference": ..., ...}
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ModelConfig(Base):
    """Self-tuned settings per target: the current champion hyperparameters and a live bias correction."""

    __tablename__ = "model_config"
    target: Mapped[str] = mapped_column(String(4), primary_key=True)
    params: Mapped[dict] = mapped_column(JSON)
    bias_correction: Mapped[float] = mapped_column(Float, default=0.0)  # added to raw predictions
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    note: Mapped[str | None] = mapped_column(String(512), nullable=True)
