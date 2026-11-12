import os

from dotenv import load_dotenv

load_dotenv()

# Local dev falls back to SQLite; production points at Supabase Postgres.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///nba.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

BACKFILL_SEASONS = os.getenv("BACKFILL_SEASONS", "2023-24,2024-25,2025-26,2026-27").split(",")
CURRENT_SEASON = os.getenv("CURRENT_SEASON", "2026-27")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

# Which stats to train/predict. Set TARGETS=pts to iterate on points alone.
TARGETS = [t.strip() for t in os.getenv("TARGETS", "pts,reb,ast").split(",") if t.strip()]
