# NBA Player Predictor

Daily projections of NBA player points, rebounds and assists, retrained every morning on the latest box scores,
with what-if controls for injuries and back-to-backs and a running record of how accurate the predictions were.

- `backend/`: Python 3.15, FastAPI, scikit-learn / XGBoost, SQLAlchemy (SQLite locally, Supabase Postgres in production)
- `web/`: Next.js website (slate, player pages, accuracy tracking, model page)

## Run locally

```bash
cd backend
uv venv --python 3.15 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m app.ingest.backfill        # ~1 min: 3 past seasons + this season, schedule, injuries
.venv/bin/python -m app.ml.train               # trains pts/reb/ast (TARGETS=pts to train points only)
.venv/bin/uvicorn app.main:app --reload        # http://localhost:8000/docs

cd ../web
npm install && npm run dev                     # http://localhost:3000
```

Tests: `cd backend && .venv/bin/python -m pytest`. They check that no feature can see the game it's predicting.

## Daily pipeline (`python -m jobs.daily`)

1. Pull the last 3 days of box scores (picks up stat corrections), rosters, schedule and the injury report
2. Fill in actual stats on yesterday's stored predictions
3. **Evaluate** each graded date: average miss, bias, % within ±3 pts / ±2 reb / ±2 ast, range hit rate, whether the
   model beat the last-10 average, broken down by B2B, conference, home/away, starters, teammate out, injury listed
4. **Self-tune**: update the bias correction from the last 14 days of live misses; train today's challenger
   setting (recency half-life, tree size, learning rate, or XGBoost vs HistGradientBoosting) against the
   champion on the last 3 weeks, and keep whichever is more accurate
5. Retrain on all history and predict today's games; store the predictions for tomorrow's evaluation

`python -m jobs.daily --refresh` re-pulls the injury report and re-predicts (no retrain). Run it before tip-off.

## Features

Player: last 5 / last 10 / season / recency-weighted averages of PTS, REB, AST, minutes, FGA, FTA, 3PA, usage;
true shooting, eFG%, 3P%, FT%, 3PA rate, FT rate; per-36 points, shots, rebounds (off/def), assists,
turnovers, fouls; scoring volatility; minutes and shooting trends; home/away splits; history vs this opponent,
including the last meeting.

Matchup: opponent defensive rating, points/rebounds/assists allowed (raw and per 100 possessions), TS% allowed,
FT and 3PA rate allowed, turnovers forced, missed shots (rebound chances), offensive rebound rate, points allowed
to the player's position; both teams' pace and the projected game pace; net-rating gap (blowout risk);
head-to-head history of the two teams.

Context: home/away, rest days, back-to-back, 3 games in 4 nights, road-trip length, opponent rest and B2B,
rest advantage, same conference / division, game number in the season, and production missing from injured
rotation teammates.

## Deploy (Supabase + Railway)

1. Create a Supabase project and copy the pooler connection string into `DATABASE_URL` (see `backend/.env.example`).
2. Run the backfill once against it: `DATABASE_URL=... python -m app.ingest.backfill && python -m app.ml.train`
3. In Railway, create three services from this repo with root directory `backend/`, each with `DATABASE_URL` set:
   - **api**: config file `backend/railway.api.json`; also set `CORS_ORIGINS` to the website's URL
   - **daily**: config file `backend/railway.daily.json` (10:00 UTC, about 5–6am ET, after last night's games are final)
   - **refresh**: config file `backend/railway.refresh.json` (22:00 UTC, about 5–6pm ET, before tip-off)
4. Deploy `web/` (Railway or Vercel) with `NEXT_PUBLIC_API_URL` pointing at the api service.

stats.nba.com sometimes rate-limits or blocks cloud IPs. If the daily job fails on the ingest step, check its logs first.
