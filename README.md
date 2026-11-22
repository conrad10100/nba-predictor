# NBA Player Predictor

Daily projections of NBA player points, rebounds and assists, retrained every morning on the latest box scores,
with what-if controls for injuries and back-to-backs and a running record of how accurate the predictions were.

- `backend/`: Python 3.15, FastAPI, scikit-learn / XGBoost, SQLAlchemy (SQLite locally, Supabase Postgres in production)
- `web/`: Next.js website (slate, player pages, accuracy tracking, over/under performance vs. DraftKings, model page)

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

1. Pull the last 3 days of box scores (picks up stat corrections), rosters, schedule and the injury report, plus
   today's player defensive ratings (DBPM) from Basketball-Reference (one page; past seasons once)
2. Fill in actual stats on yesterday's stored predictions
3. **Evaluate** each graded date: average miss, bias, % within ±3 pts / ±2 reb / ±2 ast, range hit rate, whether the
   model beat the last-10 average, broken down by B2B, conference, home/away, starters, teammate out, injury listed
4. **Self-tune**: update the bias correction from the last 14 days of live misses; train today's challenger
   setting (recency half-life, tree size, learning rate, or XGBoost vs HistGradientBoosting) against the
   champion on the last 3 weeks, and keep whichever is more accurate (a challenger must win by 0.5%+)
5. Retrain on all history and predict today's games; store the predictions for tomorrow's evaluation

Each run also stores DraftKings points/rebounds/assists player prop lines from ESPN (`player_props`): today's lines
before tip-off, yesterday's closing lines, and any finished dates this season that have none yet (a one-time backfill).
Each pre-game run also freezes every pick (line, prices, prediction) for games that haven't started
(`pick_snapshots`): the morning run as the "morning" picks, the last injury update before each game's tip-off as
the final picks. The
**Performance** page grades the morning picks and **After injuries** the final ones, so a pick can never change once
its game starts. Each grades the prediction as an over/under pick against that line: over when the prediction is
above the line, under when below; it wins when the player finishes on that side, and landing exactly on it is a push.

`python -m jobs.daily --refresh` is the injury update: re-pulls the injury report and DraftKings lines and
re-predicts (no retrain). With `--before-tip 30` it only does so when a game tips off within 30 minutes (per ESPN's
scoreboard) and otherwise exits in a few seconds, so running it every 15 minutes gives each game a final pick taken
15 to 30 minutes before its own tip-off: a 7:30 game uses the injury report as of about 7:15, a 10:30 game as of
about 10:15. Games that already started are never re-frozen.

On the **After injuries** page a night's top 5 is settled game by game. A game's last refresh (the one within 15
minutes of its tip-off, or tip-off itself if that refresh didn't run) locks its picks that are in the top 5 at that
moment (`top5_locks`; its other picks drop out, `locked_games`) and stops re-freezing its snapshots, so a locked pick
keeps its slot and its line. Later games only compete for the slots still open. Until its game is settled a top-5
pick shows as Provisional: a later refresh can still change it or push it out. The morning page is frozen all at
once and ranks the whole night.

Every run also logs the top 5 it leaves behind (`top5_seen`): each pick's first and last sighting, best rank, and
the line, side and price it first showed with. The After injuries page lists everyone who made a night's top 5,
graded at that first line, so a pick that showed at 6:30 and dropped out by 10:00 is still on record.

## Features

Player: last 5 / last 10 / season / recency-weighted averages of PTS, REB, AST, minutes, FGA, FTA, 3PA, usage;
true shooting, eFG%, 3P%, FT%, 3PA rate, FT rate; per-36 points, shots, rebounds (off/def), assists,
turnovers, fouls; scoring volatility; minutes and shooting trends; home/away splits; history vs this opponent,
including the last meeting.

Matchup: opponent defensive rating, points/rebounds/assists allowed (raw and per 100 possessions), TS% allowed,
FT and 3PA rate allowed, turnovers forced, missed shots (rebound chances), offensive rebound rate, points allowed
to the player's position; both teams' pace and the projected game pace; net-rating gap (blowout risk);
head-to-head history of the two teams.

Defenders (only while they win the weekly shadow test; see below): each player's defensive box plus/minus (DBPM)
from Basketball-Reference, as published before the game (this season's snapshot so far blended with last season's
final). Features: tonight's opposing players weighted by minutes, the same for the player's position, tonight vs.
the opponent's usual lineup (a good defender out), and the opponent's top-minutes player at the same position (the
likeliest primary defender). Who-guarded-whom tracking only exists on stats.nba.com, which blocks Railway.

Context: home/away, rest days, back-to-back, 3 games in 4 nights, road-trip length, opponent rest and B2B,
rest advantage, same conference / division, game number in the season, and production missing from injured
rotation teammates.

## Shadow test (new feature sets)

Feature sets such as the defender ratings don't switch on from a backtest. Each retrain also trains a **shadow**
model (the live model with the switch flipped, stored in `shadow_models`), and every prediction run stores its
predictions in `shadow_predictions`, apart from the live ones, so picks and the Performance page never use them.
On the first morning run 7+ days into a test, both are compared on the games played since (same players, same
final pre-game run): average miss on DraftKings-line players decides (the shadow must be 0.5%+ better over at least
150 of them), with over/under hit rate against the line shown alongside. Then a new week starts, so a switch can
also be undone a week later. The site's **A/B test** page (`/ab-test`) shows the running week and every weekly result; `python -m app.ml.modelctl shadow` prints the same.

## Model versions and reverting

Every retrain is saved in `nba.model_runs` (one row per version and stat: accuracy, settings, the features it uses
in `feature_importance`, and the model itself in `artifact`). Predictions use the newest promoted model unless one
is pinned.

```bash
python -m app.ml.modelctl status                  # what each stat is predicting with right now
python -m app.ml.modelctl list [pts]              # recent saved models and their accuracy
python -m app.ml.modelctl pin 20261117-1002 [pts] # predict with that saved model until unpinned
python -m app.ml.modelctl unpin [pts]
python -m app.ml.modelctl features on|off|auto    # force the defender features, or let the weekly shadow test decide
python -m app.ml.modelctl shadow                  # shadow test: last weekly verdict and this week so far
python -m app.ingest.defense                      # store defensive ratings now (past seasons + today's snapshot)
python -m app.ml.compare [weeks]                  # walk-forward MAE with vs. without the defender features
```

## Deployment

Production runs entirely in the cloud; nothing depends on a laptop being on.

- **Database:** Supabase project `nba-predictor` (us-east-1). The app connects as role `nba_app` to a private
  `nba` schema that is not exposed through Supabase's REST API.
- **Railway project `nba-predictor`:** each service is built from this repo with Railpack:

| Service | Root | Start command | Schedule |
|---|---|---|---|
| api | `/backend` | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` | always on, healthcheck `/api/health` |
| daily | `/backend` | `python -m jobs.daily` | `0 10 * * *` (10:00 UTC, about 5–6am ET) |
| refresh | `/backend` | `python -m jobs.daily --refresh --before-tip 30` | `*/15 15-23,0-5 * * *` (every 15 minutes, 10am–1am ET in winter; skips quickly when no game is about to tip) |
| web | `/web` | `npm run start` | always on |

Variables: backend services get `DATABASE_URL` and `CURRENT_SEASON`; api also gets `CORS_ORIGINS` (the web URL);
web gets `NEXT_PUBLIC_API_URL` (the api URL, baked in at build time). Pushing to `main` redeploys the affected services.

stats.nba.com sometimes rate-limits or blocks cloud IPs. If the daily job fails on the ingest step, check its logs first.
