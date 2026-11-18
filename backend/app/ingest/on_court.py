"""Who was on the floor, from ESPN play-by-play: per player-game on-court seconds and opponent points meanwhile.

These feed the defender on/off ratings (app.features.defenders). One ESPN summary request per game, so history is
filled in gradually: each daily run processes last night's games plus a batch of older ones, and
`python -m app.ingest.on_court [max_games]` processes more by hand. Games already processed are skipped, so any run
can be stopped and resumed.

Lineups: ESPN logs substitutions ("A enters the game for B") but not who starts each quarter. A period's starting
five per team are the players who act in it (or are subbed out) before being subbed in; any still missing carry over
from the end of the previous period (the box-score starters for the 1st). The usual play-by-play caveats apply:
points scored on free throws right after a substitution count for the new lineup.
"""

import logging
import re
import sys
import time
from collections import defaultdict
from datetime import date

from sqlalchemy import select

from app.config import BACKFILL_SEASONS
from app.db import SessionLocal, upsert
from app.ingest import nba_client
from app.models import Game, OnCourtFetch, Player, PlayerOnCourt, Team

log = logging.getLogger(__name__)

DAILY_BACKFILL_GAMES = 150  # older games processed per daily run (one ESPN request each)
MAX_CONSECUTIVE_FAILURES = 3
SUB_RE = re.compile(r"^(?P<enter>.+?) enters the game for (?P<leave>.+?)\.?$")


# ---------------------------------------------------------------- parsing (pure)


def _period_len(period: int) -> int:
    return 720 if period <= 4 else 300


def _clock_seconds(v) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    m, _, s = str(v or "0").partition(":")
    return float(m or 0) * 60 + float(s or 0) if s else float(m or 0)


def parse_summary(summary: dict) -> dict | None:
    """{"teams": {espn_team_id: abbr}, "home": id, "players": {espn_athlete_id: {name, team, on_seconds, opp_pts_on}},
    "game_seconds", "pts": {espn_team_id: points}} or None if the game has no usable play-by-play."""
    box = summary.get("boxscore", {}).get("players", [])
    plays = summary.get("plays") or []
    comps = (summary.get("header", {}).get("competitions") or [{}])[0].get("competitors", [])
    if len(box) != 2 or not plays or len(comps) != 2:
        return None
    home = next(str(c["team"]["id"]) for c in comps if c.get("homeAway") == "home")
    teams, team_of, name_to_id, starters, names = {}, {}, defaultdict(dict), defaultdict(list), {}
    for t in box:
        tid = str(t["team"]["id"])
        teams[tid] = nba_client.ESPN_TO_NBA_ABBR.get(t["team"]["abbreviation"], t["team"]["abbreviation"])
        for a in t["statistics"][0]["athletes"]:
            aid = str(a["athlete"]["id"])
            team_of[aid] = tid
            names[aid] = a["athlete"]["displayName"]
            for key in ("displayName", "shortName"):
                if a["athlete"].get(key):
                    name_to_id[tid][nba_client.normalize_name(a["athlete"][key])] = aid
            if a.get("starter"):
                starters[tid].append(aid)
    away = next(t for t in teams if t != home)

    def resolve(name: str, tid: str | None) -> str | None:
        n = nba_client.normalize_name(name)
        if tid and n in name_to_id[tid]:
            return name_to_id[tid][n]
        hits = [d[n] for d in name_to_id.values() if n in d]
        return hits[0] if len(hits) == 1 else None

    events = []  # (period, elapsed_in_period, home_score, away_score, sub: (team, enter, leave) | None, actors)
    for p in plays:
        period = int((p.get("period") or {}).get("number") or 0)
        if period < 1:
            continue
        t = _period_len(period) - _clock_seconds((p.get("clock") or {}).get("displayValue"))
        tid = str((p.get("team") or {}).get("id") or "") or None
        sub = None
        m = SUB_RE.match((p.get("text") or "").strip())
        if m or "substitution" in ((p.get("type") or {}).get("text") or "").lower():
            ids = [str(x["athlete"]["id"]) for x in p.get("participants", []) if x.get("athlete")]
            enter = resolve(m.group("enter"), tid) if m else None
            leave = resolve(m.group("leave"), tid) if m else None
            enter = enter or (ids[0] if ids else None)
            leave = leave or (ids[1] if len(ids) > 1 else None)
            team = tid or team_of.get(enter or leave or "")
            if enter and leave and team:
                sub = (team, enter, leave)
        actors = [str(x["athlete"]["id"]) for x in p.get("participants", []) if x.get("athlete")]
        events.append((period, max(0.0, t), int(p.get("homeScore") or 0), int(p.get("awayScore") or 0), sub, actors))
    if not events:
        return None

    on_sec, opp_on = defaultdict(float), defaultdict(int)
    lineup = {tid: list(starters[tid][:5]) for tid in teams}
    score = {home: 0, away: 0}
    game_seconds = 0.0
    n_periods = max(e[0] for e in events)
    for period in range(1, n_periods + 1):
        evs = [e for e in events if e[0] == period]
        lineup = _period_starters(evs, lineup, team_of, teams)
        last_t = 0.0
        for _, t, hs, as_, sub, _ in evs + [(period, _period_len(period), None, None, None, [])]:
            dt = max(0.0, t - last_t)
            for tid in teams:
                for pid in lineup[tid]:
                    on_sec[pid] += dt
            if hs is not None:
                gained = {home: hs - score[home], away: as_ - score[away]}
                if gained[home] >= 0 and gained[away] >= 0:
                    for tid, other in ((home, away), (away, home)):
                        for pid in lineup[tid]:
                            opp_on[pid] += gained[other]
                score = {home: hs, away: as_}
            last_t = max(last_t, t)
            if sub:
                team, enter, leave = sub
                cur = lineup.get(team, [])
                if leave in cur:
                    cur.remove(leave)
                if enter not in cur:
                    cur.append(enter)
                lineup[team] = cur[-5:] if len(cur) > 5 else cur
        game_seconds += _period_len(period)
    players = {
        pid: {"name": names.get(pid, ""), "team": team_of[pid], "on_seconds": on_sec[pid], "opp_pts_on": opp_on[pid]}
        for pid in team_of if on_sec[pid] > 0
    }
    return {"teams": teams, "home": home, "players": players, "game_seconds": game_seconds, "pts": dict(score)}


def _period_starters(evs, prev: dict, team_of: dict, teams: dict) -> dict:
    found = {tid: [] for tid in teams}
    subbed_in = set()
    for _, _, _, _, sub, actors in evs:
        if sub:
            team, enter, leave = sub
            if team in found and leave not in subbed_in and leave not in found[team] and len(found[team]) < 5:
                found[team].append(leave)
            subbed_in.add(enter)
            continue
        for pid in actors:
            tid = team_of.get(pid)
            if tid and pid not in subbed_in and pid not in found[tid] and len(found[tid]) < 5:
                found[tid].append(pid)
    for tid in teams:  # fill from the previous lineup, skipping anyone who only came in later this period
        for pid in prev.get(tid, []):
            if len(found[tid]) >= 5:
                break
            if pid not in found[tid] and pid not in subbed_in:
                found[tid].append(pid)
    return found


# ---------------------------------------------------------------- loading


def pending_games(limit: int | None = None) -> list[tuple[str, date, int, int]]:
    """Final games not processed yet, newest first: (game_id, date, home_team_id, away_team_id)."""
    with SessionLocal() as s:
        q = (
            select(Game.id, Game.date, Game.home_team_id, Game.away_team_id)
            .outerjoin(OnCourtFetch, OnCourtFetch.game_id == Game.id)
            .where(Game.status == "final", Game.season.in_(BACKFILL_SEASONS), OnCourtFetch.game_id.is_(None))
            .order_by(Game.date.desc(), Game.id)
        )
        return [tuple(r) for r in s.execute(q.limit(limit) if limit else q).all()]


def load_on_court(games: list[tuple[str, date, int, int]], time_budget_s: float | None = None) -> tuple[int, int, set[str]]:
    """Process the listed games. Returns (games processed, player rows stored, ESPN names that matched no player)."""
    with SessionLocal() as s:
        abbr = {t.id: t.abbr for t in s.scalars(select(Team))}
        roster = s.execute(select(Player.id, Player.name, Player.team_id)).all()
    by_name: dict[str, list[tuple[int, int | None]]] = defaultdict(list)
    for pid, name, tid in roster:
        by_name[nba_client.normalize_name(name)].append((pid, tid))

    start, done, n_rows, failures, unmatched = time.monotonic(), 0, 0, 0, set()
    by_date: dict[date, list] = defaultdict(list)
    for g in games:
        by_date[g[1]].append(g)
    for day in sorted(by_date, reverse=True):
        if time_budget_s is not None and time.monotonic() - start > time_budget_s:
            break
        try:
            events = {(e["home"], e["away"]): e["event_id"] for e in nba_client.espn_completed_events(day)}
        except Exception:
            failures += 1
            log.warning("on-court %s: scoreboard failed", day, exc_info=failures == 1)
            if failures >= MAX_CONSECUTIVE_FAILURES:
                break
            continue
        for game_id, _, home_id, away_id in by_date[day]:
            event_id = events.get((abbr.get(home_id), abbr.get(away_id)))
            rows = []
            if event_id is not None:
                try:
                    parsed = parse_summary(nba_client.espn_summary(event_id))
                except Exception:
                    failures += 1
                    log.warning("on-court %s: summary failed", game_id, exc_info=failures == 1)
                    if failures >= MAX_CONSECUTIVE_FAILURES:
                        return done, n_rows, unmatched
                    continue
                failures = 0
                if parsed:
                    rows = _rows(game_id, parsed, abbr, by_name, home_id, away_id, unmatched)
            upsert(PlayerOnCourt.__table__, rows)
            upsert(OnCourtFetch.__table__, [{"game_id": game_id, "n_rows": len(rows)}])
            done, n_rows = done + 1, n_rows + len(rows)
    return done, n_rows, unmatched


def _rows(game_id, parsed, abbr, by_name, home_id, away_id, unmatched) -> list[dict]:
    team_id = {espn: (home_id if a == abbr.get(home_id) else away_id if a == abbr.get(away_id) else None)
               for espn, a in parsed["teams"].items()}
    if None in team_id.values():
        return []
    pts = {team_id[espn]: p for espn, p in parsed["pts"].items()}
    rows = []
    for p in parsed["players"].values():
        tid = team_id[p["team"]]
        cands = by_name.get(nba_client.normalize_name(p["name"]), [])
        match = next((pid for pid, t in cands if t == tid), cands[0][0] if len(cands) == 1 else None)
        if match is None:
            unmatched.add(p["name"])
            continue
        opp = away_id if tid == home_id else home_id
        rows.append({"game_id": game_id, "player_id": int(match), "team_id": int(tid), "on_seconds": float(p["on_seconds"]),
                     "opp_pts_on": int(p["opp_pts_on"]), "game_seconds": float(parsed["game_seconds"]),
                     "opp_pts_game": int(pts.get(opp, 0))})
    return rows


if __name__ == "__main__":
    from app.init_db import init_db

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    init_db()
    todo = pending_games(int(sys.argv[1]) if len(sys.argv) > 1 else None)
    print(f"{len(todo)} games to process", flush=True)
    total = 0
    for i in range(0, len(todo), 100):
        g, n, unmatched = load_on_court(todo[i : i + 100])
        total += g
        print(f"  {total}/{len(todo)} games, {n} player rows in this batch, unmatched {sorted(unmatched)[:5]}", flush=True)
        if g == 0:
            break
