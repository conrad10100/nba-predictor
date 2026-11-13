"""Thin wrapper over nba_api + ESPN with rate limiting and retries."""

import re
import time
import unicodedata
from datetime import date

import pandas as pd
import requests
from nba_api.stats.endpoints import leaguegamelog, playerindex, scheduleleaguev2

REQUEST_GAP_S = 1.0
_last_call = 0.0

ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
# ESPN abbreviations that differ from the NBA's.
ESPN_TO_NBA_ABBR = {"GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}


def _call(fn, retries: int = 4):
    global _last_call
    for attempt in range(retries):
        wait = REQUEST_GAP_S - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            _last_call = time.monotonic()
            return fn()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))


def player_game_logs(season: str) -> pd.DataFrame:
    return _call(lambda: leaguegamelog.LeagueGameLog(season=season, player_or_team_abbreviation="P", timeout=60).get_data_frames()[0])


def team_game_logs(season: str) -> pd.DataFrame:
    return _call(lambda: leaguegamelog.LeagueGameLog(season=season, player_or_team_abbreviation="T", timeout=60).get_data_frames()[0])


def season_schedule(season: str) -> pd.DataFrame:
    return _call(lambda: scheduleleaguev2.ScheduleLeagueV2(season=season, timeout=60).get_data_frames()[0])


def player_index(season: str) -> pd.DataFrame:
    return _call(lambda: playerindex.PlayerIndex(season=season, timeout=60).get_data_frames()[0])


def espn_injuries() -> list[dict]:
    """Flat list of {name, team, status, detail, date} from ESPN's public injury feed."""
    data = _call(lambda: requests.get(ESPN_INJURIES_URL, timeout=30).json())
    out = []
    for team in data.get("injuries", []):
        for inj in team.get("injuries", []):
            out.append({
                "name": inj["athlete"]["displayName"],
                "team": team.get("displayName"),
                "status": inj.get("status", "Unknown"),
                "detail": inj.get("shortComment") or inj.get("longComment"),
            })
    return out


def espn_game_lines(on: date) -> list[dict]:
    """Spread and total for each game on a date that still has odds posted: [{home, away, home_spread, total, provider}]."""
    data = _call(lambda: requests.get(ESPN_SCOREBOARD_URL, params={"dates": on.strftime("%Y%m%d")}, timeout=30).json())
    out = []
    for event in data.get("events", []):
        comp = event["competitions"][0]
        odds = (comp.get("odds") or [{}])[0]
        if odds.get("overUnder") is None or not odds.get("details"):
            continue
        side = {t["homeAway"]: ESPN_TO_NBA_ABBR.get(t["team"]["abbreviation"], t["team"]["abbreviation"]) for t in comp["competitors"]}
        home_spread = _home_spread(odds["details"], side["home"])
        if home_spread is None:
            continue
        out.append({"home": side["home"], "away": side["away"], "home_spread": home_spread,
                    "total": float(odds["overUnder"]), "provider": (odds.get("provider") or {}).get("name")})
    return out


ESPN_CORE_URL = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"
ESPN_ROSTER_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/roster"
DRAFTKINGS_PROVIDER_ID = 100  # ESPN's odds provider id for DraftKings
# ESPN prop market name -> our stat. Each player gets two rows per market (over and under) at the same line.
PROP_MARKETS = {"Total Points": "pts", "Total Rebounds": "reb", "Total Assists": "ast"}


def espn_player_props(on: date) -> list[dict]:
    """DraftKings over/under lines for points, rebounds and assists on a date's games, via ESPN.

    [{home, away, state, espn_team_ids, espn_athlete_id, target, line, open_line, updated}]. `state` is the game's
    ESPN state (pre / in / post). Once a game is over ESPN keeps showing the last pre-game (closing) line.
    """
    board = _call(lambda: requests.get(ESPN_SCOREBOARD_URL, params={"dates": on.strftime("%Y%m%d")}, timeout=30).json())
    out = []
    for event in board.get("events", []):
        comp = event["competitions"][0]
        side = {t["homeAway"]: t["team"] for t in comp["competitors"]}
        abbr = {k: ESPN_TO_NBA_ABBR.get(t["abbreviation"], t["abbreviation"]) for k, t in side.items()}
        seen = set()
        for item in _prop_bets(event["id"]):
            target = PROP_MARKETS.get(item.get("type", {}).get("name"))
            athlete = _ref_id(item.get("athlete", {}).get("$ref", ""), "athletes")
            line = (item.get("current") or {}).get("target", {}).get("value")
            if target is None or athlete is None or line is None or (athlete, target) in seen:
                continue
            seen.add((athlete, target))
            out.append({
                "home": abbr["home"], "away": abbr["away"], "state": event["status"]["type"]["state"],
                "espn_team_ids": [side["home"]["id"], side["away"]["id"]], "espn_athlete_id": athlete, "target": target,
                "line": float(line), "open_line": _float((item.get("open") or {}).get("target", {}).get("value")),
                "updated": item.get("lastUpdated"),
            })
    return out


def _prop_bets(event_id: str) -> list[dict]:
    url = f"{ESPN_CORE_URL}/events/{event_id}/competitions/{event_id}/odds/{DRAFTKINGS_PROVIDER_ID}/propBets"
    items, page = [], 1
    while True:
        resp = _call(lambda: requests.get(url, params={"lang": "en", "region": "us", "limit": 1000, "page": page}, timeout=30))
        if resp.status_code == 404:  # no props posted for this game
            return items
        data = resp.json()
        items += data.get("items", [])
        if page >= data.get("pageCount", 1):
            return items
        page += 1


def espn_team_roster(espn_team_id: str) -> dict[int, str]:
    """{espn athlete id: display name} for a team's current roster."""
    data = _call(lambda: requests.get(ESPN_ROSTER_URL.format(team_id=espn_team_id), timeout=30).json())
    return {int(a["id"]): a["displayName"] for a in data.get("athletes", [])}


def espn_athlete_name(espn_athlete_id: int) -> str | None:
    """Look up one athlete, e.g. a player traded since the game (no longer on that team's roster)."""
    resp = _call(lambda: requests.get(f"{ESPN_CORE_URL}/athletes/{espn_athlete_id}", params={"lang": "en", "region": "us"}, timeout=30))
    return resp.json().get("displayName") if resp.ok else None


def _ref_id(ref: str, kind: str) -> int | None:
    m = re.search(rf"/{kind}/(\d+)", ref)
    return int(m.group(1)) if m else None


def _float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _home_spread(details: str, home: str) -> float | None:
    """'DET -9.5' with WAS at home -> +9.5; 'EVEN' -> 0."""
    if details.strip().upper() in ("EVEN", "PK", "PICK"):
        return 0.0
    m = re.match(r"^([A-Z]+)\s+([-+]?\d+(?:\.\d+)?)$", details.strip())
    if not m:
        return None
    favorite = ESPN_TO_NBA_ABBR.get(m.group(1), m.group(1))
    line = float(m.group(2))
    return line if favorite == home else -line


def normalize_name(name: str) -> str:
    """'Luka Dončić Jr.' -> 'luka doncic' so ESPN and NBA names can be matched."""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"[.'\-]", "", name.lower())
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", name)
    return " ".join(name.split())
