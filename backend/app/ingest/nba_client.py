"""Thin wrapper over nba_api + ESPN with rate limiting and retries."""

import time
import unicodedata
import re

import pandas as pd
import requests
from nba_api.stats.endpoints import leaguegamelog, playerindex, scheduleleaguev2

REQUEST_GAP_S = 1.0
_last_call = 0.0

ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"


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


def normalize_name(name: str) -> str:
    """'Luka Dončić Jr.' -> 'luka doncic' so ESPN and NBA names can be matched."""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"[.'\-]", "", name.lower())
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", name)
    return " ".join(name.split())
