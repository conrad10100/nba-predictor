from nba_api.stats.static import teams as static_teams

from app.db import upsert
from app.models import Team

DIVISIONS = {
    "Atlantic": ("East", ["BOS", "BKN", "NYK", "PHI", "TOR"]),
    "Central": ("East", ["CHI", "CLE", "DET", "IND", "MIL"]),
    "Southeast": ("East", ["ATL", "CHA", "MIA", "ORL", "WAS"]),
    "Northwest": ("West", ["DEN", "MIN", "OKC", "POR", "UTA"]),
    "Pacific": ("West", ["GSW", "LAC", "LAL", "PHX", "SAC"]),
    "Southwest": ("West", ["DAL", "HOU", "MEM", "NOP", "SAS"]),
}
ABBR_TO_DIV = {abbr: (conf, div) for div, (conf, abbrs) in DIVISIONS.items() for abbr in abbrs}


def seed_teams() -> None:
    rows = []
    for t in static_teams.get_teams():
        conf, div = ABBR_TO_DIV[t["abbreviation"]]
        rows.append({"id": t["id"], "abbr": t["abbreviation"], "name": t["full_name"], "conference": conf, "division": div})
    upsert(Team.__table__, rows)
