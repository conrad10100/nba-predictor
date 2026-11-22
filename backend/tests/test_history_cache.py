import numpy as np
import pandas as pd

from app.features.build import pack_frames, unpack_frames


def _frames(n=2000):
    rng = np.random.default_rng(0)
    ps = pd.DataFrame({
        "player_id": rng.integers(1, 1_700_000, n), "game_id": [f"00226{i:05d}" for i in range(n)],
        "team_id": rng.integers(1610612737, 1610612767, n), "min": rng.random(n) * 48,
        "pts": rng.integers(0, 50, n), "fg3a": rng.integers(0, 15, n),
    })
    ps.loc[3, "min"] = np.nan
    ts = pd.DataFrame({"team_id": [1, 2], "game_id": ["0022600001", "0022600001"], "is_home": [True, False],
                       "min": [240.0, 265.0], "pts": [101, 99]})
    return {"player_stats": ps, "team_stats": ts}


def test_pack_round_trips_exactly():
    f = _frames()
    back = unpack_frames(pack_frames(f))
    for name, df in f.items():
        pd.testing.assert_frame_equal(back[name], df)
    assert back["player_stats"]["game_id"].iloc[0] == "0022600000"  # leading zeros kept


def test_pack_is_small():
    f = _frames(84_000)
    blob = pack_frames(f)
    assert len(blob) < 2_000_000
