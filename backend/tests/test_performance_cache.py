import app.main as m


def test_cached_reuses_until_data_changes(monkeypatch):
    fp = [("a",)]
    monkeypatch.setattr(m, "_fingerprint", lambda: fp[0])
    calls = []

    def compute():
        calls.append(1)
        return len(calls)

    key = ("test-cache",)
    m._data_cache.pop(key, None)
    assert m._cached(key, compute) == 1
    assert m._cached(key, compute) == 1  # same data: served from cache
    fp[0] = ("b",)  # a job wrote new predictions / lines / actuals
    assert m._cached(key, compute) == 2
    monkeypatch.setattr(m, "DATA_TTL_S", -1)  # backstop expiry
    assert m._cached(key, compute) == 3


def test_backtest_cache_survives_live_writes(monkeypatch):
    """A new live prediction or line must not re-download the backtest rows (Supabase egress)."""
    fp = [(("live-1",), ("bt-1",))]
    monkeypatch.setattr(m, "_fingerprint", lambda: fp[0])
    calls = {"live": 0, "backtest": 0}

    def compute(kind):
        calls[kind] += 1
        return calls[kind]

    for kind in m.KINDS:
        m._data_cache.pop(("test-part", kind), None)
    get = lambda kind, part: m._cached(("test-part", kind), lambda: compute(kind), part=part)
    assert (get("live", m.LIVE), get("backtest", m.BACKTEST)) == (1, 1)
    fp[0] = (("live-2",), ("bt-1",))  # the evening refresh re-predicted tonight's games
    assert (get("live", m.LIVE), get("backtest", m.BACKTEST)) == (2, 1)
    fp[0] = (("live-2",), ("bt-2",))  # a backtest re-run
    assert (get("live", m.LIVE), get("backtest", m.BACKTEST)) == (2, 2)


def test_picks_concatenates_kinds(monkeypatch):
    import pandas as pd

    monkeypatch.setattr(m, "_fingerprint", lambda: (("l",), ("b",)))
    rows = {k: pd.DataFrame([{c: None for c in m.PICK_COLUMNS} | {"date": d, "player": "A", "stat": "pts", "kind": k}])
            for k, d in (("live", "2026-11-19"), ("backtest", "2025-03-01"))}
    monkeypatch.setattr(m, "_compute_picks", lambda kind: rows[kind])
    for kind in m.KINDS:
        m._data_cache.pop(("picks", kind), None)
    df = m._picks()
    assert list(df["kind"]) == ["backtest", "live"]  # sorted by date across both kinds
    assert list(df.columns) == list(m.PICK_COLUMNS)
