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
