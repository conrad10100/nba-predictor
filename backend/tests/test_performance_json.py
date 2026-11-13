import json
import math

import numpy as np

from app.main import _json_safe


def test_json_safe_replaces_nan_everywhere():
    # A prediction exactly on the line has no pick; it reached the response as NaN and broke the Performance page.
    payload = {"top_performers": [{"pick": float("nan"), "line": np.float64(20.5), "rank": np.int64(1)}],
               "totals": {"win_pct": math.inf}, "nested": [[np.nan, 1.5]]}
    clean = _json_safe(payload)
    json.dumps(clean, allow_nan=False)  # raises if any NaN/inf survived
    assert clean["top_performers"][0] == {"pick": None, "line": 20.5, "rank": 1}
    assert clean["totals"]["win_pct"] is None
    assert clean["nested"] == [[None, 1.5]]
