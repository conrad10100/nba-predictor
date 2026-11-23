import io

import joblib
import numpy as np
import pandas as pd
import pytest

from app.ml import xgb
from app.ml.compare import hit_rate
from app.ml.registry import dump
from app.ml.train import _model

PARAMS = {"model": "xgboost", "half_life_days": 365, "learning_rate": 0.08, "max_leaf_nodes": 31, "min_samples_leaf": 40,
          "l2_regularization": 1.0, "minutes_weight": 0, "defender_features": 0}


@pytest.mark.skipif(not xgb.AVAILABLE, reason="xgboost not importable")
def test_xgboost_bundle_round_trips_like_the_live_model():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=2000), "b": rng.normal(size=2000)})
    X.loc[::7, "b"] = np.nan  # missing values are allowed, like real features
    y = 3 * X["a"] + rng.normal(size=2000)
    probe = _model(PARAMS).fit(X, y, sample_weight=np.ones(len(y)))
    assert 0 < probe.n_iter_ <= 800
    final = _model(PARAMS, int(probe.n_iter_ * 1.1), early_stop=False).fit(X, y)
    bundle = joblib.load(io.BytesIO(dump({"features": ["a", "b"], "params": PARAMS, "interval": {}, "point": final})))
    np.testing.assert_allclose(bundle["point"].predict(X), final.predict(X))
    assert np.corrcoef(bundle["point"].predict(X), y)[0, 1] > 0.9


def test_hit_rate_skips_pushes_and_missing_lines():
    pred = np.array([10.0, 10.0, 5.0, 5.0, 7.0])
    actual = np.array([12.0, 8.0, 4.0, 6.0, 9.0])
    line = np.array([9.5, 9.5, 6.0, 6.0, np.nan])  # 4th is a push, 5th has no line
    rate, n = hit_rate(pred, actual, line)
    assert n == 3 and rate == pytest.approx(2 / 3)
