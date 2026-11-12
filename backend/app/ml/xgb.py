"""XGBoost behind the same fit/predict/n_iter_ interface as HistGradientBoostingRegressor.

XGBoost needs OpenMP (libomp) at import time. Linux wheels bundle it, so it's available on Railway; on a Mac
without Homebrew's libomp it isn't, and self-tuning simply skips the XGBoost challenger.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split

try:
    import xgboost

    AVAILABLE = True
except Exception:  # ImportError, or XGBoostError when libomp is missing
    xgboost = None
    AVAILABLE = False


class XGBModel(RegressorMixin, BaseEstimator):
    def __init__(self, params: dict, max_iter: int, early_stop: bool = True, quantile: float | None = None):
        self.params = params
        self.max_iter = max_iter
        self.early_stop = early_stop
        self.quantile = quantile

    def _regressor(self) -> "xgboost.XGBRegressor":
        p = self.params
        objective = {"objective": "reg:quantileerror", "quantile_alpha": self.quantile} if self.quantile else {"objective": "reg:squarederror"}
        return xgboost.XGBRegressor(
            n_estimators=self.max_iter, learning_rate=p["learning_rate"], max_leaves=p["max_leaf_nodes"], max_depth=0,
            grow_policy="lossguide", min_child_weight=p["min_samples_leaf"] / 4, reg_lambda=p["l2_regularization"],
            tree_method="hist", early_stopping_rounds=25 if self.early_stop else None, random_state=0, n_jobs=-1, **objective,
        )

    def fit(self, X, y, sample_weight=None):
        self.model_ = self._regressor()
        w = np.ones(len(y)) if sample_weight is None else sample_weight
        if self.early_stop:
            X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(X, y, w, test_size=0.1, random_state=0)
            self.model_.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], sample_weight_eval_set=[w_val], verbose=False)
            self.n_iter_ = self.model_.best_iteration + 1
        else:
            self.model_.fit(X, y, sample_weight=w, verbose=False)
            self.n_iter_ = self.max_iter
        return self

    def predict(self, X):
        return self.model_.predict(X)
