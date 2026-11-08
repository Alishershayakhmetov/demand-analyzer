"""Shared pieces for experiments that retrain LightGBM on feature subsets and score the result."""
import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config as C
from . import metrics as M
from .evaluate import Matrices
from .features import CATEGORICAL
from .inventory import inventory_at_fill, simulate

COMMON = dict(learning_rate=0.05, n_estimators=400, num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, verbose=-1, n_jobs=-1, objective="tweedie", tweedie_variance_power=1.1)
Z = np.arange(-1.0, 6.01, 0.25)
LEAD, TARGET = 3, 0.95


def fit(train: pd.DataFrame, feats: list[str], seed: int = C.SEED, **overrides) -> lgb.LGBMRegressor:
    """Tweedie LightGBM with the shared defaults; `overrides` replace single parameters (e.g. num_leaves, n_estimators)."""
    return lgb.LGBMRegressor(**{**COMMON, **overrides}, random_state=seed).fit(
        train[feats], train.y, categorical_feature=[c for c in CATEGORICAL if c in feats])


def predict_origins(model: lgb.LGBMRegressor, origins: list[pd.DataFrame], feats: list[str], n_items: int,
                    num_iteration: int | None = None) -> list[np.ndarray]:
    """One (items, HORIZON) forecast per weekly origin; days without a price are set to 0."""
    return [np.clip(model.predict(rows[feats], num_iteration=num_iteration), 0, None).reshape(n_items, C.HORIZON)
            * rows.on_sale.to_numpy().reshape(n_items, C.HORIZON) for rows in origins]


def moving_average_origins(m: Matrices, cutoff: int, origins: list[pd.DataFrame]) -> list[np.ndarray]:
    n = len(m.ids)
    return [np.repeat(m.Y[:, :cutoff + 7 * k][:, -28:].mean(axis=1, keepdims=True), C.HORIZON, axis=1)
            * rows.on_sale.to_numpy().reshape(n, C.HORIZON) for k, rows in enumerate(origins)]


def window_metrics(m: Matrices, cutoff: int, forecast: np.ndarray) -> dict[str, float]:
    actual = m.Y[:, cutoff:cutoff + C.HORIZON].astype(float)
    return dict(daily_wape=M.wape(actual, forecast),
                weekly_wape=M.wape(M.block_sums(actual), M.block_sums(forecast)),
                wape_28d=M.wape(M.block_sums(actual, 28), M.block_sums(forecast, 28)))


def inventory_days(m: Matrices, rolls: dict[int, list[np.ndarray]], folds: list[int], lead: int = LEAD, target: float = TARGET) -> float:
    """Average inventory (days of sales) for a revenue-weighted fill rate of `target`; weekly refresh, order-up-to policy."""
    fill, inv = inventory_frontier(m, rolls, folds, lead)
    return inventory_at_fill(fill, inv, target)


def inventory_frontier(m: Matrices, rolls: dict[int, list[np.ndarray]], folds: list[int], lead: int = LEAD) -> tuple[np.ndarray, np.ndarray]:
    """Fill rate and average inventory (days of sales) for every z in Z, pooled over `folds`."""
    sold, stock, dem = inventory_by_item(m, rolls, folds, lead)
    return sold.sum(axis=1) / dem.sum(), stock.sum(axis=1) / dem.sum()


def inventory_by_item(m: Matrices, rolls: dict[int, list[np.ndarray]], folds: list[int], lead: int = LEAD
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per item, summed over folds: revenue sold and end-of-day stock value for every z (z, items), and demand value (items)."""
    span = lead + 7
    n = len(m.ids)
    sold, stock, dem = np.zeros((len(Z), n)), np.zeros((len(Z), n)), np.zeros(n)
    for cutoff in folds:
        actual = m.Y[:, cutoff:cutoff + C.HORIZON]
        price = np.nanmedian(m.PR[:, cutoff - 91:cutoff], axis=1)
        active = m.Y[:, cutoff - 28:cutoff].sum(axis=1) > 0
        init = np.ceil(m.Y[:, cutoff - 28:cutoff].mean(axis=1) * lead)
        means = np.stack([rolls[cutoff][r][:, :span].sum(axis=1) for r in range(4)])
        sigmas = np.stack([m.Y[:, :cutoff + 7 * r][:, -91:].std(axis=1) * np.sqrt(span) for r in range(4)])
        s, k = simulate(actual, means, sigmas, Z, lead, init)
        w = price * active
        sold += s * w
        stock += k * w
        dem += actual[:, lead:].sum(axis=1) * w
    return sold, stock, dem
