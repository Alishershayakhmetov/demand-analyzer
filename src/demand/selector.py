"""Choose the forecasting method for an uploaded file by a backtest on the file's own history.

Candidates: seasonal naive, 28-day average, weekday average (8 weeks), the M5 LightGBM with the range guard, the scale-normalised
LightGBM, and Chronos-Bolt (if installed) alone and averaged with the weekday average. The last one or two 28-day windows of the
history are forecast from the data before them; the method with the lowest weekly WAPE wins, but a complex method must beat the best
simple one by at least MARGIN, otherwise the simple one is used. Files too short for a backtest get the weekday average.
"""
import numpy as np
import pandas as pd

from . import baselines as B
from . import metrics as M
from . import serving as S
from . import universal as U
from .config import HORIZON

MARGIN = 0.02
SIMPLE = ["weekday average", "28-day average", "seasonal naive"]


def chronos_available() -> bool:
    try:
        import chronos  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _status_fix(f: np.ndarray, Y: np.ndarray, first_idx: np.ndarray) -> np.ndarray:
    """Same treatment of new and dormant items for every method."""
    f = np.clip(f, 0, None).copy()
    T = Y.shape[1]
    dormant = Y[:, -28:].sum(axis=1) == 0
    new = (T - first_idx) < 28
    for i in np.where(new & ~dormant)[0]:
        f[i] = Y[i, first_idx[i]:].mean() if first_idx[i] < T else 0.0
    f[dormant] = 0.0
    return f


def candidate_forecasts(shop: S.Shop, origin: int, booster, features, limit, norm_model, use_chronos: bool) -> tuple[dict, np.ndarray]:
    """Forecasts of every candidate from the first `origin` days; also the mask of items the guard moved to the weekday average."""
    hist = shop.Y[:, :origin]
    first = np.minimum(shop.first_idx, origin)
    sh = S.Shop(shop.ids, shop.dates[:origin], hist, None, first)
    wd8 = B.weekday_mean(hist)
    out = {"weekday average": wd8, "28-day average": B.moving_average(hist), "seasonal naive": B.seasonal_naive(hist)}
    lg, guard = S.forecast_with_info(sh, booster, features, limit)
    out["LightGBM (M5)"] = lg
    if norm_model is not None:
        out["LightGBM (scale-normalised)"] = U.predict_normalised(norm_model, hist, first)
    if use_chronos:
        ch = U.chronos_forecast(hist)
        out["Chronos-Bolt"] = ch
        out["Chronos-Bolt + weekday average"] = (ch + wd8) / 2
    return {k: _status_fix(v, hist, first) if k not in ("LightGBM (M5)",) else v for k, v in out.items()}, guard


def select(shop: S.Shop, booster, features, limit, norm_model=None, use_chronos: bool | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Returns (forecast for the next 28 days, guard mask (only for the M5 LightGBM), info dict)."""
    use_chronos = chronos_available() if use_chronos is None else use_chronos
    T = shop.Y.shape[1]
    origins = [T - HORIZON * (k + 1) for k in range(2) if T - HORIZON * (k + 1) >= U.MIN_HISTORY]
    scores: dict[str, list[float]] = {}
    for o in origins:
        fc, _ = candidate_forecasts(shop, o, booster, features, limit, norm_model, use_chronos)
        actual = shop.Y[:, o:o + HORIZON]
        for name, f in fc.items():
            scores.setdefault(name, []).append(M.wape(M.block_sums(actual), M.block_sums(f)))
    final, guard = candidate_forecasts(shop, T, booster, features, limit, norm_model, use_chronos)
    if not scores:
        return final["weekday average"], np.zeros(len(shop.ids), bool), dict(method="weekday average", reason="short_history", backtest={}, windows=0)
    mean = {k: float(np.mean(v)) for k, v in scores.items()}
    best_simple = min(SIMPLE, key=mean.get)
    best = min(mean, key=mean.get)
    method = best if best in SIMPLE or mean[best] <= mean[best_simple] * (1 - MARGIN) else best_simple
    f = final[method]
    return f, (guard if method == "LightGBM (M5)" else np.zeros(len(shop.ids), bool)), dict(
        method=method, reason="backtest", backtest={k: round(v, 4) for k, v in sorted(mean.items(), key=lambda kv: kv[1])}, windows=len(origins))
