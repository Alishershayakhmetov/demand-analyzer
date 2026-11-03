"""Baseline forecasters. All take a history matrix (items x days) and return an (items x HORIZON) forecast."""
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, CrostonSBA, TSB

from .config import HORIZON


def _same_weekday(hist: np.ndarray, weeks: int) -> np.ndarray:
    """(items, HORIZON, weeks) values observed on the same weekday as each forecast day, most recent first."""
    t = hist.shape[1]
    h = np.arange(HORIZON)
    k = (h // 7 + 1)[:, None] + np.arange(weeks)[None, :]
    return hist[:, t + h[:, None] - 7 * k]


def naive(hist: np.ndarray) -> np.ndarray:
    return np.repeat(hist[:, -1:], HORIZON, axis=1).astype(float)


def seasonal_naive(hist: np.ndarray) -> np.ndarray:
    return _same_weekday(hist, 1)[:, :, 0].astype(float)


def moving_average(hist: np.ndarray, window: int = 28) -> np.ndarray:
    return np.repeat(hist[:, -window:].mean(axis=1, keepdims=True), HORIZON, axis=1)


def weekday_mean(hist: np.ndarray, weeks: int = 8) -> np.ndarray:
    return _same_weekday(hist, weeks).mean(axis=2)


def weekday_quantile(hist: np.ndarray, q: float, weeks: int = 8) -> np.ndarray:
    return np.quantile(_same_weekday(hist, weeks), q, axis=2)


def statsforecast_models(hist: np.ndarray, first_day: np.ndarray, item_ids: list[str], max_len: int = 730) -> dict[str, np.ndarray]:
    """ETS, Croston-SBA and TSB per series. Each series starts at its first sale-day and is capped at max_len days."""
    t = hist.shape[1]
    frames = []
    for i, item in enumerate(item_ids):
        start = int(max(first_day[i] - 1, t - max_len))
        y = hist[i, start:]
        ds = pd.date_range("2000-01-01", periods=len(y), freq="D")
        frames.append(pd.DataFrame({"unique_id": item, "ds": ds, "y": y.astype(float)}))
    sf = StatsForecast(models=[AutoETS(season_length=7), CrostonSBA(), TSB(alpha_d=0.2, alpha_p=0.2)], freq="D", n_jobs=1)
    fc = sf.forecast(df=pd.concat(frames, ignore_index=True), h=HORIZON)
    if "unique_id" not in fc.columns:
        fc = fc.reset_index()
    fc["h"] = fc.groupby("unique_id").cumcount()  # series have different end dates, so index by horizon step
    out = {}
    for name in ["AutoETS", "CrostonSBA", "TSB"]:
        wide = fc.pivot(index="unique_id", columns="h", values=name)
        out[name] = wide.loc[item_ids].to_numpy()
    return out
