"""Scale-free forecasting for sales histories of any size, and a Chronos-Bolt wrapper.

Scale-normalised LightGBM: every item is divided by its own level (mean of the last 91 days + EPS), the model predicts the ratio
of future sales to that level, and the forecast is multiplied back. Demand of 0.4 and of 400 units a day then look alike to the model,
so it does not fall apart on volumes it never saw. Origins are Sundays (the forecast starts on Monday), as in the app.
"""
import numpy as np
import pandas as pd

from .baselines import _same_weekday
from .config import HORIZON
from .serving import Shop

EPS = 0.05
MIN_HISTORY = 91
FEATURES = ["r7", "r14", "r28", "r56", "rstd28", "nz28", "nz91", "dsl", "age", "logscale", "h", "wd_r8"]
PARAMS = dict(objective="tweedie", tweedie_variance_power=1.2, learning_rate=0.05, n_estimators=300, num_leaves=63, min_child_samples=100,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=-1, random_state=42)


def level(hist: np.ndarray) -> np.ndarray:
    return hist[:, -91:].mean(axis=1) + EPS


def rows_at(Y: np.ndarray, first_idx: np.ndarray, origin: int, with_target: bool = True) -> pd.DataFrame:
    """Feature rows (items x HORIZON) for the forecast made after `origin` days of history."""
    hist = Y[:, :origin]
    n, H = hist.shape[0], HORIZON
    s = level(hist)
    sold = hist[:, -91:] > 0
    dsl = np.where(sold.any(axis=1), np.argmax(sold[:, ::-1], axis=1), 91)
    per_item = {"r7": hist[:, -7:].mean(1) / s, "r14": hist[:, -14:].mean(1) / s, "r28": hist[:, -28:].mean(1) / s, "r56": hist[:, -56:].mean(1) / s,
                "rstd28": hist[:, -28:].std(1) / s, "nz28": (hist[:, -28:] > 0).mean(1), "nz91": sold.mean(1), "dsl": dsl,
                "age": np.minimum(origin - first_idx, 1000), "logscale": np.clip(np.log10(s), -1.5, 3.5)}
    cols = {k: np.repeat(v[:, None], H, axis=1) for k, v in per_item.items()}
    cols["h"] = np.tile(np.arange(1, H + 1), (n, 1))
    cols["wd_r8"] = _same_weekday(hist, 8).mean(axis=2) / s[:, None]
    if with_target:
        cols["y"] = Y[:, origin:origin + H] / s[:, None]
    df = pd.DataFrame({k: v.reshape(-1) for k, v in cols.items()})
    df["scale"] = np.repeat(s, H)
    return df


def sunday_origins(dates: pd.DatetimeIndex, last_origin: int, n_max: int, stride: int = 7) -> list[int]:
    """Origins o (days of history) with dates[o-1] a Sunday, newest first, at most n_max, each with 28 days of targets before `last_origin`+28."""
    out = []
    o = last_origin
    while o >= MIN_HISTORY and len(out) < n_max:
        if dates[o - 1].weekday() == 6:
            out.append(o)
            o -= stride
        else:
            o -= 1
    return out


def training_frame(shop: Shop, last_origin: int | None = None, n_origins: int = 60, max_items: int = 300, max_rows: int = 500_000, seed: int = 0) -> pd.DataFrame:
    T = shop.Y.shape[1]
    last = min(T - HORIZON, last_origin if last_origin is not None else T - HORIZON)
    rng = np.random.default_rng(seed)
    items = np.arange(shop.Y.shape[0])
    if len(items) > max_items:
        items = np.sort(rng.choice(items, max_items, replace=False))
    parts = []
    for o in sunday_origins(shop.dates, last, n_origins):
        keep = items[(o - shop.first_idx[items]) >= 28]
        if len(keep):
            parts.append(rows_at(shop.Y[keep], shop.first_idx[keep], o))
    df = pd.concat(parts, ignore_index=True)
    return df.sample(max_rows, random_state=seed) if len(df) > max_rows else df


def predict_normalised(model, Y: np.ndarray, first_idx: np.ndarray) -> np.ndarray:
    rows = rows_at(Y, first_idx, Y.shape[1], with_target=False)
    p = np.clip(model.predict(rows[FEATURES]), 0, None) * rows["scale"].to_numpy()
    return p.reshape(Y.shape[0], HORIZON)


_PIPE = {}


def chronos_forecast(Y: np.ndarray, name: str = "amazon/chronos-bolt-small", context: int = 512, batch: int = 128) -> np.ndarray:
    """Zero-shot Chronos-Bolt forecast (mean of the predicted quantiles) for every row of Y; scale-invariant by construction."""
    import torch
    from chronos import BaseChronosPipeline
    if name not in _PIPE:
        _PIPE[name] = BaseChronosPipeline.from_pretrained(name, device_map="cpu", torch_dtype=torch.float32)
    out = []
    for i in range(0, Y.shape[0], batch):
        ctx = [torch.tensor(y[-context:], dtype=torch.float32) for y in Y[i:i + batch]]
        _, mean = _PIPE[name].predict_quantiles(ctx, prediction_length=HORIZON, quantile_levels=[0.1, 0.3, 0.5, 0.7, 0.9])
        out.append(np.clip(mean.numpy(), 0, None))
    return np.concatenate(out)


# ---------- weekly-sum model: predicts the sum of the next 4 weeks directly, then spreads it over the days ----------
WEEK_FEATURES = ["r7", "r14", "r28", "r56", "rstd28", "nz28", "nz91", "dsl", "age", "logscale", "w"]


def week_rows_at(Y: np.ndarray, first_idx: np.ndarray, origin: int, with_target: bool = True) -> pd.DataFrame:
    """Feature rows (items x 4 weeks); the target is the weekly sum divided by 7 x level."""
    hist = Y[:, :origin]
    n = hist.shape[0]
    s = level(hist)
    sold = hist[:, -91:] > 0
    dsl = np.where(sold.any(axis=1), np.argmax(sold[:, ::-1], axis=1), 91)
    per_item = {"r7": hist[:, -7:].mean(1) / s, "r14": hist[:, -14:].mean(1) / s, "r28": hist[:, -28:].mean(1) / s, "r56": hist[:, -56:].mean(1) / s,
                "rstd28": hist[:, -28:].std(1) / s, "nz28": (hist[:, -28:] > 0).mean(1), "nz91": sold.mean(1), "dsl": dsl,
                "age": np.minimum(origin - first_idx, 1000), "logscale": np.clip(np.log10(s), -1.5, 3.5)}
    cols = {k: np.repeat(v[:, None], 4, axis=1) for k, v in per_item.items()}
    cols["w"] = np.tile(np.arange(1, 5), (n, 1))
    if with_target:
        cols["y"] = Y[:, origin:origin + 28].reshape(n, 4, 7).sum(axis=2) / (7 * s[:, None])
    df = pd.DataFrame({k: v.reshape(-1) for k, v in cols.items()})
    df["scale"] = np.repeat(s, 4)
    return df


def week_training_frame(shop: Shop, last_origin: int | None = None, n_origins: int = 60, max_items: int = 300, seed: int = 0) -> pd.DataFrame:
    T = shop.Y.shape[1]
    last = min(T - HORIZON, last_origin if last_origin is not None else T - HORIZON)
    rng = np.random.default_rng(seed)
    items = np.arange(shop.Y.shape[0])
    if len(items) > max_items:
        items = np.sort(rng.choice(items, max_items, replace=False))
    parts = []
    for o in sunday_origins(shop.dates, last, n_origins):
        keep = items[(o - shop.first_idx[items]) >= 28]
        if len(keep):
            parts.append(week_rows_at(shop.Y[keep], shop.first_idx[keep], o))
    return pd.concat(parts, ignore_index=True)


def predict_weekly(model, Y: np.ndarray, first_idx: np.ndarray) -> np.ndarray:
    """(items, 28): weekly sums from the model, spread over the days by the weekday profile of the last 8 weeks."""
    rows = week_rows_at(Y, first_idx, Y.shape[1], with_target=False)
    weekly = (np.clip(model.predict(rows[WEEK_FEATURES]), 0, None) * rows["scale"].to_numpy() * 7).reshape(Y.shape[0], 4)
    prof = _same_weekday(Y, 8).mean(axis=2).reshape(Y.shape[0], 4, 7)
    tot = prof.sum(axis=2, keepdims=True)
    share = np.where(tot > 0, prof / np.where(tot > 0, tot, 1), 1 / 7)
    return (weekly[:, :, None] * share).reshape(Y.shape[0], 28)


# ---------- exposure-offset model: counts as targets, log(level) as offset, no ratio targets (works for tiny volumes) ----------
OFFSET_FEATURES = [f for f in FEATURES if f != "logscale"]
OFFSET_PARAMS = {**PARAMS, "objective": "tweedie", "tweedie_variance_power": 1.2}


def fit_offset(frame: pd.DataFrame):
    import lightgbm as lgb
    return lgb.LGBMRegressor(**OFFSET_PARAMS).fit(frame[OFFSET_FEATURES], frame.y * frame.scale, init_score=np.log(frame.scale.to_numpy()))


def predict_offset(model, Y: np.ndarray, first_idx: np.ndarray) -> np.ndarray:
    rows = rows_at(Y, first_idx, Y.shape[1], with_target=False)
    raw = model.predict(rows[OFFSET_FEATURES], raw_score=True) + np.log(rows["scale"].to_numpy())
    return np.exp(raw).reshape(Y.shape[0], HORIZON)
