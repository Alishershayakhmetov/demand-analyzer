"""Choose the forecasting method for an uploaded file by a backtest on the file's own history.

Candidates: seasonal naive, 28-day average, weekday average (8 weeks), the M5 LightGBM with the range guard, the scale-normalised LightGBM,
the weekly-sum LightGBM, Chronos-Bolt (raw and with its bias corrected on the file), Chronos + weekday average, and Chronos + M5 LightGBM +
weekday average. The last up to MAX_WINDOWS 28-day windows are forecast from the data before them. Final forecast = mean of the TOP_K best
candidates by weekly WAPE (K = 1 reproduces the earlier rule: best single method, a complex one only if it beats the best simple one by MARGIN).
Files too short for a backtest get the weekday average.
"""
import numpy as np

from . import baselines as B
from . import metrics as M
from . import serving as S
from . import universal as U
from .config import HORIZON

MARGIN = 0.02
MAX_WINDOWS = 4
TOP_K = 2
SIMPLE = ["weekday average", "28-day average", "seasonal naive"]
LEGACY = SIMPLE + ["LightGBM (M5)", "LightGBM (scale-normalised)", "Chronos-Bolt", "Chronos-Bolt + weekday average"]


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


def raw_candidates(shop: S.Shop, origin: int, booster, features, limit, norm_model, weekly_model, use_chronos: bool) -> tuple[dict, np.ndarray]:
    """Forecasts of every candidate (Chronos not yet bias-corrected) from the first `origin` days, and the guard mask of the M5 LightGBM."""
    hist = shop.Y[:, :origin]
    first = np.minimum(shop.first_idx, origin)
    sh = S.Shop(shop.ids, shop.dates[:origin], hist, None, first)
    wd8 = B.weekday_mean(hist)
    out = {"weekday average": wd8, "28-day average": B.moving_average(hist), "seasonal naive": B.seasonal_naive(hist)}
    lg, guard = S.forecast_with_info(sh, booster, features, limit)
    out["LightGBM (M5)"] = lg
    if norm_model is not None:
        out["LightGBM (scale-normalised)"] = U.predict_normalised(norm_model, hist, first)
    if weekly_model is not None:
        out["LightGBM (weekly)"] = U.predict_weekly(weekly_model, hist, first)
    if use_chronos:
        out["Chronos-Bolt"] = U.chronos_forecast(hist)
    return {k: (v if k == "LightGBM (M5)" else _status_fix(v, hist, first)) for k, v in out.items()}, guard


def _combine(fc: dict, wd8: np.ndarray, corr: float | None) -> dict:
    """Add the methods derived from others: bias-corrected Chronos and the averages with Chronos."""
    out = dict(fc)
    if "Chronos-Bolt" in fc:
        ch = fc["Chronos-Bolt"]
        out["Chronos-Bolt + weekday average"] = (ch + wd8) / 2
        out["Chronos-Bolt + LightGBM (M5) + weekday average"] = (ch + fc["LightGBM (M5)"] + wd8) / 3
        if corr is not None:
            out["Chronos-Bolt (bias-corrected)"] = ch * corr
    return out


def _multiplier(actual: list[np.ndarray], fc: list[np.ndarray]) -> float | None:
    if not actual:
        return None
    return float(np.clip(sum(a.sum() for a in actual) / max(sum(f.sum() for f in fc), 1e-9), 0.5, 3.0))


def select(shop: S.Shop, booster, features, limit, norm_model=None, weekly_model=None, use_chronos: bool | None = None,
           max_windows: int = MAX_WINDOWS, top_k: int = TOP_K, names: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Returns (forecast for the next 28 days, guard mask (only if the M5 LightGBM is part of the answer), info dict)."""
    use_chronos = chronos_available() if use_chronos is None else use_chronos
    T = shop.Y.shape[1]
    origins = [T - HORIZON * (k + 1) for k in range(max_windows) if T - HORIZON * (k + 1) >= U.MIN_HISTORY]
    raws = []
    for o in origins:
        fc, _ = raw_candidates(shop, o, booster, features, limit, norm_model, weekly_model, use_chronos)
        raws.append((o, fc))
    final_raw, guard = raw_candidates(shop, T, booster, features, limit, norm_model, weekly_model, use_chronos)
    scores: dict[str, list[float]] = {}
    combos = []
    for i, (o, fc) in enumerate(raws):
        actual = shop.Y[:, o:o + HORIZON]
        others = [(shop.Y[:, o2:o2 + HORIZON], fc2["Chronos-Bolt"]) for j, (o2, fc2) in enumerate(raws) if j != i and "Chronos-Bolt" in fc2]
        corr = _multiplier([a for a, _ in others], [f for _, f in others])  # leave-one-window-out, so the score is honest
        combo = _combine(fc, fc["weekday average"], corr)
        combos.append(combo)
        for name, f in combo.items():
            scores.setdefault(name, []).append(M.wape(M.block_sums(actual), M.block_sums(f)))
    all_corr = _multiplier([shop.Y[:, o:o + HORIZON] for o, _ in raws], [fc["Chronos-Bolt"] for _, fc in raws]) if use_chronos and raws else None
    final = _combine(final_raw, final_raw["weekday average"], all_corr if len(raws) >= 2 else None)
    if names is not None:
        scores = {k: v for k, v in scores.items() if k in names}
        final = {k: v for k, v in final.items() if k in names}
    scores = {k: v for k, v in scores.items() if len(v) == len(raws) and k in final}
    if not scores:
        return final["weekday average"], np.zeros(len(shop.ids), bool), dict(method="weekday average", reason="short_history", backtest={}, windows=0)
    mean = {k: float(np.mean(v)) for k, v in scores.items()}
    ranked = sorted(mean, key=mean.get)
    if top_k == 1:
        best_simple = min(SIMPLE, key=mean.get)
        chosen = [ranked[0] if ranked[0] in SIMPLE or mean[ranked[0]] <= mean[best_simple] * (1 - MARGIN) else best_simple]
    else:
        chosen = ranked[:top_k]
    f = np.mean([final[k] for k in chosen], axis=0)
    use_guard = guard if "LightGBM (M5)" in chosen else np.zeros(len(shop.ids), bool)
    bt = [(o, np.mean([c[k] for k in chosen], axis=0)) for (o, _), c in zip(raws, combos)]  # backtest forecasts of the answer, for service-level calibration
    return f, use_guard, dict(method=" + ".join(chosen), reason="backtest", backtest={k: round(mean[k], 4) for k in ranked}, windows=len(raws), _bt=bt)
