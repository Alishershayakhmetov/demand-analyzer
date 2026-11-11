"""A forecasting model trained on the uploaded file itself that uses events known in advance: price, promotion / discount and holiday flags.

The file must contain these columns for the 28 days to forecast too (rows with a date and covariates but no sales). Because the model is
trained on the file's own history, no cross-domain transfer is needed. Counts are the target and log(level) the offset, as in `universal`.
"""
import numpy as np
import pandas as pd

from . import universal as U
from .config import HORIZON
from .serving import Shop

BASE = [f for f in U.OFFSET_FEATURES if f != "h"]
EXTRA = ["h", "wday", "price_rel", "price_chg", "promo", "promo_recent", "holiday", "holiday_prev", "holiday_next"]
FEATS = BASE + EXTRA


def available(shop: Shop) -> bool:
    c = shop.cov
    return bool(c) and c["future_known"] and any(k in c for k in ("price", "promo", "holiday"))


def cov_rows(shop: Shop, first_idx: np.ndarray, o: int, with_target: bool, items: np.ndarray | None = None) -> pd.DataFrame:
    """Rows (items x 28) for the forecast made after `o` days of history."""
    items = np.arange(shop.Y.shape[0]) if items is None else items
    Y = shop.Y[items]
    df = U.rows_at(Y, first_idx[items], o, with_target=with_target)
    n, H = len(items), HORIZON
    tgt = o + np.arange(H)
    c = shop.cov
    tile = lambda a: np.broadcast_to(a, (n, H)).reshape(-1)  # noqa: E731
    wd = np.array([c["dates"][min(t, len(c["dates"]) - 1)].weekday() for t in tgt])
    df["wday"] = tile(wd)
    if "price" in c:
        P = c["price"][items]
        ref = np.nanmedian(P[:, max(o - 91, 0):o], axis=1)
        ref = np.where(np.isfinite(ref) & (ref > 0), ref, 1.0)
        pt = np.nan_to_num(P[:, tgt], nan=1.0)
        df["price_rel"] = (pt / ref[:, None]).reshape(-1)
        prev = np.nan_to_num(P[:, np.maximum(tgt - 7, 0)], nan=1.0)
        df["price_chg"] = (pt / np.where(prev > 0, prev, 1.0)).reshape(-1)
    else:
        df["price_rel"] = 1.0
        df["price_chg"] = 1.0
    if "promo" in c:
        PR = c["promo"][items]
        df["promo"] = PR[:, tgt].reshape(-1)
        df["promo_recent"] = np.repeat(PR[:, max(o - 28, 0):o].mean(axis=1), H)
    else:
        df["promo"] = 0.0
        df["promo_recent"] = 0.0
    if "holiday" in c:
        hl = c["holiday"]
        df["holiday"] = tile(hl[tgt])
        df["holiday_prev"] = tile(hl[np.maximum(tgt - 1, 0)])
        df["holiday_next"] = tile(hl[np.minimum(tgt + 1, len(hl) - 1)])
    else:
        df["holiday"] = df["holiday_prev"] = df["holiday_next"] = 0.0
    return df


def fit(shop: Shop, last_origin: int, n_origins: int = 40, max_items: int = 300, max_rows: int = 400_000, seed: int = 0):
    """Train on Sunday origins whose targets end before `last_origin` + 28; returns None if the file is too short."""
    import lightgbm as lgb
    rng = np.random.default_rng(seed)
    items = np.arange(shop.Y.shape[0])
    if len(items) > max_items:
        items = np.sort(rng.choice(items, max_items, replace=False))
    parts = []
    for o in U.sunday_origins(shop.dates, last_origin, n_origins):
        keep = items[(o - shop.first_idx[items]) >= 28]
        if len(keep):
            parts.append(cov_rows(shop, shop.first_idx, o, True, keep))
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed)
    params = {**U.OFFSET_PARAMS, "n_estimators": 200, "min_child_samples": 50}
    return lgb.LGBMRegressor(**params).fit(df[FEATS], df.y * df.scale, init_score=np.log(df.scale.to_numpy()))


def predict(model, shop: Shop, o: int) -> np.ndarray:
    """(items, 28) forecast made after `o` days of history."""
    first = np.minimum(shop.first_idx, o)
    rows = cov_rows(shop, first, o, False)
    raw = model.predict(rows[FEATS], raw_score=True) + np.log(rows["scale"].to_numpy())
    return np.exp(raw).reshape(shop.Y.shape[0], HORIZON)
