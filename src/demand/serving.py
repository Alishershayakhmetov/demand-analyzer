"""Turn a shop's sales history into a 28-day forecast and a purchase plan, using the model trained in scripts/14_train_final.py.

The model was trained on forecast origins that are Sundays (the forecast starts on Monday), so the history is cut at
the last Sunday it contains. Features are built by the same `build_rows` code as in training.
"""
import json
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config as C
from .baselines import weekday_mean
from .evaluate import Matrices
from .features import build_rows

MODEL_DIR = C.ROOT / "models"
MIN_DAYS = 91
NEW_ITEM_DAYS = 28


@dataclass
class Shop:
    ids: list[str]
    dates: pd.DatetimeIndex
    Y: np.ndarray                 # (items, days) units sold
    price: np.ndarray | None      # (items,) median recent price, None without a price column
    first_idx: np.ndarray         # (items,) index of the first non-zero sale (= days if never sold)
    notes: list[str] = field(default_factory=list)


def to_shop(df: pd.DataFrame, date_col: str, item_col: str, sales_col: str, price_col: str | None = None) -> Shop:
    """Clean a long table (one row per item and day) into a Shop; everything odd is counted and reported in `notes`."""
    notes = []
    d = pd.DataFrame({"date": pd.to_datetime(df[date_col], errors="coerce"), "item": df[item_col].astype(str),
                      "sales": pd.to_numeric(df[sales_col], errors="coerce")})
    if price_col:
        d["price"] = pd.to_numeric(df[price_col], errors="coerce")
    bad = d.date.isna() | d.sales.isna()
    if bad.any():
        notes.append(f"{int(bad.sum())} rows with an unreadable date or sales value were dropped")
    d = d[~bad]
    if (d.sales < 0).any():
        notes.append(f"{int((d.sales < 0).sum())} negative sales values (returns) were set to 0")
        d["sales"] = d.sales.clip(lower=0)
    if d.duplicated(["date", "item"]).any():
        notes.append(f"{int(d.duplicated(['date', 'item']).sum())} duplicate item-day rows were summed")
    agg = {"sales": "sum", **({"price": "mean"} if price_col else {})}
    d["date"] = d.date.dt.normalize()
    d = d.groupby(["item", "date"]).agg(agg).reset_index()
    last = d.date.max()
    cut = (last.weekday() + 1) % 7  # days after the last Sunday
    if cut:
        last = last - pd.Timedelta(days=int(cut))
        d = d[d.date <= last]
        notes.append(f"history cut at the last Sunday ({last.date()}); the forecast starts the next Monday")
    dates = pd.date_range(d.date.min(), last, freq="D")
    if len(dates) < MIN_DAYS:
        raise ValueError(f"need at least {MIN_DAYS} days of history, got {len(dates)}")
    sales = d.pivot(index="item", columns="date", values="sales").reindex(columns=dates).fillna(0.0)
    price = None
    if price_col:
        pr = d.pivot(index="item", columns="date", values="price").reindex(index=sales.index, columns=dates)
        price = pr.iloc[:, -MIN_DAYS:].median(axis=1).to_numpy()
    Y = sales.to_numpy()
    first = np.where((Y > 0).any(axis=1), (Y > 0).argmax(axis=1), Y.shape[1])
    return Shop(sales.index.tolist(), dates, Y, price, first, notes)


def load_model() -> tuple[lgb.Booster, dict]:
    meta = json.loads((MODEL_DIR / "meta.json").read_text(encoding="utf-8"))
    return lgb.Booster(model_file=str(MODEL_DIR / "lean_model.txt")), meta


def item_status(shop: Shop) -> np.ndarray:
    T = shop.Y.shape[1]
    status = np.full(len(shop.ids), "ok", dtype=object)
    status[(T - shop.first_idx) < NEW_ITEM_DAYS] = "new"
    status[shop.Y[:, -NEW_ITEM_DAYS:].sum(axis=1) == 0] = "dormant"
    return status


def build_features(shop: Shop) -> pd.DataFrame:
    """Feature rows (items x 28 horizon steps) from the shop history; same code path as training."""
    n, T = shop.Y.shape
    horizon = C.HORIZON
    m = Matrices(ids=shop.ids, items=pd.DataFrame(), Y=shop.Y, PR=np.ones((n, T + horizon)), first_day=shop.first_idx + 1,
                 dept=np.zeros(n, int), cat=np.zeros(n, int), snap=np.zeros(T + horizon, int), wday=np.zeros(T + horizon, int),
                 month=np.zeros(T + horizon, int), dom=np.zeros(T + horizon, int), event=np.zeros(T + horizon, int))
    return build_rows(m, T, with_target=False)


def forecast(shop: Shop, booster: lgb.Booster, features: list[str]) -> np.ndarray:
    """(items, 28) daily forecast for the 28 days after the last Sunday; new items get their average so far."""
    return forecast_with_info(shop, booster, features)[0]


def forecast_with_info(shop: Shop, booster: lgb.Booster, features: list[str], level_limit: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Like `forecast`, plus a mask of items that were forecast by the weekday average instead of the model.

    The model was trained on items selling at most `level_limit` units per day (28-day mean). Trees cannot extrapolate beyond the
    levels they have seen, so an item above that limit would be forecast far too low; it gets the average of the same weekday over
    the last 8 weeks instead.
    """
    n, T = shop.Y.shape
    rows = build_features(shop)
    f = np.clip(booster.predict(rows[features]), 0, None).reshape(n, C.HORIZON)
    status = item_status(shop)
    for i in np.where(status == "new")[0]:
        history = shop.Y[i, shop.first_idx[i]:] if shop.first_idx[i] < T else np.zeros(1)
        f[i] = history.mean()
    f[status == "dormant"] = 0.0
    fallback = np.zeros(n, dtype=bool)
    if level_limit is not None:
        fallback = (shop.Y[:, -NEW_ITEM_DAYS:].mean(axis=1) > level_limit) & (status == "ok")
        if fallback.any():
            f[fallback] = np.clip(weekday_mean(shop.Y)[fallback], 0, None)
    return f, fallback


def z_for_service(calibration: dict, lead: int, target: float) -> float:
    """Safety factor z that reached `target` fill rate on the M5 backtests for this lead time."""
    c = calibration[str(lead)]
    fill = np.maximum.accumulate(np.array(c["fill"]))
    return float(np.interp(target, fill, np.array(c["z"])))


def purchase_plan(shop: Shop, f: np.ndarray, lead: int, z: float, on_hand: np.ndarray | None = None,
                  on_order: np.ndarray | None = None, status: np.ndarray | None = None) -> pd.DataFrame:
    """Order-up-to plan for one review: cover demand over lead time + 7 days plus z standard deviations.

    `status` defaults to `item_status(shop)`; pass your own to carry extra labels such as "range"."""
    span = lead + 7
    n = len(shop.ids)
    on_hand = np.zeros(n) if on_hand is None else on_hand
    on_order = np.zeros(n) if on_order is None else on_order
    mean = f[:, :span].sum(axis=1)
    sigma = shop.Y[:, -91:].std(axis=1) * np.sqrt(span)
    target = mean + z * sigma
    qty = np.ceil(np.maximum(target - on_hand - on_order, 0))
    status = item_status(shop) if status is None else status
    qty[status == "dormant"] = 0
    out = pd.DataFrame({"item_id": shop.ids, "status": status, "forecast_next_7d": f[:, :7].sum(axis=1).round(1),
                        f"forecast_{span}d": mean.round(1), "safety_stock": (z * sigma).round(1),
                        "target_level": target.round(1), "on_hand": on_hand, "on_order": on_order, "order_qty": qty.astype(int)})
    if shop.price is not None:
        out["order_value"] = (qty * np.nan_to_num(shop.price)).round(2)
    return out
