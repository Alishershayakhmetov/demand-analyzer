"""Feature construction for the global direct forecaster.

One row = (item, forecast origin, horizon step h). Demand features use only days <= origin;
price, SNAP and calendar columns describe the target day and are known in advance.
"""
import warnings

import numpy as np
import pandas as pd

from .baselines import _same_weekday
from .config import HORIZON
from .evaluate import Matrices

MIN_ORIGIN = 400
ORIGIN_STRIDE = 7
N_ORIGINS = 150


def build_rows(m: Matrices, origin: int, with_target: bool = True) -> pd.DataFrame:
    """Rows for all items at one origin; `origin` days of history are visible (d_1 .. d_origin)."""
    hist = m.Y[:, :origin]
    n, H = hist.shape[0], HORIZON
    tgt = np.arange(origin, origin + H)  # 0-based indices of the forecast days
    last91 = hist[:, -91:]
    sold = last91 > 0
    dsl = np.where(sold.any(axis=1), np.argmax(sold[:, ::-1], axis=1), 91)
    weekday = _same_weekday(hist, 8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        price_med = np.nanmedian(m.PR[:, origin - 91:origin], axis=1)
    price_origin = m.PR[:, origin - 1]
    price_t = m.PR[:, tgt]

    per_item = {f"mean{w}": hist[:, -w:].mean(axis=1) for w in (7, 14, 28, 56, 91)}
    per_item.update(std28=hist[:, -28:].std(axis=1), nz28=(hist[:, -28:] > 0).mean(axis=1),
                    nz91=sold.mean(axis=1), days_since_sale=dsl, age=np.minimum(origin - m.first_day, 1000),
                    dept=m.dept, cat=m.cat)
    cols = {k: np.repeat(v[:, None], H, axis=1) for k, v in per_item.items()}
    cols.update(
        wd_mean8=weekday.mean(axis=2), wd_mean4=weekday[:, :, :4].mean(axis=2), wd_last=weekday[:, :, 0],
        h=np.tile(np.arange(1, H + 1), (n, 1)),
        price=price_t, price_vs_median=price_t / price_med[:, None], price_vs_origin=price_t / price_origin[:, None],
        on_sale=(~np.isnan(price_t)).astype(int),
        snap=np.tile(m.snap[tgt], (n, 1)), wday=np.tile(m.wday[tgt], (n, 1)), month=np.tile(m.month[tgt], (n, 1)),
        dom=np.tile(m.dom[tgt], (n, 1)), event=np.tile(m.event[tgt], (n, 1)),
        item=np.repeat(np.arange(n)[:, None], H, axis=1),
    )
    if with_target:
        cols["y"] = m.Y[:, tgt]
    df = pd.DataFrame({k: v.reshape(-1) for k, v in cols.items()})
    df["origin"] = origin
    return df


FEATURES = ["mean7", "mean14", "mean28", "mean56", "mean91", "std28", "nz28", "nz91", "days_since_sale", "age",
            "wd_mean8", "wd_mean4", "wd_last", "h", "price", "price_vs_median", "price_vs_origin",
            "snap", "wday", "month", "dom", "event", "dept", "cat"]
CATEGORICAL = ["wday", "month", "event", "dept", "cat"]


def training_rows(m: Matrices, cutoff: int) -> pd.DataFrame:
    """Rows whose whole forecast window ends on or before `cutoff`; items need >= 28 days of history."""
    last_origin = cutoff - HORIZON
    origins = [o for o in range(last_origin, last_origin - ORIGIN_STRIDE * N_ORIGINS, -ORIGIN_STRIDE) if o >= MIN_ORIGIN]
    df = pd.concat([build_rows(m, o) for o in origins], ignore_index=True)
    return df[(df.on_sale == 1) & (df.age >= 28)].reset_index(drop=True)
