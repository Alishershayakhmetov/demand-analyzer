from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from . import metrics as M

EVENT_TYPES = ["Sporting", "Cultural", "National", "Religious"]


@dataclass
class Matrices:
    ids: list[str]
    items: pd.DataFrame
    Y: np.ndarray            # (items, days) units sold
    PR: np.ndarray           # (items, days) price, NaN when the item is not on sale
    first_day: np.ndarray    # (items,) first day an item had a price
    dept: np.ndarray
    cat: np.ndarray
    snap: np.ndarray         # (days,)
    wday: np.ndarray
    month: np.ndarray
    dom: np.ndarray
    event: np.ndarray        # (days,) 0 = no event, 1..4 = event type

    @property
    def on_sale(self) -> np.ndarray:
        return ~np.isnan(self.PR)


def load_matrices() -> Matrices:
    panel = pd.read_parquet(C.PROC / f"panel_{C.STORE}.parquet")
    items = pd.read_csv(C.PROC / "items.csv").sort_values("item_id").reset_index(drop=True)
    ids = items.item_id.tolist()
    piv = lambda col: panel.pivot(index="item_id", columns="d", values=col).loc[ids].to_numpy()  # noqa: E731
    cal = panel.drop_duplicates("d").sort_values("d")
    static = panel.drop_duplicates("item_id").set_index("item_id").loc[ids]
    event = cal.event_type_1.map({t: i + 1 for i, t in enumerate(EVENT_TYPES)}).fillna(0).to_numpy()
    return Matrices(
        ids=ids, items=items, Y=piv("sales"), PR=piv("sell_price"),
        first_day=items.first_d.to_numpy().astype(int),
        dept=static.dept_id.astype("category").cat.codes.to_numpy(),
        cat=static.cat_id.astype("category").cat.codes.to_numpy(),
        snap=cal.snap.to_numpy(), wday=cal.wday.to_numpy(), month=cal.month.to_numpy(),
        dom=cal.date.dt.day.to_numpy(), event=event,
    )


def score(actual: np.ndarray, f: np.ndarray, hist: np.ndarray, first_day: np.ndarray) -> dict:
    per_item = [M.rmsse(actual[i], f[i], hist[i, max(first_day[i] - 1, 0):]) for i in range(len(actual))]
    return dict(wape=M.wape(actual, f), bias=M.bias(actual, f), rmsse=float(np.nanmean(per_item)))
