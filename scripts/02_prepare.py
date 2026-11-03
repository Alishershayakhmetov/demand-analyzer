"""Build the single-store panel: select ~300 items, join prices and calendar, save to data/processed."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demand import config as C  # noqa: E402

ID_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


def load_sales() -> pd.DataFrame:
    train = pd.read_csv(C.RAW / "sales_train_evaluation.csv")
    test = pd.read_csv(C.RAW / "sales_test_evaluation.csv")
    train = train[train.store_id == C.STORE].reset_index(drop=True)
    test = test[test.store_id == C.STORE].reset_index(drop=True)
    assert (train.item_id.values == test.item_id.values).all()
    day_cols = [c for c in train.columns if c.startswith("d_")] + [c for c in test.columns if c.startswith("d_")]
    wide = pd.concat([train, test.drop(columns=ID_COLS)], axis=1)
    assert len(day_cols) == C.LAST_DAY
    return wide


def load_calendar() -> pd.DataFrame:
    cal = pd.read_csv(C.RAW / "calendar.csv", parse_dates=["date"])
    cal.insert(0, "d", np.arange(1, len(cal) + 1))
    assert cal.date.is_monotonic_increasing and len(cal) == C.LAST_DAY
    cal["snap"] = cal[f"snap_{C.STATE}"]
    return cal.drop(columns=["snap_CA", "snap_TX", "snap_WI", "weekday"])


def load_prices() -> pd.DataFrame:
    p = pd.read_csv(C.RAW / "sell_prices.csv", dtype={"store_id": "category", "item_id": "category"})
    p = p[p.store_id == C.STORE].copy()
    p["store_id"] = p.store_id.astype(str)
    p["item_id"] = p.item_id.astype(str)
    return p


def demand_stats(y: np.ndarray) -> tuple[float, float]:
    """Average demand interval and squared coefficient of variation of non-zero sizes."""
    nz = y[y > 0]
    if len(nz) == 0:
        return np.inf, np.nan
    adi = len(y) / len(nz)
    cv2 = (nz.std() / nz.mean()) ** 2
    return adi, cv2


def demand_class(adi: float, cv2: float) -> str:
    if adi < 1.32:
        return "smooth" if cv2 < 0.49 else "erratic"
    return "intermittent" if cv2 < 0.49 else "lumpy"


def select_items(wide: pd.DataFrame, prices: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    """Pick items using only data up to SELECTION_CUTOFF so the choice cannot see the folds."""
    day_cols = [f"d_{i}" for i in range(1, C.SELECTION_CUTOFF + 1)]
    y = wide[day_cols].to_numpy()
    first_price_wk = prices.groupby("item_id").wm_yr_wk.min()
    wk_to_first_d = cal.groupby("wm_yr_wk").d.min()
    first_d = first_price_wk.map(wk_to_first_d)

    rows = []
    last_year = slice(C.SELECTION_CUTOFF - 365, C.SELECTION_CUTOFF)
    for i, item in enumerate(wide.item_id):
        fd = first_d.get(item, np.nan)
        series = y[i, int(fd) - 1:] if fd == fd else np.array([])
        adi, cv2 = demand_stats(series) if len(series) else (np.inf, np.nan)
        rows.append(dict(item_id=item, first_d=fd, units_last_year=y[i, last_year].sum(), adi=adi, cv2=cv2))
    stats = pd.DataFrame(rows)
    eligible = stats[(stats.first_d <= C.SELECTION_CUTOFF - 365) & (stats.units_last_year > 0)].copy()
    eligible["stratum"] = pd.qcut(eligible.units_last_year.rank(method="first"), 3, labels=["low", "mid", "high"])
    picked = (eligible.groupby("stratum", observed=True)
              .sample(n=C.N_PER_STRATUM, random_state=C.SEED)
              .reset_index(drop=True))
    picked["demand_class"] = [demand_class(a, c) for a, c in zip(picked.adi, picked.cv2)]
    print(f"items in store {len(stats)}, eligible {len(eligible)}, picked {len(picked)}")
    return picked


def main() -> None:
    C.PROC.mkdir(parents=True, exist_ok=True)
    wide, cal, prices = load_sales(), load_calendar(), load_prices()
    items = select_items(wide, prices, cal)
    items.to_csv(C.PROC / "items.csv", index=False)

    sub = wide[wide.item_id.isin(items.item_id)]
    day_cols = [f"d_{i}" for i in range(1, C.LAST_DAY + 1)]
    long = sub.melt(id_vars=ID_COLS, value_vars=day_cols, var_name="d", value_name="sales")
    long["d"] = long.d.str[2:].astype(int)
    long = long.merge(cal, on="d", how="left").merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    long["on_sale"] = long.sell_price.notna().astype("int8")
    long["sales"] = long.sales.astype("int16")
    long = long.sort_values(["item_id", "d"]).reset_index(drop=True)
    long.to_parquet(C.PROC / f"panel_{C.STORE}.parquet", index=False)
    print(long.shape, long.date.min().date(), long.date.max().date())
    print(items.groupby(["stratum", "demand_class"], observed=True).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
