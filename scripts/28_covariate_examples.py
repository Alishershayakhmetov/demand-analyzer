"""Build example files with price / promotion / holiday columns from open data and with the 28 future days as covariate-only rows.
Run isolated: python -I -X utf8 scripts/28_covariate_examples.py   (needs scripts/18 downloads and scripts/02 for M5)
Writes data/external/covariates/<name>_full.csv (every day with sales, used by scripts/29) and examples/<name>_with_future.csv
(history up to the last Sunday minus 28 days, then 28 rows per item with covariates and empty sales).
- rohlik: Rohlik grocery, 300 best sellers of warehouse Prague_1; price = sell_price_main, promo = largest discount of the 7 discount types, holiday = holiday or shops closed
- rossmann: Rossmann drugstores, first 300 stores; promo = Promo (0/1), holiday = state holiday or store closed
- m5: M5 CA_1, 300 items; price = sell_price, promo = SNAP day, holiday = event day
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
EXT, OUT = ROOT / "data" / "external", ROOT / "examples"
(EXT / "covariates").mkdir(parents=True, exist_ok=True)


def explode(df, id_col, ts, cols):
    n = df[ts].map(len).to_numpy()
    out = pd.DataFrame({"item": np.repeat(df[id_col].to_numpy(), n), "date": pd.to_datetime(np.concatenate(df[ts].to_list())).strftime("%Y-%m-%d")})
    for k, v in cols.items():
        out[k] = np.concatenate([np.asarray(x) for x in df[v].to_list()])
    return out


def rohlik():
    c = ["name", "warehouse", "timestamp", "sales", "sell_price_main", "holiday", "shops_closed"] + [f"type_{i}_discount" for i in range(7)]
    r = pq.read_table(EXT / "rohlik" / "rohlik_sales_1D.parquet", columns=c).to_pandas()
    r = r[r.warehouse == "Prague_1"].reset_index(drop=True)
    r = r.assign(total=r.sales.map(np.nansum)).nlargest(300, "total").reset_index(drop=True)
    r["promo"] = [np.nanmax(np.stack([np.asarray(r[f"type_{i}_discount"][j], dtype=float) for i in range(7)]), axis=0) for j in range(len(r))]
    r["hol"] = [np.maximum(np.nan_to_num(np.asarray(a, float)), np.nan_to_num(np.asarray(b, float))) for a, b in zip(r.holiday, r.shops_closed)]
    return explode(r, "name", "timestamp", {"sales": "sales", "price": "sell_price_main", "promo": "promo", "holiday": "hol"})


def rossmann():
    r = pq.read_table(EXT / "rossmann" / "rossmann_1D.parquet", columns=["id", "timestamp", "Sales", "Promo", "StateHoliday", "Open"]).to_pandas().iloc[:300].reset_index(drop=True)
    r["hol"] = [((np.asarray(a) != "0") | (np.asarray(o) == 0)).astype(float) for a, o in zip(r.StateHoliday, r.Open)]
    return explode(r, "id", "timestamp", {"sales": "Sales", "promo": "Promo", "holiday": "hol"})[["item", "date", "sales", "promo", "holiday"]].rename(columns={"item": "store"})


def m5():
    p = pd.read_parquet(ROOT / "data" / "processed" / "panel_CA_1.parquet", columns=["item_id", "date", "d", "sales", "sell_price", "snap", "event_name_1"])
    p = p[p.d <= 1941]
    return pd.DataFrame({"item": p.item_id, "date": p.date.dt.strftime("%Y-%m-%d"), "sales": p.sales, "price": p.sell_price, "promo": p.snap.astype(float),
                         "holiday": p.event_name_1.notna().astype(float)})


def main():
    for name, fn in [("rohlik", rohlik), ("rossmann", rossmann), ("m5", m5)]:
        df = fn()
        df.to_csv(EXT / "covariates" / f"{name}_full.csv", index=False)
        d = pd.to_datetime(df.date)
        last = d.max() - pd.Timedelta(days=int((d.max().weekday() + 1) % 7))
        cut = last - pd.Timedelta(days=28)
        hist = df[d <= cut]
        fut = df[(d > cut) & (d <= last)].copy()
        fut["sales"] = np.nan
        pd.concat([hist, fut], ignore_index=True).to_csv(OUT / f"{name}_with_future.csv", index=False)
        print(name, len(df), "rows; history to", cut.date(), "future rows", len(fut), "columns", list(df.columns))


if __name__ == "__main__":
    sys.exit(main())
