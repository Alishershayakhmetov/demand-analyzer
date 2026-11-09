"""Download open sales datasets other than M5 and turn them into CSV files that can be uploaded to the app (examples/).

Run with isolation, the downloaded files are untrusted data:  python -I -X utf8 scripts/18_external_examples.py
Sources (no account needed):
- bakery:       Kaggle "Transactions from a bakery" (BreadBasket_DMS.csv), GitHub mirror yortos/bakery
- hierarchical: daily quantities, 118 series, 2014-2018 (autogluon/fev_datasets on Hugging Face)
- rohlik:       Rohlik Group grocery e-commerce, daily sales per product and warehouse (autogluon/fev_datasets)
- rossmann:     Rossmann drugstores, daily SALES IN EURO per store (autogluon/fev_datasets); a scale stress test, not unit sales
- online_retail: UCI Online Retail II (CC BY 4.0), transactions of a UK gift wholesaler, raw invoice lines (needs openpyxl)
Check each source's terms before reusing the data outside coursework.
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
EXT, OUT = ROOT / "data" / "external", ROOT / "examples"
HF = "https://huggingface.co/datasets/autogluon/fev_datasets/resolve/main"
SOURCES = {
    "bakery/BreadBasket_DMS.csv": "https://raw.githubusercontent.com/yortos/bakery/master/BreadBasket_DMS.csv",
    "hierarchical/hierarchical_sales_1D.parquet": f"{HF}/hierarchical_sales/1D/train-00000-of-00001.parquet",
    "rohlik/rohlik_sales_1D.parquet": f"{HF}/rohlik_sales/1D/train-00000-of-00001.parquet",
    "rossmann/rossmann_1D.parquet": f"{HF}/rossmann/1D/train-00000-of-00001.parquet",
    "online_retail/online_retail_ii.zip": "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
}


def fetch() -> None:
    for rel, url in SOURCES.items():
        path = EXT / rel
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        print("downloading", url)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as r, open(path, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)


def explode(df: pd.DataFrame, id_col: str, ts_col: str, value_cols: dict[str, str]) -> pd.DataFrame:
    """One row per series holds arrays; return one row per series and day."""
    n = df[ts_col].map(len).to_numpy()
    out = pd.DataFrame({"item": np.repeat(df[id_col].to_numpy(), n), "date": pd.to_datetime(np.concatenate(df[ts_col].to_list())).strftime("%Y-%m-%d")})
    for src, dst in value_cols.items():
        out[dst] = np.concatenate(df[src].to_list())
    return out


def bakery() -> pd.DataFrame:
    b = pd.read_csv(EXT / "bakery" / "BreadBasket_DMS.csv")
    b = b[b.Item != "NONE"]
    return b.groupby(["Date", "Item"]).size().rename("sales").reset_index().rename(columns={"Date": "date", "Item": "item"})


def hierarchical() -> pd.DataFrame:
    h = pq.read_table(EXT / "hierarchical" / "hierarchical_sales_1D.parquet").to_pandas()
    return explode(h, "id", "timestamp", {"target": "sales"})[["date", "item", "sales"]]


def rohlik() -> pd.DataFrame:
    r = pq.read_table(EXT / "rohlik" / "rohlik_sales_1D.parquet", columns=["name", "warehouse", "timestamp", "sales", "sell_price_main"]).to_pandas()
    r = r[r.warehouse == "Prague_1"].reset_index(drop=True)
    assert r.name.is_unique
    keep = r.assign(total=r.sales.map(np.nansum)).nlargest(300, "total").reset_index(drop=True)
    return explode(keep, "name", "timestamp", {"sales": "sales", "sell_price_main": "price"})[["date", "item", "sales", "price"]]


def rossmann() -> pd.DataFrame:
    r = pq.read_table(EXT / "rossmann" / "rossmann_1D.parquet", columns=["id", "timestamp", "Sales"]).to_pandas()
    r = r.iloc[:300].reset_index(drop=True)
    return explode(r, "id", "timestamp", {"Sales": "sales"})[["date", "item", "sales"]].rename(columns={"item": "store"})


def online_retail() -> pd.DataFrame:
    with zipfile.ZipFile(EXT / "online_retail" / "online_retail_ii.zip") as z:
        z.extract("online_retail_II.xlsx", EXT / "online_retail")
    sheets = pd.read_excel(EXT / "online_retail" / "online_retail_II.xlsx", sheet_name=None, usecols=["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price"])
    df = pd.concat(sheets.values(), ignore_index=True)
    df["StockCode"] = df.StockCode.astype(str)
    product = df.StockCode.str.match(r"^\d{5}")  # real products; drops POST, DOT, M, BANK CHARGES and similar
    df = df[product]
    top = df[df.Quantity > 0].groupby("StockCode").Quantity.sum().nlargest(300).index
    df = df[df.StockCode.isin(top)]
    df["InvoiceDate"] = pd.to_datetime(df.InvoiceDate).dt.strftime("%Y-%m-%d %H:%M:%S")
    return df[["InvoiceDate", "StockCode", "Quantity", "Price"]]  # raw invoice lines: cancellations are negative, many lines per day


def main() -> None:
    fetch()
    OUT.mkdir(exist_ok=True)
    for name, fn in [("bakery_daily", bakery), ("hierarchical_sales", hierarchical), ("rohlik_prague1_top300", rohlik),
                     ("rossmann_300_stores_euro", rossmann), ("online_retail_ii_top300_invoice_lines", online_retail)]:
        df = fn()
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}.csv: {len(df):,} rows, columns {list(df.columns)}")


if __name__ == "__main__":
    sys.exit(main())
