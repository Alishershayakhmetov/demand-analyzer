"""EDA on the development period (d <= TEST_CUTOFF). The test fold is not touched."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demand import config as C  # noqa: E402

FIG = C.REPORTS / "figures"


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(C.PROC / f"panel_{C.STORE}.parquet")
    items = pd.read_csv(C.PROC / "items.csv")
    df = panel[panel.d <= C.TEST_CUTOFF].merge(items[["item_id", "stratum", "demand_class"]], on="item_id")
    out = []

    def say(s: str = "") -> None:
        print(s)
        out.append(s)

    say(f"panel {panel.shape}, items {panel.item_id.nunique()}, days {panel.d.max()}")
    say(f"missing sales: {panel.sales.isna().sum()}, negative sales: {(panel.sales < 0).sum()}")
    say(f"rows not on sale: {(panel.on_sale == 0).mean():.3%}; sales>0 while not on sale: {((panel.on_sale == 0) & (panel.sales > 0)).sum()}")
    say(f"zero-sales share (on sale): {(df[df.on_sale == 1].sales == 0).mean():.3f}")
    say("\nzero share and mean units/day by stratum:")
    say(df[df.on_sale == 1].groupby("stratum", observed=True).sales.agg(zero_share=lambda s: (s == 0).mean(), mean="mean", max="max").round(3).to_string())
    say("\nitems by category:")
    say(items.merge(panel[["item_id", "cat_id"]].drop_duplicates(), on="item_id").cat_id.value_counts().to_string())

    daily = df.groupby("date").sales.sum()
    wd = df.groupby(df.date.dt.dayofweek).sales.mean()
    say("\nmean units per item-day by weekday (0=Mon): " + ", ".join(f"{v:.2f}" for v in wd))
    snap = df[df.on_sale == 1].groupby("snap").sales.mean()
    say(f"SNAP day mean {snap[1]:.3f} vs non-SNAP {snap[0]:.3f} (+{snap[1] / snap[0] - 1:.1%})")
    ev = df[df.on_sale == 1].assign(event=df.event_name_1.notna()).groupby("event").sales.mean()
    say(f"event day mean {ev[True]:.3f} vs normal {ev[False]:.3f} ({ev[True] / ev[False] - 1:+.1%})")
    by_event = (df[df.on_sale == 1].dropna(subset=["event_name_1"]).groupby("event_name_1").sales.mean() / ev[False] - 1).sort_values()
    say("lowest/highest events vs normal day: " + ", ".join(f"{k} {v:+.0%}" for k, v in pd.concat([by_event.head(3), by_event.tail(3)]).items()))

    d = df[df.on_sale == 1].copy()
    d["price_ratio"] = d.sell_price / d.groupby("item_id").sell_price.transform("median")
    d["pr_bin"] = pd.cut(d.price_ratio, [0, 0.9, 0.98, 1.02, 1.1, 10], labels=["<0.9", "0.9-0.98", "~median", "1.02-1.1", ">1.1"])
    say("\nmean sales by price relative to item median (items with price changes):")
    say(d.groupby("pr_bin", observed=True).sales.agg(["mean", "size"]).round(3).to_string())

    say("\ndemand class x stratum:")
    say(pd.crosstab(items.stratum, items.demand_class).to_string())
    (C.REPORTS / "eda_summary.txt").write_text("\n".join(out), encoding="utf-8")

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    daily.plot(ax=ax[0, 0], lw=0.6)
    ax[0, 0].set_title(f"Total daily units, {C.STORE}, 300 items")
    ax[0, 1].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], wd.values)
    ax[0, 1].set_title("Mean units per item-day by weekday")
    m = df.assign(ym=df.date.dt.to_period("M")).groupby("ym").sales.sum()
    m.index = m.index.to_timestamp()
    m.iloc[1:-1].plot(ax=ax[1, 0])
    ax[1, 0].set_title("Monthly units (seasonality and trend)")
    for s, g in items.groupby("stratum", observed=True):
        ax[1, 1].hist(np.log10(g.adi.replace(np.inf, np.nan).dropna()), bins=20, alpha=0.6, label=s)
    ax[1, 1].set_title("log10(ADI) by volume stratum")
    ax[1, 1].legend()
    fig.tight_layout()
    fig.savefig(FIG / "eda_overview.png", dpi=110)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for ax_, (s, g) in zip(ax, items.groupby("stratum", observed=True)):
        item = g.sort_values("units_last_year").item_id.iloc[len(g) // 2]
        x = df[(df.item_id == item) & (df.d > C.DEV_CUTOFFS[0] - 200)]
        ax_.plot(x.date, x.sales, lw=0.8)
        ax_.set_title(f"{s}: {item}")
    fig.tight_layout()
    fig.savefig(FIG / "eda_examples.png", dpi=110)


if __name__ == "__main__":
    main()
