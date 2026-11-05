"""Error analysis of the stored forecasts on the dev + validation folds (the test fold is not touched).

Main model is `lgbm_default` (never tuned on these folds); `moving_avg_28`, `AutoETS` and `seasonal_naive` are the comparison.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demand import config as C  # noqa: E402
from demand import metrics as M  # noqa: E402
from demand.evaluate import load_forecasts, load_matrices  # noqa: E402

MODELS = ["lgbm_default", "moving_avg_28", "AutoETS", "seasonal_naive"]
SHORT = {"lgbm_default": "LightGBM", "moving_avg_28": "MA28", "AutoETS": "ETS", "seasonal_naive": "SNaive"}
FIG = C.REPORTS / "figures"
PRICE_BINS = [0, 0.9, 0.98, 1.02, 1.1, 100]
PRICE_LABELS = ["promo (<0.9)", "small cut", "normal", "small rise", "rise (>1.1)"]
out: list[str] = []


def say(s: str = "") -> None:
    print(s)
    out.append(s)


def table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """WAPE per model, LightGBM bias and its gain over MA28, per segment of `col`."""
    g = df.groupby(col, observed=True)
    res = pd.DataFrame({"rows": g.size(), "units_share": g.y.sum() / df.y.sum()})
    for m in MODELS:
        res[SHORT[m]] = g.apply(lambda x, m=m: (x.y - x[m]).abs().sum() / x.y.sum(), include_groups=False)
    res["LGBM_bias"] = g.apply(lambda x: (x.lgbm_default - x.y).sum() / x.y.sum(), include_groups=False)
    res["gain_vs_MA28"] = 1 - res["LightGBM"] / res["MA28"]
    return res.round(3)


def main() -> None:
    m = load_matrices()
    fc = load_forecasts("baselines", "lgbm")
    items = m.items
    n = len(m.ids)
    H = C.HORIZON
    frames = []
    for cutoff, d in sorted(fc.items()):
        days = np.arange(cutoff, cutoff + H)
        med = np.nanmedian(m.PR[:, cutoff - 91:cutoff], axis=1)
        cols = {
            "cutoff": cutoff, "item": np.repeat(np.arange(n), H), "h": np.tile(np.arange(1, H + 1), n),
            "d": np.tile(days + 1, n), "y": d["actual"].ravel(),
            "price_ratio": (m.PR[:, days] / med[:, None]).ravel(), "on_sale": m.on_sale[:, days].ravel(),
            "snap": np.tile(m.snap[days], n), "event": np.tile(m.event[days], n), "wday": np.tile(m.wday[days], n),
            "q90": d["lgbm_default_q90"].ravel(), "q90_base": d["weekday_q90_8w"].ravel(),
        }
        for mod in MODELS:
            cols[mod] = d[mod].ravel()
        frames.append(pd.DataFrame(cols))
    df = pd.concat(frames, ignore_index=True)
    for c in ["demand_class", "stratum"]:
        df[c] = items[c].to_numpy()[df.item]
    df["cat"] = pd.Series(m.cat).map(dict(enumerate(sorted(m.items.item_id.str.split("_").str[0].unique())))).to_numpy()[df.item]
    df["price_bin"] = pd.cut(df.price_ratio, PRICE_BINS, labels=PRICE_LABELS)
    sale = df[df.on_sale]

    say(f"rows {len(df):,} (items {n}, folds {sorted(fc)}), on-sale rows {len(sale):,}")
    say("WAPE per segment, daily level. gain_vs_MA28 > 0 means LightGBM is better.\n")
    for name, col in [("demand class", "demand_class"), ("volume stratum", "stratum"), ("category", "cat"),
                      ("price vs 91-day median", "price_bin"), ("SNAP day", "snap"), ("weekday (1=Sat)", "wday"),
                      ("event day", "event")]:
        say(f"-- by {name}")
        say(table(sale, col).to_string())
        say()
    sale = sale.assign(h_block=pd.cut(sale.h, [0, 7, 14, 21, 28], labels=["days 1-7", "8-14", "15-21", "22-28"]))
    say("-- by horizon block")
    say(table(sale, "h_block").to_string())

    err = (sale.y - sale.lgbm_default).abs()
    say("\n-- where the LightGBM absolute error comes from")
    say(f"actual == 0 days: {len(sale[sale.y == 0]) / len(sale):.1%} of on-sale item-days, {err[sale.y == 0].sum() / err.sum():.1%} of the error")
    over = (sale.lgbm_default - sale.y).clip(lower=0).sum() / err.sum()
    say(f"over-forecast share of the error {over:.1%}, under-forecast {1 - over:.1%}")
    say("error share by stratum: " + ", ".join(f"{k} {v:.1%}" for k, v in (err.groupby(sale.stratum, observed=True).sum() / err.sum()).items()))
    per_item = err.groupby(sale.item).sum().sort_values(ascending=False)
    say(f"top 10% items ({n // 10}) carry {per_item.head(n // 10).sum() / per_item.sum():.1%} of the error")

    # weekly view
    wk_rows = []
    for cutoff, d in sorted(fc.items()):
        days = np.arange(cutoff, cutoff + H)
        med = np.nanmedian(m.PR[:, cutoff - 91:cutoff], axis=1)
        ratio = (m.PR[:, days] / med[:, None]).reshape(n, 4, 7)
        w = {"cutoff": cutoff, "item": np.repeat(np.arange(n), 4), "week": np.tile(np.arange(1, 5), n),
             "y": M.block_sums(d["actual"]).ravel(), "min_ratio": np.nanmin(np.where(np.isnan(ratio), np.inf, ratio), axis=2).ravel(),
             "snap_days": m.snap[days].reshape(4, 7).sum(axis=1)[np.tile(np.arange(4), n)]}
        for mod in MODELS:
            w[mod] = M.block_sums(d[mod]).ravel()
        wk_rows.append(pd.DataFrame(w))
    wk = pd.concat(wk_rows, ignore_index=True)
    wk = wk[np.isfinite(wk.min_ratio)]
    for c in ["demand_class", "stratum"]:
        wk[c] = items[c].to_numpy()[wk.item]
    wk["promo_week"] = np.where(wk.min_ratio < 0.9, "promo week (<0.9)", "no promo")
    wk["week_ahead"] = wk.week
    say("\n\nWEEKLY level (item-week sums)")
    for name, col in [("demand class", "demand_class"), ("volume stratum", "stratum"), ("promo week", "promo_week"),
                      ("week ahead", "week_ahead"), ("SNAP days in week", "snap_days")]:
        say(f"-- by {name}")
        say(table(wk, col).to_string())
        say()

    # quantile calibration
    say("-- P90 calibration (share of days with actual <= forecast; target 0.90)")
    cal = sale.groupby("demand_class", observed=True).apply(
        lambda x: pd.Series({"LGBM q90": (x.y <= x.q90).mean(), "weekday q90": (x.y <= x.q90_base).mean()}), include_groups=False)
    cal2 = sale.groupby("stratum", observed=True).apply(
        lambda x: pd.Series({"LGBM q90": (x.y <= x.q90).mean(), "weekday q90": (x.y <= x.q90_base).mean()}), include_groups=False)
    say(pd.concat([cal, cal2]).round(3).to_string())

    # worst items
    info = df.groupby("item").agg(units=("y", "sum"))
    info["lgbm_abs_err"] = err.groupby(sale.item).sum()
    info["err_share"] = info.lgbm_abs_err / info.lgbm_abs_err.sum()
    info["daily_wape"] = info.lgbm_abs_err / info.units
    info["bias"] = (sale.lgbm_default - sale.y).groupby(sale.item).sum() / info.units
    info["wape_ma28"] = (sale.y - sale.moving_avg_28).abs().groupby(sale.item).sum() / info.units
    wkw = wk.groupby("item").apply(lambda x: (x.y - x.lgbm_default).abs().sum() / x.y.sum(), include_groups=False)
    info["weekly_wape"] = wkw
    pre = [m.Y[:, c - 91:c].mean(axis=1) for c in sorted(fc)]
    post = [m.Y[:, c:c + H].mean(axis=1) for c in sorted(fc)]
    info["level_shift"] = np.nanmedian(np.abs(np.log((np.array(post) + 0.05) / (np.array(pre) + 0.05))), axis=0)
    info["promo_days"] = (sale.price_ratio < 0.9).groupby(sale.item).sum()
    info["item_id"] = m.ids
    info["class"] = items.demand_class.to_numpy()
    info["adi"] = items.adi.to_numpy().round(2)
    info["cv2"] = items.cv2.to_numpy().round(2)
    worst = info.sort_values("lgbm_abs_err", ascending=False).head(10)
    say("\n-- 10 items with the largest LightGBM absolute error (4 folds)")
    say(worst[["item_id", "class", "adi", "cv2", "units", "err_share", "daily_wape", "weekly_wape", "wape_ma28", "bias", "level_shift", "promo_days"]]
        .round(3).to_string(index=False))
    say(f"median level_shift over all items {info.level_shift.median():.2f}, over the worst 10 {worst.level_shift.median():.2f}")
    worst.to_csv(C.REPORTS / "worst_items.csv", index=False)
    (C.REPORTS / "error_analysis.txt").write_text("\n".join(out), encoding="utf-8")

    # figures
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for a, (title, t) in zip(ax, [("weekly WAPE by demand class", table(wk, "demand_class")),
                                  ("weekly WAPE by volume stratum", table(wk, "stratum")),
                                  ("daily WAPE by price vs median", table(sale, "price_bin"))]):
        t[[SHORT[k] for k in MODELS]].plot.bar(ax=a, rot=20, width=0.8)
        a.set_title(title)
        a.set_xlabel("")
    fig.tight_layout()
    fig.savefig(FIG / "error_by_segment.png", dpi=110)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    byh = sale.groupby("h").apply(lambda x: pd.Series({SHORT[k]: (x.y - x[k]).abs().sum() / x.y.sum() for k in MODELS}), include_groups=False)
    byh.plot(ax=ax[0])
    ax[0].set_title("daily WAPE by forecast horizon step")
    t = table(sale, "wday")
    ax[1].bar(t.index.astype(str), t.LGBM_bias)
    ax[1].axhline(0, c="k", lw=0.5)
    ax[1].set_title("LightGBM bias by weekday (1=Sat)")
    fig.tight_layout()
    fig.savefig(FIG / "error_horizon_weekday.png", dpi=110)

    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    for a, (idx, row) in zip(axes.ravel(), worst.iterrows()):
        c = C.VAL_CUTOFF
        hist = m.Y[idx, c - 90:c]
        a.plot(range(-90, 0), hist, lw=0.7, c="grey")
        a.plot(range(0, H), fc[c]["actual"][idx], c="C0", label="actual")
        a.plot(range(0, H), fc[c]["lgbm_default"][idx], c="C3", label="LightGBM")
        a.plot(range(0, H), fc[c]["moving_avg_28"][idx], c="C2", ls="--", label="MA28")
        a.set_title(f"{row['item_id']} ({row['class']})", fontsize=9)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Worst 10 items, validation fold: 90 days of history, then 28-day actual vs forecast")
    fig.tight_layout()
    fig.savefig(FIG / "error_worst_items.png", dpi=100)


if __name__ == "__main__":
    main()
