"""Leave-one-dataset-out comparison of methods meant to work on any data (nothing is tuned on the held-out dataset).

Datasets: M5 CA_1 (the 300 app items), hierarchical sales, bakery, Rohlik, Online Retail II, Rossmann. Training-only extras: three more M5 stores
(TX_1, WI_1, CA_3). Three 28-day windows per dataset (cutoffs T-28, T-56, T-84 on Sundays, where the history allows >= 91 days); weekly WAPE
averaged over windows.

Methods
  seasonal naive, 28-day average, weekday average 8w   simple baselines
  current service    M5-trained lean model + range guard (not shown for M5, it was trained on it)
  norm-M5            scale-normalised LightGBM trained on M5 CA_1 only (targets before the first window)        [method 2]
  norm-multi         scale-normalised LightGBM trained on all other datasets; the held-out dataset (all M5 stores for M5) unseen   [methods 2+3]
  chronos            Chronos-Bolt-small zero-shot, mean of predicted quantiles                                   [method 4]
  ensemble           mean of norm-multi, chronos and weekday average                                             (declared before the run)

Declared success rule: on every dataset the best of {norm-multi, chronos, ensemble} is within 3% of the best simple baseline's weekly WAPE
or better, and on M5 the ensemble is not worse than the current lean model's validation WAPE (0.341) by more than 3%.
"""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demand import baselines as B  # noqa: E402
from demand import config as C  # noqa: E402
from demand import metrics as M  # noqa: E402
from demand import serving as S  # noqa: E402
from demand import universal as U  # noqa: E402
from demand import webapi  # noqa: E402

EX = C.ROOT / "examples"
EVAL = ["m5_CA_1", "hierarchical_sales", "bakery_daily", "rohlik_prague1_top300", "online_retail_ii_top300_invoice_lines", "rossmann_300_stores_euro"]
EXTRA_M5 = ["TX_1", "WI_1", "CA_3"]


def from_matrix(ids, dates, Y) -> S.Shop:
    first = np.where((Y > 0).any(axis=1), (Y > 0).argmax(axis=1), Y.shape[1])
    return S.Shop(list(ids), dates, Y.astype(float), None, first)


def load_all() -> dict[str, S.Shop]:
    shops = {}
    p = pd.read_parquet(C.PROC / f"panel_{C.STORE}.parquet", columns=["item_id", "date", "d", "sales"])
    p = p[p.d <= C.TEST_CUTOFF]
    shops["m5_CA_1"] = S.to_shop(p, "date", "item_id", "sales")
    for name in EVAL[1:]:
        c = webapi.EXAMPLES[name]
        df = webapi.read_csv_text((EX / f"{name}.csv").read_text(encoding="utf-8"))
        shops[name] = S.to_shop(df, c["date"], c["item"], c["sales"])
    cal = pd.read_csv(C.RAW / "calendar.csv", parse_dates=["date"]).date.iloc[:C.TEST_CUTOFF]
    raw = pd.read_csv(C.RAW / "sales_train_evaluation.csv")
    rng = np.random.default_rng(1)
    for st in EXTRA_M5:
        r = raw[raw.store_id == st]
        r = r.iloc[np.sort(rng.choice(len(r), 300, replace=False))]
        shops[f"m5_{st}"] = from_matrix(r.item_id, pd.DatetimeIndex(cal), r[[f"d_{i}" for i in range(1, C.TEST_CUTOFF + 1)]].to_numpy())
    return shops


def fit(frame: pd.DataFrame) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(**U.PARAMS).fit(frame[U.FEATURES], frame.y)


def main() -> None:
    t0 = time.time()
    shops = load_all()
    print({k: v.Y.shape for k, v in shops.items()}, f"{time.time() - t0:.0f}s", flush=True)
    frames = {k: U.training_frame(v) for k, v in shops.items()}
    frames["m5_CA_1_early"] = U.training_frame(shops["m5_CA_1"], last_origin=C.TEST_CUTOFF - 112)  # targets end before the first evaluation window
    print({k: len(v) for k, v in frames.items()}, flush=True)
    norm_m5 = fit(frames["m5_CA_1_early"])
    booster, meta = S.load_model()
    limit = meta["level_range"]["mean28_p999"]
    rows = []
    for held in EVAL:
        shop = shops[held]
        group = [k for k in frames if k != "m5_CA_1_early" and k != held and not (held == "m5_CA_1" and k.startswith("m5_"))]
        multi = fit(pd.concat([frames[k] for k in group], ignore_index=True))
        T = shop.Y.shape[1]
        per = {}
        for k in range(3):
            o = T - 28 * (k + 1)
            if o < U.MIN_HISTORY:
                continue
            hist, first = shop.Y[:, :o], np.minimum(shop.first_idx, o)
            actual = shop.Y[:, o:o + 28]
            wd8 = B.weekday_mean(hist)
            f = {"seasonal naive": B.seasonal_naive(hist), "28-day average": B.moving_average(hist), "weekday average 8w": wd8,
                 "norm-M5": U.predict_normalised(norm_m5, hist, first), "norm-multi": U.predict_normalised(multi, hist, first),
                 "chronos": U.chronos_forecast(hist)}
            if held != "m5_CA_1":
                sh = S.Shop(shop.ids, shop.dates[:o], hist, None, first)
                f["current service"] = S.forecast_with_info(sh, booster, meta["features"], limit)[0]
            f["ensemble"] = (f["norm-multi"] + f["chronos"] + wd8) / 3
            for name, fc in f.items():
                per.setdefault(name, []).append((M.wape(M.block_sums(actual), M.block_sums(fc)), M.wape(actual, fc), M.bias(actual, fc)))
        for name, v in per.items():
            a = np.array(v)
            rows.append(dict(dataset=held, method=name, windows=len(v), weekly=a[:, 0].mean(), daily=a[:, 1].mean(), bias=a[:, 2].mean()))
        print(f"{held} done ({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(rows)
    res.round(4).to_csv(C.REPORTS / "universal_lodo.csv", index=False)
    tab = res.pivot(index="method", columns="dataset", values="weekly")[EVAL].round(3)
    order = ["seasonal naive", "28-day average", "weekday average 8w", "current service", "norm-M5", "norm-multi", "chronos", "ensemble"]
    out = ["Weekly WAPE (lower is better), mean of up to 3 windows", tab.loc[order].to_string()]
    base = tab.loc[["seasonal naive", "28-day average", "weekday average 8w"]].min()
    cand = tab.loc[["norm-multi", "chronos", "ensemble"]].min()
    out.append("\nbest simple baseline: " + ", ".join(f"{d} {base[d]:.3f}" for d in EVAL))
    out.append("best of the new methods vs best baseline: " + ", ".join(f"{d} {cand[d] / base[d] - 1:+.1%}" for d in EVAL))
    out.append("rule 1 (every dataset within +3%): " + ("PASSED" if ((cand / base - 1) <= 0.03).all() else "NOT PASSED"))
    out.append(f"rule 2 (M5 ensemble {tab.loc['ensemble', 'm5_CA_1']:.3f} vs 0.341 lean model, within +3%): " + ("PASSED" if tab.loc["ensemble", "m5_CA_1"] <= 0.341 * 1.03 else "NOT PASSED"))
    out.append("\nBias (mean over windows)\n" + res.pivot(index="method", columns="dataset", values="bias")[EVAL].loc[order].round(3).to_string())
    text = "\n".join(out)
    print(text)
    (C.REPORTS / "universal_lodo.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
