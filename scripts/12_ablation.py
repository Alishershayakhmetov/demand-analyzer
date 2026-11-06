"""Feature-group ablation of the default LightGBM on the dev + validation folds (test fold not used).

Each variant is retrained on every fold without a feature group and scored on the 28-day forecast (WAPE at three
aggregation levels) and on the inventory needed for a 95% revenue-weighted fill rate (lead time 3 days, weekly refresh,
same simulator as scripts/10_inventory.py). The full model is trained with three seeds to measure the noise floor.
"""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demand import config as C  # noqa: E402
from demand import metrics as M  # noqa: E402
from demand.evaluate import load_matrices  # noqa: E402
from demand.features import CATEGORICAL, FEATURES, build_rows, training_rows  # noqa: E402
from demand.modeling import COMMON, inventory_days  # noqa: E402

FOLDS = C.DEV_CUTOFFS + [C.VAL_CUTOFF]
PRICE = ["price", "price_vs_median", "price_vs_origin"]
WD_HIST = ["wd_mean8", "wd_mean4", "wd_last"]
LEVEL = ["mean7", "mean14", "mean28", "mean56", "mean91", "std28", "nz28", "nz91", "days_since_sale", "age"]
VARIANTS = {
    "full (seed 42)": ([], 42),
    "full (seed 1)": ([], 1),
    "full (seed 2)": ([], 2),
    "no price": (PRICE, 42),
    "no SNAP": (["snap"], 42),
    "no events": (["event"], 42),
    "no month / day of month": (["month", "dom"], 42),
    "no weekday of target day": (["wday"], 42),
    "no weekday history": (WD_HIST, 42),
    "no weekday at all": (["wday", *WD_HIST], 42),
    "no known-future covariates": ([*PRICE, "snap", "event", "month", "dom", "wday"], 42),
    "no long history (56/91 d, age)": (["mean56", "mean91", "nz91", "age"], 42),
    "no item type": (["dept", "cat"], 42),
    "no horizon step": (["h"], 42),
    "recent-demand level + horizon only": ([f for f in FEATURES if f not in [*LEVEL, "h"]], 42),
}


def main() -> None:
    m = load_matrices()
    n, H = len(m.ids), C.HORIZON
    preds = {v: {} for v in [*VARIANTS, "28-day average (reference)"]}
    for cutoff in FOLDS:
        t0 = time.time()
        train = training_rows(m, cutoff)
        origins = [build_rows(m, cutoff + 7 * k, with_target=False) for k in range(4)]
        for name, (drop, seed) in VARIANTS.items():
            feats = [f for f in FEATURES if f not in drop]
            model = lgb.LGBMRegressor(**COMMON, random_state=seed).fit(
                train[feats], train.y, categorical_feature=[c for c in CATEGORICAL if c in feats])
            preds[name][cutoff] = [np.clip(model.predict(rows[feats]), 0, None).reshape(n, H) * rows.on_sale.to_numpy().reshape(n, H)
                                   for rows in origins]
        preds["28-day average (reference)"][cutoff] = [
            np.repeat(m.Y[:, :cutoff + 7 * k][:, -28:].mean(axis=1, keepdims=True), H, axis=1) * rows.on_sale.to_numpy().reshape(n, H)
            for k, rows in enumerate(origins)]
        print(f"fold {cutoff} done in {time.time() - t0:.0f}s", flush=True)

    rows = []
    for name, by_fold in preds.items():
        d, w, m28 = [], [], []
        for cutoff in FOLDS:
            actual, f = m.Y[:, cutoff:cutoff + H].astype(float), by_fold[cutoff][0]
            d.append(M.wape(actual, f))
            w.append(M.wape(M.block_sums(actual), M.block_sums(f)))
            m28.append(M.wape(M.block_sums(actual, 28), M.block_sums(f, 28)))
        rows.append(dict(variant=name, daily_wape=np.mean(d), weekly_wape=np.mean(w), wape_28d=np.mean(m28),
                         inv_days_95=inventory_days(m, by_fold, FOLDS)))
    res = pd.DataFrame(rows).set_index("variant")
    res.to_csv(C.REPORTS / "ablation_raw.csv")

    full = res.loc[[v for v in res.index if v.startswith("full")]]
    ref, noise = full.mean(), full.std()
    out = res.drop(index=[v for v in res.index if v.startswith("full")]).copy()
    out.loc["FULL model (mean of 3 seeds)"] = ref
    delta = (out - ref) / ref
    table = out.round(4).join(delta.add_prefix("rel_").round(4))
    table = table.loc[["FULL model (mean of 3 seeds)", *[v for v in table.index if v != "FULL model (mean of 3 seeds)"]]]
    text = ["Ablation, mean over 4 folds (default hyper-parameters, 28-day average is the reference baseline)",
            table.to_string(),
            "",
            "noise floor, std of the full model over 3 seeds (absolute): " + ", ".join(f"{k} {v:.4f}" for k, v in noise.items()),
            "noise floor relative to the mean: " + ", ".join(f"{k} {v / ref[k]:.2%}" for k, v in noise.items())]
    print("\n".join(text))
    (C.REPORTS / "ablation.txt").write_text("\n".join(text), encoding="utf-8")
    table.to_csv(C.REPORTS / "ablation.csv")


if __name__ == "__main__":
    main()
