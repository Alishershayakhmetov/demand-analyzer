"""Do price / promotion / holiday columns known for the next 28 days improve the forecast? End to end on Rohlik (price, discounts, holidays), Rossmann (promo,
holidays) and M5 CA_1 (price, SNAP, events); three 28-day windows per dataset; the file carries the covariates of the window to forecast but not its sales.
Declared before the run:
  (a) the selector with the covariate candidates has a lower mean weekly WAPE, relative to the best simple baseline over the three datasets, than the same
      selector without covariates;
  (b) the covariate model alone is better than the weekday average on Rohlik and on Rossmann (the two sets with real promotions).
For M5 the selector excludes the candidates trained on M5 (they have seen these days)."""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demand import baselines as B  # noqa: E402
from demand import config as C  # noqa: E402
from demand import covariate as CV  # noqa: E402
from demand import metrics as M  # noqa: E402
from demand import selector as SEL  # noqa: E402
from demand import serving as S  # noqa: E402

booster, meta = S.load_model(); limit = meta["level_range"]["mean28_p999"]
norm = lgb.Booster(model_file=str(S.MODEL_DIR / "norm_model.txt")); weekly = lgb.Booster(model_file=str(S.MODEL_DIR / "weekly_model.txt"))
BASE = SEL.SIMPLE + ["Chronos-Bolt", "Chronos-Bolt (bias-corrected)", "Chronos-Bolt + weekday average", "MSTL (weekly season + trend)", "TSB", "Last year (level-adjusted)"]
TRAINED = ["LightGBM (M5)", "LightGBM (scale-normalised)", "LightGBM (weekly)", "Chronos-Bolt + LightGBM (M5) + weekday average"]
COVN = ["Covariate LightGBM", "Covariate LightGBM + Chronos-Bolt"]
SETS = {"rohlik": dict(item="item", cols=dict(price_col="price", promo_col="promo", holiday_col="holiday")),
        "rossmann": dict(item="store", cols=dict(promo_col="promo", holiday_col="holiday")),
        "m5": dict(item="item", cols=dict(price_col="price", promo_col="promo", holiday_col="holiday"))}
rows, log = [], []
for name, c in SETS.items():
    full = pd.read_csv(C.ROOT / "data" / "external" / "covariates" / f"{name}_full.csv")
    d = pd.to_datetime(full.date)
    end = d.max() - pd.Timedelta(days=int((d.max().weekday() + 1) % 7))
    truth = S.to_shop(full, "date", c["item"], "sales")
    fi = {i: j for j, i in enumerate(truth.ids)}
    for k in range(3):
        cut = end - pd.Timedelta(days=28 * (k + 1))
        part = full[(d <= cut + pd.Timedelta(days=28))].copy()
        part.loc[pd.to_datetime(part.date) > cut, "sales"] = np.nan
        shop = S.to_shop(part, "date", c["item"], "sales", **c["cols"])
        plain = S.to_shop(part[pd.to_datetime(part.date) <= cut], "date", c["item"], "sales")
        ids = [i for i in shop.ids if i in fi]
        pos = truth.dates.get_loc(cut) + 1
        actual = np.stack([truth.Y[fi[i], pos:pos + 28] for i in ids])
        keep = [shop.ids.index(i) for i in ids]
        sub = S.Shop(ids, shop.dates, shop.Y[keep], None, shop.first_idx[keep], [], {**shop.cov, **{k2: v[keep] for k2, v in shop.cov.items() if isinstance(v, np.ndarray) and v.ndim == 2}})
        m5_trained = name == "m5"
        nm, wm = (None, None) if m5_trained else (norm, weekly)
        names_plain = BASE if m5_trained else BASE + TRAINED
        nocov = S.Shop(ids, plain.dates, plain.Y[[plain.ids.index(i) for i in ids]], None, plain.first_idx[[plain.ids.index(i) for i in ids]])
        a = SEL.select(nocov, booster, meta["features"], limit, nm, wm, True, names=names_plain, use_stat=True)
        b = SEL.select(sub, booster, meta["features"], limit, nm, wm, True, names=names_plain + COVN, use_stat=True)
        bt = CV.fit(sub, sub.Y.shape[1] - 28); only = CV.predict(bt, sub, sub.Y.shape[1]) if bt is not None else None
        hy = sub.Y
        res = {"selector without covariates": a[0], "selector with covariates": b[0], "seasonal naive": B.seasonal_naive(hy), "28-day average": B.moving_average(hy), "weekday average": B.weekday_mean(hy)}
        if only is not None:
            res["covariate model alone"] = only
        for mname, f in res.items():
            rows.append(dict(dataset=name, method=mname, weekly=M.wape(M.block_sums(actual), M.block_sums(f)), bias=M.bias(actual, f)))
        log.append(f"{name} w{k}: without -> {a[2]['method']} | with -> {b[2]['method']}")
r = pd.DataFrame(rows)
tab = r.groupby(["method", "dataset"]).weekly.mean().unstack()[list(SETS)]
base = tab.loc[["seasonal naive", "28-day average", "weekday average"]].min()
rel = tab.loc[["selector without covariates", "selector with covariates", "covariate model alone"]] / base - 1
ok_a = rel.loc["selector with covariates"].mean() < rel.loc["selector without covariates"].mean()
ok_b = all(tab.loc["covariate model alone", d] < tab.loc["weekday average", d] for d in ("rohlik", "rossmann"))
text = "\n".join(log) + "\n\nWeekly WAPE\n" + tab.round(3).to_string() + "\n\nrelative to the best simple baseline\n" + rel.map(lambda v: f"{v:+.1%}").to_string() + \
       f"\nmean: without {rel.loc['selector without covariates'].mean():+.1%}, with {rel.loc['selector with covariates'].mean():+.1%}\n\nbias of the covariate model alone: " + \
       r[r.method == "covariate model alone"].groupby("dataset").bias.mean().round(3).to_dict().__str__() + \
       f"\nrule (a): {'PASSED' if ok_a else 'NOT PASSED'}\nrule (b): {'PASSED' if ok_b else 'NOT PASSED'}"
print(text); (C.REPORTS / "covariate_check.txt").write_text(text, encoding="utf-8")
