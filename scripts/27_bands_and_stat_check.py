"""Check of (1) the weekly 80% corridor and (2) the new candidates (MSTL, TSB, last year) in the selector, end to end on the open datasets.
Declared before the run:
  (1) realised coverage of the corridor on the next window, mean over datasets, within 80% +- 8 points, and every dataset within +- 15 points;
  (2) selector v3 (new candidates) has a lower mean weekly WAPE relative to the best simple baseline than v2 (-4.2% in reports/selector_v2_check.txt),
      and does better than v2 on Online Retail II.
Three 28-day windows per dataset; M5 excluded (the shipped models were trained on it)."""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demand import baselines as B  # noqa: E402
from demand import config as C  # noqa: E402
from demand import metrics as M  # noqa: E402
from demand import selector as SEL  # noqa: E402
from demand import serving as S  # noqa: E402
from demand import webapi  # noqa: E402

booster, meta = S.load_model(); limit = meta["level_range"]["mean28_p999"]
norm = lgb.Booster(model_file=str(S.MODEL_DIR / "norm_model.txt")); weekly = lgb.Booster(model_file=str(S.MODEL_DIR / "weekly_model.txt"))
V2 = SEL.SIMPLE + ["LightGBM (M5)", "LightGBM (scale-normalised)", "LightGBM (weekly)", "Chronos-Bolt", "Chronos-Bolt (bias-corrected)", "Chronos-Bolt + weekday average",
                   "Chronos-Bolt + LightGBM (M5) + weekday average"]
rows, cov, log = [], [], []
for name, c in webapi.EXAMPLES.items():
    df = webapi.read_csv_text((C.ROOT / "examples" / f"{name}.csv").read_text(encoding="utf-8"))
    cols = dict(date_col=c["date"], item_col=c["item"], sales_col=c["sales"], price_col=None)
    full = S.to_shop(df, **cols)
    d = pd.to_datetime(df[c["date"]], errors="coerce").dt.normalize()
    for k in range(3):
        cut = full.dates[-1] - pd.Timedelta(days=28 * (k + 1))
        try:
            hist = S.to_shop(df[d <= cut], **cols)
        except ValueError:
            continue
        ids = [i for i in hist.ids if i in set(full.ids)]
        fi = {i: j for j, i in enumerate(full.ids)}; hi = {i: j for j, i in enumerate(hist.ids)}
        pos = full.dates.get_loc(cut) + 1
        actual = np.stack([full.Y[fi[i], pos:pos + 28] for i in ids])
        sub = S.Shop(ids, hist.dates, np.stack([hist.Y[hi[i]] for i in ids]), None, np.array([hist.first_idx[hi[i]] for i in ids]))
        v2 = SEL.select(sub, booster, meta["features"], limit, norm, weekly, True, names=V2, use_stat=False)
        v3 = SEL.select(sub, booster, meta["features"], limit, norm, weekly, True, use_stat=True)
        for m, f in {"selector v2": v2[0], "selector v3": v3[0], "seasonal naive": B.seasonal_naive(sub.Y), "28-day average": B.moving_average(sub.Y), "weekday average": B.weekday_mean(sub.Y)}.items():
            rows.append(dict(dataset=name, method=m, weekly=M.wape(M.block_sums(actual), M.block_sums(f))))
        band = v3[2].get("band")
        if band:
            ahead = v3[0].reshape(-1, 4, 7).sum(2); act_w = actual.reshape(-1, 4, 7).sum(2)
            lo = np.clip(ahead + band[0] * np.sqrt(ahead + 1), 0, None); hi_ = ahead + band[1] * np.sqrt(ahead + 1)
            cov.append(dict(dataset=name, window=k, coverage=float(((act_w >= lo) & (act_w <= hi_)).mean())))
        log.append(f"{name} w{k}: v3 -> {v3[2]['method']}")
r = pd.DataFrame(rows); tab = r.groupby(["method", "dataset"]).weekly.mean().unstack()[list(webapi.EXAMPLES)]
base = tab.loc[["seasonal naive", "28-day average", "weekday average"]].min()
rel = tab.loc[["selector v2", "selector v3"]] / base - 1
cv = pd.DataFrame(cov).groupby("dataset").coverage.mean()
ok1 = abs(cv.mean() - 0.8) <= 0.08 and (abs(cv - 0.8) <= 0.15).all()
ok2 = rel.loc["selector v3"].mean() < rel.loc["selector v2"].mean() and rel.loc["selector v3", "online_retail_ii_top300_invoice_lines"] < rel.loc["selector v2", "online_retail_ii_top300_invoice_lines"]
text = "\n".join(log) + "\n\nWeekly WAPE\n" + tab.round(3).to_string() + "\n\nrelative to best simple baseline\n" + rel.map(lambda v: f"{v:+.1%}").to_string() + \
       f"\nmean: v2 {rel.loc['selector v2'].mean():+.1%}, v3 {rel.loc['selector v3'].mean():+.1%}\n\nCorridor coverage (target 80%)\n" + cv.round(3).to_string() + \
       f"\nmean {cv.mean():.3f}\nrule 1 (corridor): {'PASSED' if ok1 else 'NOT PASSED'}\nrule 2 (new candidates): {'PASSED' if ok2 else 'NOT PASSED'}"
print(text); (C.REPORTS / "bands_and_stat_check.txt").write_text(text, encoding="utf-8")
