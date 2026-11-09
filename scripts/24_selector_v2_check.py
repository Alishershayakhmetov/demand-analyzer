"""Old selector (2 windows, best single method, 7 candidates) vs new selector (up to 4 windows, mean of the top 2, 10 candidates incl. weekly-sum
LightGBM, bias-corrected Chronos and Chronos + LightGBM + weekday average), end to end on the last three 28-day windows of each open dataset.
Declared before the run: the new selector is better if (a) the mean over datasets of its weekly WAPE relative to the best simple baseline is lower,
and (b) no dataset gets worse than +3% of that baseline. M5 is excluded (the shipped models were trained on it)."""
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

booster, meta = S.load_model()
limit = meta["level_range"]["mean28_p999"]
norm = lgb.Booster(model_file=str(S.MODEL_DIR / "norm_model.txt"))
weekly = lgb.Booster(model_file=str(S.MODEL_DIR / "weekly_model.txt"))
rows, log = [], []
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
        fi = {i: j for j, i in enumerate(full.ids)}
        hi = {i: j for j, i in enumerate(hist.ids)}
        pos = full.dates.get_loc(cut) + 1
        actual = np.stack([full.Y[fi[i], pos:pos + 28] for i in ids])
        sub = S.Shop(ids, hist.dates, np.stack([hist.Y[hi[i]] for i in ids]), None, np.array([hist.first_idx[hi[i]] for i in ids]))
        old = SEL.select(sub, booster, meta["features"], limit, norm, None, True, max_windows=2, top_k=1, names=SEL.LEGACY)
        new = SEL.select(sub, booster, meta["features"], limit, norm, weekly, True)
        hy = sub.Y
        for m, f in {"old selector": old[0], "new selector": new[0], "seasonal naive": B.seasonal_naive(hy), "28-day average": B.moving_average(hy), "weekday average": B.weekday_mean(hy)}.items():
            rows.append(dict(dataset=name, window=k, method=m, weekly=M.wape(M.block_sums(actual), M.block_sums(f))))
        log.append(f"{name} w{k}: old -> {old[2]['method']} | new -> {new[2]['method']}")
r = pd.DataFrame(rows)
tab = r.groupby(["method", "dataset"]).weekly.mean().unstack()[list(webapi.EXAMPLES)]
base = tab.loc[["seasonal naive", "28-day average", "weekday average"]].min()
rel = (tab.loc[["old selector", "new selector"]] / base - 1)
out = ["\n".join(log), "", tab.round(3).to_string(), "", "relative to the best simple baseline:", rel.map(lambda v: f"{v:+.1%}").to_string(),
       "", f"mean relative: old {rel.loc['old selector'].mean():+.1%}, new {rel.loc['new selector'].mean():+.1%}",
       "rule (a) new mean lower: " + ("YES" if rel.loc["new selector"].mean() < rel.loc["old selector"].mean() else "NO"),
       "rule (b) no dataset above +3%: " + ("YES" if (rel.loc["new selector"] <= 0.03).all() else "NO")]
text = "\n".join(out); print(text)
(C.REPORTS / "selector_v2_check.txt").write_text(text, encoding="utf-8")
