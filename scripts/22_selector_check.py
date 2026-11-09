"""End-to-end check of the method selector: the service (webapi.analyze, backtest selection inside) forecasts the last three 28-day windows of each
open dataset from the CSV before it. Rule declared before the run: on every dataset the service's weekly WAPE (mean of windows) is within +3% of
the best simple baseline's. M5 is left out: both shipped models were trained on it."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demand import baselines as B  # noqa: E402
from demand import config as C  # noqa: E402
from demand import metrics as M  # noqa: E402
from demand import serving as S  # noqa: E402
from demand import webapi  # noqa: E402

rows, log = [], []
for name, c in webapi.EXAMPLES.items():
    df = webapi.read_csv_text((C.ROOT / "examples" / f"{name}.csv").read_text(encoding="utf-8"))
    cols = dict(date_col=c["date"], item_col=c["item"], sales_col=c["sales"], price_col=None)
    full = S.to_shop(df, **cols)
    d = pd.to_datetime(df[c["date"]], errors="coerce").dt.normalize()
    for k in range(3):
        cut = full.dates[-1] - pd.Timedelta(days=28 * (k + 1))
        hist_df = df[d <= cut]
        try:
            hist = S.to_shop(hist_df, **cols)
        except ValueError:
            continue
        pc = {x: c[x] for x in ("date", "item", "sales")}
        res = webapi.analyze(dict(source="csv", csv=hist_df.to_csv(index=False), cols=pc))
        ids = [i for i in hist.ids if i in set(full.ids)]
        fi = {i: j for j, i in enumerate(full.ids)}
        pos = full.dates.get_loc(cut) + 1
        actual = np.stack([full.Y[fi[i], pos:pos + 28] for i in ids])
        hy = np.stack([hist.Y[hist.ids.index(i)] for i in ids])
        svc = {it["id"]: it["fc"] for it in res["items"]}
        f = {"service (selector)": np.stack([svc[i] for i in ids]), "seasonal naive": B.seasonal_naive(hy), "28-day average": B.moving_average(hy), "weekday average": B.weekday_mean(hy)}
        for m, fc in f.items():
            rows.append(dict(dataset=name, window=k, method=m, weekly=M.wape(M.block_sums(actual), M.block_sums(fc))))
        log.append(f"{name} window {k} (cut {cut.date()}): chose {res['method']['method']} ({res['method']['reason']}), backtest {res['method']['backtest']}")
r = pd.DataFrame(rows)
tab = r.groupby(["method", "dataset"]).weekly.mean().unstack()[list(webapi.EXAMPLES)].round(3)
base = tab.loc[["seasonal naive", "28-day average", "weekday average"]].min()
out = ["\n".join(log), "", "Weekly WAPE, mean of windows", tab.to_string(), "", "service vs best simple baseline: " + ", ".join(f"{d} {tab.loc['service (selector)', d] / base[d] - 1:+.1%}" for d in tab.columns)]
out.append("rule (all within +3%): " + ("PASSED" if (tab.loc["service (selector)"] / base - 1 <= 0.03).all() else "NOT PASSED"))
text = "\n".join(out); print(text)
(C.REPORTS / "selector_check.txt").write_text(text, encoding="utf-8")
