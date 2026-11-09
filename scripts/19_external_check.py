"""Does the service forecast data it was never trained on? Hold-out check on the example files from scripts/18_external_examples.py.

For each file: the CSV is read exactly like the app reads an upload, history is cut 28 days before the end of the data (on a Sunday),
the service (`webapi.analyze`) forecasts the next 28 days, and the forecast is scored against what really happened next to simple
baselines computed on the same history. Nothing here is tuned; the model was trained on M5 only.
"""
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

EX = C.ROOT / "examples"
CASES = webapi.EXAMPLES


def train_scale() -> tuple[float, float]:
    """Typical and extreme 28-day mean daily sales of an item in the M5 subset the model was trained on."""
    p = pd.read_parquet(C.PROC / f"panel_{C.STORE}.parquet", columns=["item_id", "d", "sales"]).sort_values(["item_id", "d"])
    m28 = p.groupby("item_id").sales.transform(lambda s: s.rolling(28).mean())
    return float(m28.quantile(0.5)), float(m28.quantile(0.999))


def score(actual: np.ndarray, f: np.ndarray) -> dict:
    return dict(daily=M.wape(actual, f), weekly=M.wape(M.block_sums(actual), M.block_sums(f)), d28=M.wape(M.block_sums(actual, 28), M.block_sums(f, 28)),
                bias=M.bias(actual, f))


def main() -> None:
    med, top = train_scale()
    out = [f"scale of the training items (28-day mean units per day): median {med:.2f}, 99.9th percentile {top:.1f}\n"]
    rows = []
    for name, c in CASES.items():
        path = EX / f"{name}.csv"
        if not path.exists():
            out.append(f"{name}: file missing, run scripts/18_external_examples.py")
            continue
        df = webapi.read_csv_text(path.read_text(encoding="utf-8"))
        cols = dict(date_col=c["date"], item_col=c["item"], sales_col=c["sales"], price_col=c["price"])
        full = S.to_shop(df, **cols)
        cutoff = full.dates[-1] - pd.Timedelta(days=28)
        assert cutoff.weekday() == 6
        d = pd.to_datetime(df[c["date"]], errors="coerce").dt.normalize()
        hist_df = df[d <= cutoff]
        hist = S.to_shop(hist_df, **cols)
        payload_cols = {k: c[k] for k in ("date", "item", "sales")}
        if c["price"]:
            payload_cols["price"] = c["price"]
        res = webapi.analyze(dict(source="csv", csv=hist_df.to_csv(index=False), cols=payload_cols))
        service = {it["id"]: it for it in res["items"]}
        ids = [i for i in hist.ids if i in set(full.ids)]
        fi = {i: k for k, i in enumerate(full.ids)}
        actual = np.stack([full.Y[fi[i], -28:] for i in ids])
        hy = np.stack([hist.Y[hist.ids.index(i)] for i in ids])
        f_service = np.stack([service[i]["fc"] for i in ids])
        methods = {"service (model trained on M5 + range guard)": f_service, "seasonal naive": B.seasonal_naive(hy), "28-day average": B.moving_average(hy),
                   "weekday average, 8 weeks": B.weekday_mean(hy)}
        plan = webapi.plan(res["token"], 3, 95)
        recent = hy[:, -28:].mean(axis=1)
        out.append(f"== {name}: {c['what']}")
        out.append(f"history to {cutoff.date()} ({hist.Y.shape[1]} days), forecast window {(cutoff + pd.Timedelta(days=1)).date()}..{full.dates[-1].date()}, {len(ids)} items, "
                   f"median recent level {np.median(recent):.2f}/day, {np.mean(recent > top):.0%} of items above the training maximum; share of zero days in the window {np.mean(actual == 0):.0%}")
        out.append(f"plan check (lead 3 d, 95%): {plan['n_order']} items to order, {plan['units']:,} units; {res['n_range']} items above the model's demand range use the weekday average; notes: {res['notes'] or 'none'}")
        for k, f in methods.items():
            s = score(actual, f)
            rows.append(dict(dataset=name, method=k, items=len(ids), **s))
            out.append(f"  {k:36s} daily WAPE {s['daily']:.3f}  weekly {s['weekly']:.3f}  28-day {s['d28']:.3f}  bias {s['bias']:+.1%}")
        out.append("")
    pd.DataFrame(rows).round(4).to_csv(C.REPORTS / "external_check.csv", index=False)
    text = "\n".join(out)
    print(text)
    (C.REPORTS / "external_check.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
