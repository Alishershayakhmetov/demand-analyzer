"""Evaluate baseline forecasters on the dev and validation folds (the test fold is not evaluated here)."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demand import baselines as B  # noqa: E402
from demand import config as C  # noqa: E402
from demand import metrics as M  # noqa: E402
from demand.evaluate import score  # noqa: E402

FOLDS = C.DEV_CUTOFFS + [C.VAL_CUTOFF]


def main() -> None:
    panel = pd.read_parquet(C.PROC / f"panel_{C.STORE}.parquet")
    items = pd.read_csv(C.PROC / "items.csv").sort_values("item_id").reset_index(drop=True)
    ids = items.item_id.tolist()
    Y = panel.pivot(index="item_id", columns="d", values="sales").loc[ids].to_numpy()
    ON = panel.pivot(index="item_id", columns="d", values="on_sale").loc[ids].to_numpy()
    first_day = items.first_d.to_numpy().astype(int)

    rows = []
    for cutoff in FOLDS:
        t0 = time.time()
        hist = Y[:, :cutoff]
        actual = Y[:, cutoff:cutoff + C.HORIZON].astype(float)
        on_sale = ON[:, cutoff:cutoff + C.HORIZON]
        forecasts = {
            "naive": B.naive(hist),
            "seasonal_naive": B.seasonal_naive(hist),
            "moving_avg_28": B.moving_average(hist),
            "weekday_mean_8w": B.weekday_mean(hist),
            **B.statsforecast_models(hist, first_day, ids),
        }
        quantiles = {0.5: B.weekday_quantile(hist, 0.5), 0.9: B.weekday_quantile(hist, 0.9)}
        for name, f in forecasts.items():
            f = np.clip(f, 0, None) * on_sale  # price is known ahead; an item with no price that week is not sold
            rows.append(dict(cutoff=cutoff, model=name, **score(actual, f, hist, first_day)))
        print(f"fold {cutoff} done in {time.time() - t0:.0f}s")
        for q, f in quantiles.items():
            f = f * on_sale
            rows.append(dict(cutoff=cutoff, model=f"weekday_q{int(q * 100)}_8w", pinball=np.nanmean(
                [M.pinball(actual[i], f[i], q) for i in range(len(ids))]), coverage=M.coverage(actual, f)))
    res = pd.DataFrame(rows)
    res.to_csv(C.REPORTS / "baselines_by_fold.csv", index=False)

    point = res.dropna(subset=["wape"]).groupby("model", sort=False)[["wape", "bias", "rmsse"]].mean().round(4)
    point = point.join(res[res.cutoff == C.VAL_CUTOFF].set_index("model")[["wape"]].rename(columns={"wape": "wape_val"}).round(4))
    print("\nPoint forecasts (mean over 4 folds; last column = validation fold only):")
    print(point.sort_values("wape").to_string())
    point.sort_values("wape").to_csv(C.REPORTS / "baselines_summary.csv")

    q = res.dropna(subset=["pinball"]).groupby("model")[["pinball", "coverage"]].mean().round(4)
    print("\nQuantile baselines (mean over folds):")
    print(q.to_string())


if __name__ == "__main__":
    main()
