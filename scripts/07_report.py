"""Daily, weekly and 28-day metrics for every stored forecast; quantile calibration. Reads data/processed/forecasts_*.npz."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demand import config as C  # noqa: E402
from demand import metrics as M  # noqa: E402
from demand.evaluate import load_forecasts  # noqa: E402

QUANTILE = {"q50": 0.5, "q90": 0.9}


def main() -> None:
    fc = load_forecasts("baselines", "lgbm")
    point, quant = [], []
    for cutoff, models in sorted(fc.items()):
        actual = models["actual"]
        for name, f in models.items():
            if name == "actual":
                continue
            suffix = next((s for s in QUANTILE if name.endswith(f"_{s}") or name.endswith(f"_{s}_8w")), None)
            if suffix:
                q = QUANTILE[suffix]
                quant.append(dict(cutoff=cutoff, model=name, pinball=np.nanmean([M.pinball(actual[i], f[i], q) for i in range(len(f))]),
                                  coverage=M.coverage(actual, f)))
            else:
                point.append(dict(cutoff=cutoff, model=name, daily_wape=M.wape(actual, f),
                                  weekly_wape=M.wape(M.block_sums(actual), M.block_sums(f)),
                                  wape_28d=M.wape(M.block_sums(actual, 28), M.block_sums(f, 28)),
                                  weekly_bias=M.bias(M.block_sums(actual), M.block_sums(f))))
    point, quant = pd.DataFrame(point), pd.DataFrame(quant)
    point.to_csv(C.REPORTS / "results_by_fold.csv", index=False)

    val = point[point.cutoff == C.VAL_CUTOFF].set_index("model")[["daily_wape", "weekly_wape", "wape_28d"]].add_prefix("val_")
    mean = point.groupby("model")[["daily_wape", "weekly_wape", "wape_28d", "weekly_bias"]].mean().join(val)
    mean = mean.sort_values("weekly_wape").round(4)
    mean.to_csv(C.REPORTS / "results_summary.csv")
    print("Point forecasts: mean over 4 folds (dev folds were used for tuning), val_* = validation fold d1913 only")
    print(mean.to_string())

    q = quant.groupby("model")[["pinball", "coverage"]].mean().join(
        quant[quant.cutoff == C.VAL_CUTOFF].set_index("model")[["pinball", "coverage"]].add_prefix("val_")).round(4)
    q.to_csv(C.REPORTS / "results_quantiles.csv")
    print("\nQuantile forecasts:")
    print(q.to_string())


if __name__ == "__main__":
    main()
