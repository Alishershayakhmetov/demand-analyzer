"""Global LightGBM (direct multi-horizon) vs the baselines on the dev and validation folds. Test fold not used."""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demand import config as C  # noqa: E402
from demand import metrics as M  # noqa: E402
from demand.evaluate import load_matrices, score  # noqa: E402
from demand.features import CATEGORICAL, FEATURES, build_rows, training_rows  # noqa: E402

FOLDS = C.DEV_CUTOFFS + [C.VAL_CUTOFF]
COMMON = dict(learning_rate=0.05, n_estimators=400, num_leaves=63, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=C.SEED, verbose=-1, n_jobs=-1)
MODELS = {
    "lgbm_tweedie": dict(objective="tweedie", tweedie_variance_power=1.1),
    "lgbm_q50": dict(objective="quantile", alpha=0.5),
    "lgbm_q90": dict(objective="quantile", alpha=0.9),
}


def main() -> None:
    m = load_matrices()
    rows = []
    for cutoff in FOLDS:
        t0 = time.time()
        train = training_rows(m, cutoff)
        test = build_rows(m, cutoff)
        n = len(m.ids)
        actual = test.y.to_numpy().reshape(n, C.HORIZON).astype(float)
        on_sale = test.on_sale.to_numpy().reshape(n, C.HORIZON)
        hist = m.Y[:, :cutoff]
        for name, params in MODELS.items():
            model = lgb.LGBMRegressor(**COMMON, **params)
            model.fit(train[FEATURES], train.y, categorical_feature=CATEGORICAL)
            f = np.clip(model.predict(test[FEATURES]), 0, None).reshape(n, C.HORIZON) * on_sale
            if name == "lgbm_tweedie":
                rows.append(dict(cutoff=cutoff, model=name, **score(actual, f, hist, m.first_day)))
            else:
                q = params["alpha"]
                rows.append(dict(cutoff=cutoff, model=name, coverage=M.coverage(actual, f),
                                 pinball=np.nanmean([M.pinball(actual[i], f[i], q) for i in range(n)])))
        print(f"fold {cutoff}: {len(train):,} train rows, {time.time() - t0:.0f}s", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(C.REPORTS / "lgbm_by_fold.csv", index=False)

    base = pd.read_csv(C.REPORTS / "baselines_by_fold.csv")
    allres = pd.concat([base, res], ignore_index=True)
    point = allres.dropna(subset=["wape"]).groupby("model", sort=False)[["wape", "bias", "rmsse"]].mean().round(4)
    print("\nPoint forecasts, mean over 4 folds:")
    print(point.sort_values("wape").to_string())
    q = allres.dropna(subset=["pinball"]).groupby("model")[["pinball", "coverage"]].mean().round(4)
    print("\nQuantile forecasts, mean over 4 folds:")
    print(q.to_string())
    print("\nWAPE by fold:")
    print(allres.dropna(subset=["wape"]).pivot(index="model", columns="cutoff", values="wape").round(4).to_string())


if __name__ == "__main__":
    main()
