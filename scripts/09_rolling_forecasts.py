"""Weekly refreshed forecasts: the default LightGBM of each fold (fixed at its cutoff) predicts again from the origins
cutoff + 7k, k = 0..3, with features updated by the demand observed in between. This is how a shop would run it:
the model is not retrained, only its inputs are refreshed. Origins stay below d1941, so the test fold is untouched.
"""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demand import config as C  # noqa: E402
from demand.evaluate import load_matrices, save_forecasts  # noqa: E402
from demand.features import CATEGORICAL, FEATURES, build_rows, training_rows  # noqa: E402

COMMON = dict(learning_rate=0.05, n_estimators=400, num_leaves=63, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=C.SEED, verbose=-1, n_jobs=-1)


def main() -> None:
    m = load_matrices()
    n = len(m.ids)
    store = {}
    for cutoff in C.DEV_CUTOFFS + [C.VAL_CUTOFF]:
        t0 = time.time()
        train = training_rows(m, cutoff)
        models = {
            "mean": lgb.LGBMRegressor(**COMMON, objective="tweedie", tweedie_variance_power=1.1),
            "q90": lgb.LGBMRegressor(**COMMON, objective="quantile", alpha=0.9),
        }
        for model in models.values():
            model.fit(train[FEATURES], train.y, categorical_feature=CATEGORICAL)
        for k in range(4):
            origin = cutoff + 7 * k
            assert origin + C.HORIZON <= C.TEST_CUTOFF + C.HORIZON and origin <= C.VAL_CUTOFF + 21
            rows = build_rows(m, origin, with_target=False)
            on_sale = rows.on_sale.to_numpy().reshape(n, C.HORIZON)
            for name, model in models.items():
                p = np.clip(model.predict(rows[FEATURES]), 0, None).reshape(n, C.HORIZON) * on_sale
                store[f"{cutoff}|roll{k}_{name}"] = p
        print(f"fold {cutoff} done in {time.time() - t0:.0f}s", flush=True)
    save_forecasts("rolling", store)


if __name__ == "__main__":
    main()
