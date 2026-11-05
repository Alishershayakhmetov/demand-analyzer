"""Random-search tuning of the LightGBM Tweedie model on the three dev folds only.

Selection rule, fixed before running: lowest mean weekly WAPE over the dev folds (ordering is done weekly).
The validation fold (d1913) is not used for selection. After selection, default and tuned models are refit on every
fold (Tweedie + q50 + q90) and their forecasts stored for the report.
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
from demand.evaluate import load_matrices, save_forecasts  # noqa: E402
from demand.features import CATEGORICAL, FEATURES, build_rows, training_rows  # noqa: E402

N_TRIALS = 16
ITERS = [200, 400, 600, 800]
BASE = dict(learning_rate=0.05, subsample=0.8, subsample_freq=1, random_state=C.SEED, verbose=-1, n_jobs=-1)
DEFAULT = dict(num_leaves=63, min_child_samples=50, colsample_bytree=0.8, reg_lambda=0.0, tweedie_variance_power=1.1)
SPACE = dict(num_leaves=[15, 31, 63, 127], min_child_samples=[20, 50, 100, 200, 400],
             colsample_bytree=[0.5, 0.7, 0.9], reg_lambda=[0.0, 1.0, 10.0, 30.0],
             tweedie_variance_power=[1.05, 1.1, 1.2, 1.3, 1.5])


def fold_data(m, cutoff):
    train, test = training_rows(m, cutoff), build_rows(m, cutoff)
    n = len(m.ids)
    return dict(train=train, test=test, n=n, actual=test.y.to_numpy().reshape(n, C.HORIZON).astype(float),
                on_sale=test.on_sale.to_numpy().reshape(n, C.HORIZON))


def predict(model, d, num_iteration=None):
    p = model.predict(d["test"][FEATURES], num_iteration=num_iteration)
    return np.clip(p, 0, None).reshape(d["n"], C.HORIZON) * d["on_sale"]


def fit(d, params, n_estimators, objective, **extra):
    model = lgb.LGBMRegressor(**BASE, **params, n_estimators=n_estimators, objective=objective, **extra)
    model.fit(d["train"][FEATURES], d["train"].y, categorical_feature=CATEGORICAL)
    return model


def main() -> None:
    m = load_matrices()
    rng = np.random.default_rng(C.SEED)
    trials = [DEFAULT] + [{k: v[rng.integers(len(v))] for k, v in SPACE.items()} for _ in range(N_TRIALS)]

    data = {c: fold_data(m, c) for c in C.DEV_CUTOFFS}
    log = []
    for t, params in enumerate(trials):
        t0 = time.time()
        for cutoff, d in data.items():
            model = fit(d, params, max(ITERS), "tweedie")
            for k in ITERS:
                f = predict(model, d, k)
                log.append(dict(trial=t, n_iter=k, cutoff=cutoff, **params,
                                daily_wape=M.wape(d["actual"], f),
                                weekly_wape=M.wape(M.block_sums(d["actual"]), M.block_sums(f))))
        print(f"trial {t} {params} {time.time() - t0:.0f}s", flush=True)
    log = pd.DataFrame(log)
    log.to_csv(C.REPORTS / "tuning_log.csv", index=False)

    keys = ["trial", "n_iter", *DEFAULT]
    agg = log.groupby(keys, as_index=False)[["daily_wape", "weekly_wape"]].mean().sort_values("weekly_wape")
    print("\nTop 8 by mean weekly WAPE on dev folds:")
    print(agg.head(8).round(4).to_string(index=False))
    default_row = agg[(agg.trial == 0) & (agg.n_iter == 400)].iloc[0]
    print(f"\nprevious default (400 trees) on dev folds: daily {default_row.daily_wape:.4f}, weekly {default_row.weekly_wape:.4f}")
    best = agg.iloc[0]
    best_params = {k: (int(best[k]) if k in ("num_leaves", "min_child_samples") else float(best[k])) for k in DEFAULT}
    best_iter = int(best.n_iter)
    print("\nselected:", best_params, "trees:", best_iter)

    configs = {"default": (DEFAULT, 400), "tuned": (best_params, best_iter)}
    store = {}
    for cutoff in C.DEV_CUTOFFS + [C.VAL_CUTOFF]:
        d = data.get(cutoff) or fold_data(m, cutoff)
        store[f"{cutoff}|actual"] = d["actual"]
        for label, (params, n_trees) in configs.items():
            store[f"{cutoff}|lgbm_{label}"] = predict(fit(d, params, n_trees, "tweedie"), d)
            shape = {k: v for k, v in params.items() if k != "tweedie_variance_power"}
            for alpha in (0.5, 0.9):
                store[f"{cutoff}|lgbm_{label}_q{int(alpha * 100)}"] = predict(fit(d, shape, n_trees, "quantile", alpha=alpha), d)
        print(f"final fit fold {cutoff} done", flush=True)
    save_forecasts("lgbm", store)


if __name__ == "__main__":
    main()
