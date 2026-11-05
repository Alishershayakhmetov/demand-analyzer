"""SHAP values of the default LightGBM (Tweedie) on the validation-fold forecast rows (origin d1913).

Contributions come from LightGBM's own `pred_contrib` (exact tree SHAP, valid for categorical splits); they are in the
log scale of the Tweedie link, so exp(shap) is a multiplicative effect on the predicted demand.
"""
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demand import config as C  # noqa: E402
from demand.evaluate import load_matrices  # noqa: E402
from demand.features import CATEGORICAL, FEATURES, build_rows, training_rows  # noqa: E402

GROUPS = {
    "recent demand level": ["mean7", "mean14", "mean28", "mean56", "mean91", "std28", "nz28", "nz91", "days_since_sale", "age"],
    "weekday history": ["wd_mean8", "wd_mean4", "wd_last"],
    "price": ["price", "price_vs_median", "price_vs_origin"],
    "calendar": ["wday", "month", "dom"],
    "SNAP": ["snap"],
    "events": ["event"],
    "horizon": ["h"],
    "item type": ["dept", "cat"],
}
COMMON = dict(learning_rate=0.05, n_estimators=400, num_leaves=63, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=C.SEED, verbose=-1, n_jobs=-1,
              objective="tweedie", tweedie_variance_power=1.1)


def main() -> None:
    m = load_matrices()
    cutoff = C.VAL_CUTOFF
    train, rows = training_rows(m, cutoff), build_rows(m, cutoff)
    model = lgb.LGBMRegressor(**COMMON).fit(train[FEATURES], train.y, categorical_feature=CATEGORICAL)
    X = rows[FEATURES]
    contrib = model.predict(X, pred_contrib=True)
    sv, base = contrib[:, :-1], contrib[:, -1]
    assert np.allclose(sv.sum(axis=1) + base, model.predict(X, raw_score=True), atol=1e-6)

    out = []

    def say(s: str = "") -> None:
        print(s)
        out.append(s)

    imp = pd.Series(np.abs(sv).mean(axis=0), index=FEATURES).sort_values(ascending=False)
    say(f"mean |SHAP| (log scale) on {len(X):,} forecast rows of fold {cutoff}; base value exp({base[0]:.2f}) = {np.exp(base[0]):.2f} units")
    say(imp.round(3).to_string())
    group_imp = pd.Series({g: np.abs(sv[:, [FEATURES.index(f) for f in fs]].sum(axis=1)).mean() for g, fs in GROUPS.items()}).sort_values(ascending=False)
    say("\nmean |sum of SHAP| per feature group:")
    say((group_imp / group_imp.sum()).round(3).to_string())

    say("\nmean SHAP by weekday (1=Sat), exp = multiplicative effect: " + ", ".join(
        f"{w}: {np.exp(sv[rows.wday == w, FEATURES.index('wday')].mean()):.2f}" for w in range(1, 8)))
    snap_i = FEATURES.index("snap")
    say(f"SNAP day effect {np.exp(sv[rows.snap == 1, snap_i].mean()):.3f}x vs non-SNAP {np.exp(sv[rows.snap == 0, snap_i].mean()):.3f}x")
    say("event type effect: " + ", ".join(f"{k}: {np.exp(sv[rows.event == k, FEATURES.index('event')].mean()):.3f}x" for k in sorted(rows.event.unique())))
    pr = rows.price_vs_median
    for lo, hi, label in [(0, 0.9, "price < 0.9 of median"), (0.9, 0.98, "0.9-0.98"), (0.98, 1.02, "about median"), (1.02, 10, "> 1.02")]:
        mask = (pr >= lo) & (pr < hi)
        if mask.sum():
            say(f"price_vs_median {label}: {mask.sum()} rows, price-group SHAP {np.exp(sv[mask][:, [FEATURES.index(f) for f in GROUPS['price']]].sum(axis=1).mean()):.3f}x")
    say("horizon effect by step: " + ", ".join(f"{h}: {np.exp(sv[rows.h == h, FEATURES.index('h')].mean()):.2f}" for h in (1, 7, 14, 21, 28)))
    (C.REPORTS / "shap_summary.txt").write_text("\n".join(out), encoding="utf-8")

    fig = plt.figure(figsize=(9, 7))
    shap.summary_plot(sv, X, max_display=15, show=False, plot_size=None)
    plt.title("SHAP values (log scale), forecast rows of fold d1913")
    plt.tight_layout()
    fig.savefig(C.REPORTS / "figures" / "shap_beeswarm.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    (group_imp / group_imp.sum()).sort_values().plot.barh(ax=ax[0])
    ax[0].set_title("share of mean |SHAP| by feature group")
    ax[1].scatter(rows.price_vs_median.clip(0.6, 1.4), sv[:, [FEATURES.index(f) for f in GROUPS["price"]]].sum(axis=1), s=3, alpha=0.4)
    ax[1].set_xlabel("price / 91-day median")
    ax[1].set_ylabel("price-group SHAP")
    ax[1].set_title("price effect")
    ax[2].scatter(rows.h + np.random.default_rng(0).uniform(-0.3, 0.3, len(rows)), sv[:, FEATURES.index("h")], s=3, alpha=0.3)
    ax[2].set_xlabel("horizon step")
    ax[2].set_ylabel("horizon SHAP")
    ax[2].set_title("horizon effect")
    fig.tight_layout()
    fig.savefig(C.REPORTS / "figures" / "shap_groups.png", dpi=110)


if __name__ == "__main__":
    main()
