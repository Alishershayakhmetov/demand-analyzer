"""Do calendar and price features help once the model is tuned? Dev folds only, then one look at d1913. Test fold not used.

Question: the shipped 11-feature model has no calendar, price or trend input, so its weekly forecast is almost flat.
Earlier single-group tests used default hyper-parameters and found no gain. Here each feature set gets the same tuning budget.

Declared before running:
- sets: A = shipped lean set (11 features); B = A + calendar (snap, events, month, day of month, weekday of target day);
  C = all 24 features (B + price, weekday history, item type);
- the same 13 configurations (default + 12 random, seed 42) x 3 dev folds x {200, 400, 600} trees for every set;
- criterion: mean weekly WAPE over the dev folds d1829, d1857, d1885; best (configuration, trees) per set;
- B or C replaces A only if its best dev weekly WAPE is lower than A's best by more than 0.0029 (the earlier noise threshold)
  AND on the validation fold d1913 (3 seeds) its weekly WAPE is not above that of the shipped model;
- reported besides accuracy: how much the 4-week forecast varies from week to week, against how much real sales vary.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demand import config as C  # noqa: E402
from demand import metrics as M  # noqa: E402
from demand.evaluate import load_matrices  # noqa: E402
from demand.features import FEATURES, build_rows, training_rows  # noqa: E402
from demand.modeling import fit, inventory_days, predict_origins, window_metrics  # noqa: E402

DEV, VAL = C.DEV_CUTOFFS, C.VAL_CUTOFF
LEVEL = ["mean7", "mean14", "mean28", "mean56", "mean91", "std28", "nz28", "nz91", "days_since_sale", "age"]
CAL = ["snap", "event", "month", "dom", "wday"]
SETS = {"A lean (11)": [*LEVEL, "h"], "B lean + calendar (16)": [*LEVEL, "h", *CAL], "C all 24": list(FEATURES)}
SHIPPED = dict(num_leaves=63, min_child_samples=50, colsample_bytree=0.8, reg_lambda=0.0, tweedie_variance_power=1.1)
SPACE = dict(num_leaves=[7, 15, 31, 63, 127], min_child_samples=[20, 50, 100, 200, 400, 800], colsample_bytree=[0.5, 0.7, 0.9],
             reg_lambda=[0.0, 1.0, 10.0, 30.0], tweedie_variance_power=[1.1, 1.2, 1.3])
ITERS = [200, 400, 600]
THRESHOLD = 0.0029
m = load_matrices()
N = len(m.ids)


def ordered(fs) -> list[str]:
    return [f for f in FEATURES if f in set(fs)]


def flatness(f: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """Median |week2 / week1 - 1| of the forecast and of real sales, over items with some volume."""
    w, a = M.block_sums(f), M.block_sums(actual)
    keep = w.sum(axis=1) > 2
    return (float(np.median(np.abs(w[keep, 1] / w[keep, 0] - 1))),
            float(np.median(np.abs(a[keep, 1] / np.maximum(a[keep, 0], 1) - 1))))


def main() -> None:
    t0 = time.time()
    out = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        out.append(s)

    rng = np.random.default_rng(C.SEED)
    configs = [SHIPPED] + [{k: v[rng.integers(len(v))] for k, v in SPACE.items()} for _ in range(12)]
    train = {c: training_rows(m, c) for c in [*DEV, VAL]}
    rows0 = {c: build_rows(m, c, with_target=False) for c in [*DEV, VAL]}
    actual = {c: m.Y[:, c:c + C.HORIZON].astype(float) for c in [*DEV, VAL]}

    log = []
    for name, feats in SETS.items():
        fs = ordered(feats)
        for ci, cfg in enumerate(configs):
            for c in DEV:
                model = fit(train[c], fs, C.SEED, n_estimators=max(ITERS), **cfg)
                for k in ITERS:
                    f = predict_origins(model, [rows0[c]], fs, N, k)[0]
                    log.append(dict(set=name, config=ci, iters=k, cutoff=c, **window_metrics(m, c, f),
                                    flat_fc=flatness(f, actual[c])[0], flat_real=flatness(f, actual[c])[1]))
            print(f"{name} config {ci} done ({time.time() - t0:.0f}s)", flush=True)
    lg = pd.DataFrame(log)
    lg.to_csv(C.REPORTS / "calendar_dev_log.csv", index=False)
    agg = lg.groupby(["set", "config", "iters"], as_index=False)[["daily_wape", "weekly_wape", "wape_28d", "flat_fc", "flat_real"]].mean()

    say("Dev folds only. Best (configuration, trees) per feature set by mean weekly WAPE; the shipped configuration (config 0, 400 trees) for reference\n")
    best = {}
    for name in SETS:
        a = agg[agg.set == name].sort_values("weekly_wape")
        best[name] = a.iloc[0]
        ship = a[(a.config == 0) & (a.iters == 400)].iloc[0]
        say(f"{name}: best config {int(a.iloc[0].config)} / {int(a.iloc[0].iters)} trees weekly {a.iloc[0].weekly_wape:.4f} daily {a.iloc[0].daily_wape:.4f} "
            f"28d {a.iloc[0].wape_28d:.4f}; shipped configuration weekly {ship.weekly_wape:.4f}; "
            f"forecast week-to-week change {a.iloc[0].flat_fc:.1%} vs real {a.iloc[0].flat_real:.1%}")
        say(f"    params {configs[int(a.iloc[0].config)]}")
    base = best["A lean (11)"].weekly_wape
    say(f"\nthreshold {THRESHOLD}")
    passed = {}
    for name in SETS:
        if name == "A lean (11)":
            continue
        gain = base - best[name].weekly_wape
        passed[name] = gain > THRESHOLD
        say(f"{name}: best dev weekly {best[name].weekly_wape:.4f} vs A {base:.4f} -> gain {gain:+.4f} -> {'passes' if passed[name] else 'does not pass'} the dev rule")

    say("\nValidation fold d1913, 3 seeds each (first look at this fold for these configurations)")
    origins = [build_rows(m, VAL + 7 * k, with_target=False) for k in range(4)]
    contenders = {"shipped lean model (default config)": (SETS["A lean (11)"], SHIPPED, 400)}
    for name, b in best.items():
        contenders[f"{name}, best dev config"] = (SETS[name], configs[int(b.config)], int(b.iters))
    rows = []
    for label, (feats, cfg, iters) in contenders.items():
        fs = ordered(feats)
        res, inv, flat = [], [], []
        for seed in (42, 1, 2):
            model = fit(train[VAL], fs, seed, n_estimators=iters, **cfg)
            rolls = predict_origins(model, origins, fs, N)
            res.append(window_metrics(m, VAL, rolls[0]))
            inv.append(inventory_days(m, {VAL: rolls}, [VAL]))
            flat.append(flatness(rolls[0], actual[VAL])[0])
        rows.append(dict(model=label, **{k: np.mean([r[k] for r in res]) for k in res[0]},
                         weekly_std=np.std([r["weekly_wape"] for r in res], ddof=1), inv_days_95=np.mean(inv), forecast_week_change=np.mean(flat)))
    vdf = pd.DataFrame(rows).set_index("model")
    vdf.round(4).to_csv(C.REPORTS / "calendar_val_results.csv")
    say(vdf.round(4).to_string())
    ship_val = vdf.loc["shipped lean model (default config)", "weekly_wape"]
    say("\nVerdict")
    for name in SETS:
        if name == "A lean (11)":
            continue
        v = vdf.loc[f"{name}, best dev config", "weekly_wape"]
        ok = passed[name] and v <= ship_val
        say(f"{name}: dev rule {'passed' if passed[name] else 'not passed'}; validation weekly {v:.4f} vs shipped {ship_val:.4f} -> {'ADOPT' if ok else 'DO NOT ADOPT'}")
    say(f"\nruntime {time.time() - t0:.0f}s")
    (C.REPORTS / "calendar_dev.txt").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
