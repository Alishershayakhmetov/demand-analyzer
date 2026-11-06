"""One-time scoring of the untouched test fold (forecast origin d1941, window d1942..d1969).

Usage: python scripts/15_test_fold.py            -> real test run; refuses to run twice
       python scripts/15_test_fold.py 1913       -> dry run of the same code on the validation fold (plumbing check only)

Declared before the run (the test fold had not been scored by any script):
- Model under test: the stored final model models/lean_model.txt (selected 11 features, trained on days <= d1941, its
  SHA-256 is checked against models/meta.json). Nothing is tuned or selected on this fold.
- Headline: weekly WAPE of the final model vs the comparators; inventory days for 95% revenue-weighted fill, lead time 3 days,
  weekly refresh, same simulator and dormant-item rule as for the other folds.
- H1: weekly WAPE of the final model is at least 10% below seasonal naive.
- H2: weekly WAPE of the final model is not more than 3% above the 28-day average.
- H3: at 95% fill and lead time 3 the final model needs at least 10% less inventory than the owner rule (last 7 days) and
  less inventory than the 28-day average, both with a paired-bootstrap 90% interval that excludes zero.
Results are reported whatever they are. A crash is fixed and the run repeated; a result is never a reason to change anything.
Forecast origins of the weekly refresh go up to d1962, whose protection interval passes d1969: prices beyond the data are
carried forward from the last day (affects only the on-sale mask of those days).
"""
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demand import baselines as B  # noqa: E402
from demand import config as C  # noqa: E402
from demand import metrics as M  # noqa: E402
from demand.evaluate import load_matrices  # noqa: E402
from demand.features import FEATURES, build_rows, training_rows  # noqa: E402
from demand.inventory import inventory_at_fill  # noqa: E402
from demand.modeling import Z, fit, inventory_by_item, predict_origins, window_metrics  # noqa: E402

CUTOFF = int(sys.argv[1]) if len(sys.argv) > 1 else C.TEST_CUTOFF
REAL = CUTOFF == C.TEST_CUTOFF
OUT = C.REPORTS / ("test_results.txt" if REAL else f"dryrun_{CUTOFF}.txt")
LEADS, TARGETS, N_BOOT = [1, 3, 7], [0.90, 0.95, 0.98], 1000
POLICY_NAMES = {"owner": "owner: last 7 days", "ma28": "28-day average", "wd8": "weekday average, 8 weeks",
                "full24": "LightGBM full 24 features", "lean": "LightGBM selected 11 features (final)"}


def pad(m, extra: int = 35):
    """Extend the day axis so origins close to the end of the data can build rows; prices are carried forward."""
    ext = lambda a: np.concatenate([a, np.repeat(a[..., -1:], extra, axis=-1)], axis=-1)  # noqa: E731
    return dataclasses.replace(m, PR=ext(m.PR), snap=ext(m.snap), wday=ext(m.wday), month=ext(m.month), dom=ext(m.dom), event=ext(m.event))


def main() -> None:
    if REAL and OUT.exists():
        sys.exit(f"{OUT.name} exists: the test fold has been scored; it is opened once")
    t0 = time.time()
    out: list[str] = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        out.append(s)

    m = load_matrices()
    mp = pad(m)
    n, H, c = len(m.ids), C.HORIZON, CUTOFF
    hist, actual = m.Y[:, :c], m.Y[:, c:c + H].astype(float)
    on_sale = m.on_sale[:, c:c + H]
    feats = json.loads((C.REPORTS / "selected_features.json").read_text())["features"]
    feats = [f for f in FEATURES if f in feats]
    say(f"scoring forecast origin d{c} ({'TEST FOLD' if REAL else 'dry run, not the test fold'}), window d{c + 1}..d{c + H}, {n} items")

    train = training_rows(m, c)
    origins = [build_rows(mp, c + 7 * k, with_target=False) for k in range(4)]
    if REAL:
        meta = json.loads((C.ROOT / "models" / "meta.json").read_text())
        sha = hashlib.sha256((C.ROOT / "models" / "lean_model.txt").read_bytes()).hexdigest()
        assert sha == meta["sha256"], "model file differs from the one recorded in meta.json"
        assert meta["trained_through_day"] == C.TEST_CUTOFF and meta["features"] == feats
        lean = lgb.Booster(model_file=str(C.ROOT / "models" / "lean_model.txt"))
        say(f"model SHA-256 {sha[:16]}... matches models/meta.json")
    else:
        lean = fit(train, feats, C.SEED).booster_
    lean_rolls = predict_origins(lean, origins, feats, n)
    if REAL:
        pre = pd.read_csv(C.REPORTS / "test_window_forecast.csv", index_col=0)
        assert list(pre.index) == m.ids
        say(f"stored pre-test forecast vs this run (masked by price availability): max abs diff "
            f"{np.abs(pre.to_numpy() * on_sale - lean_rolls[0]).max():.2e}")
    full = {s: predict_origins(fit(train, FEATURES, s).booster_, origins, FEATURES, n) for s in (42, 1, 2)}
    lean_seeds = {s: predict_origins(fit(train, feats, s).booster_, origins, feats, n) for s in (1, 2)}

    # point forecasts at the origin
    point = {"naive": B.naive(hist), "seasonal naive": B.seasonal_naive(hist), "28-day average": B.moving_average(hist),
             "weekday average, 8 weeks": B.weekday_mean(hist), **B.statsforecast_models(hist, m.first_day, m.ids)}
    point = {k: np.clip(v, 0, None) * on_sale for k, v in point.items()}
    point["LightGBM full 24 (3 seeds avg)"] = np.mean([full[s][0] for s in full], axis=0)
    point["LightGBM selected 11 (seeds 42, 1, 2 avg)"] = np.mean([lean_rolls[0], lean_seeds[1][0], lean_seeds[2][0]], axis=0)
    point["LightGBM selected 11 (FINAL, stored model)"] = lean_rolls[0]
    rows = []
    for name, f in point.items():
        w = window_metrics(m, c, f)
        rows.append(dict(model=name, **w, weekly_bias=M.bias(M.block_sums(actual), M.block_sums(f)),
                         daily_rmsse=float(np.nanmean([M.rmsse(actual[i], f[i], hist[i, max(m.first_day[i] - 1, 0):]) for i in range(n)]))))
    pt = pd.DataFrame(rows).set_index("model")
    pt.round(4).to_csv(C.REPORTS / ("test_point_metrics.csv" if REAL else f"dryrun_{c}_point_metrics.csv"))
    say("\nPoint forecasts on the fold (WAPE; bias > 0 means over-forecast)")
    say(pt.sort_values("weekly_wape").round(4).to_string())
    fin = pt.loc["LightGBM selected 11 (FINAL, stored model)"]
    sn, ma = pt.loc["seasonal naive"], pt.loc["28-day average"]

    # weekly refresh policies for the inventory simulation
    rolls = {"owner": [], "ma28": [], "wd8": []}
    for r in range(4):
        seen = m.Y[:, :c + 7 * r]
        rolls["owner"].append(np.repeat(seen[:, -7:].mean(axis=1, keepdims=True), H, axis=1))
        rolls["ma28"].append(np.repeat(seen[:, -28:].mean(axis=1, keepdims=True), H, axis=1))
        rolls["wd8"].append(B.weekday_mean(seen))
    rolls["full24"], rolls["lean"] = full[42], lean_rolls
    active = hist[:, -28:].sum(axis=1) > 0
    say(f"\nInventory simulation: {int(active.sum())} of {n} items active (no sale in the 28 days before the origin are left out)")
    by_item = {(k, L): inventory_by_item(m, {c: v}, [c], L) for k, v in rolls.items() for L in LEADS}
    say("Inventory (days of sales) for a revenue-weighted fill rate of 90 / 95 / 98%")
    inv_tab = []
    for L in LEADS:
        for k in POLICY_NAMES:
            s, st, d = by_item[(k, L)]
            fill, inv = s.sum(axis=1) / d.sum(), st.sum(axis=1) / d.sum()
            inv_tab.append(dict(lead=L, policy=POLICY_NAMES[k], **{f"inv@{int(t * 100)}%": inventory_at_fill(fill, inv, t) for t in TARGETS},
                                max_fill=fill.max()))
    inv_df = pd.DataFrame(inv_tab)
    inv_df.round(3).to_csv(C.REPORTS / ("test_inventory.csv" if REAL else f"dryrun_{c}_inventory.csv"), index=False)
    say(inv_df.round(2).to_string(index=False))

    say(f"\nPaired bootstrap over items ({N_BOOT} resamples): change in inventory for the same fill, median [90% interval]")
    rng = np.random.default_rng(C.SEED)
    boot = rng.integers(0, n, size=(N_BOOT, n))
    verdict = {}
    for L in LEADS:
        for target in TARGETS:
            vals = {k: [] for k in POLICY_NAMES}
            for idx in boot:
                for k in POLICY_NAMES:
                    s, st, d = by_item[(k, L)]
                    vals[k].append(inventory_at_fill(s[:, idx].sum(axis=1) / d[idx].sum(), st[:, idx].sum(axis=1) / d[idx].sum(), target))
            vals = {k: np.array(v) for k, v in vals.items()}
            for other in ("owner", "ma28", "wd8", "full24"):
                ratio = vals["lean"] / vals[other] - 1
                ok = ratio[~np.isnan(ratio)]
                if len(ok) < 0.8 * N_BOOT:
                    say(f"L={L} fill {target:.0%}: final vs {POLICY_NAMES[other]}: target often unreachable ({len(ok)}/{N_BOOT})")
                    continue
                lo, med, hi = np.percentile(ok, [5, 50, 95])
                verdict[(L, target, other)] = (med, lo, hi)
                say(f"L={L} fill {target:.0%}: final vs {POLICY_NAMES[other]}: {med:+.1%} [{lo:+.1%}, {hi:+.1%}]")

    say("\nWeekly WAPE by segment (final model vs comparators)")
    wk = lambda f, idx: M.wape(M.block_sums(actual[idx]), M.block_sums(f[idx]))  # noqa: E731
    seg = {}
    for col in ("stratum", "demand_class"):
        for val in sorted(m.items[col].unique()):
            idx = np.where(m.items[col].to_numpy() == val)[0]
            seg[f"{col}={val} ({len(idx)})"] = {name: wk(point[name], idx) for name in
                                              ("seasonal naive", "28-day average", "weekday average, 8 weeks", "LightGBM selected 11 (FINAL, stored model)")}
    say(pd.DataFrame(seg).T.round(4).to_string())

    prev = pd.read_csv(C.REPORTS / "feature_selection_results.csv")
    prev = prev[prev.model.isin(["selected (11)", "28-day average"])][["split", "model", "daily_wape", "weekly_wape", "wape_28d", "inv_days_95"]]
    say("\nSame metrics on the dev folds and the validation fold (earlier runs, for comparison)")
    say(prev.round(4).to_string(index=False))

    say("\nPre-declared hypotheses")
    h1 = fin.weekly_wape <= 0.9 * sn.weekly_wape
    h2 = fin.weekly_wape <= 1.03 * ma.weekly_wape
    say(f"H1 weekly WAPE final {fin.weekly_wape:.4f} vs seasonal naive {sn.weekly_wape:.4f} ({fin.weekly_wape / sn.weekly_wape - 1:+.1%}, needs <= -10%): "
        f"{'SUPPORTED' if h1 else 'NOT SUPPORTED'}")
    say(f"H2 weekly WAPE final {fin.weekly_wape:.4f} vs 28-day average {ma.weekly_wape:.4f} ({fin.weekly_wape / ma.weekly_wape - 1:+.1%}, needs <= +3%): "
        f"{'SUPPORTED' if h2 else 'NOT SUPPORTED'}")
    vo, vm = verdict.get((3, 0.95, "owner")), verdict.get((3, 0.95, "ma28"))
    if vo and vm:
        a = vo[0] <= -0.10 and vo[2] < 0
        b = vm[2] < 0
        say(f"H3 inventory at 95% fill, L=3: vs owner {vo[0]:+.1%} [{vo[1]:+.1%}, {vo[2]:+.1%}] (needs <= -10% and interval below 0): "
            f"{'SUPPORTED' if a else 'NOT SUPPORTED'}; vs 28-day average {vm[0]:+.1%} [{vm[1]:+.1%}, {vm[2]:+.1%}] (interval below 0): "
            f"{'SUPPORTED' if b else 'NOT SUPPORTED'}")
    else:
        say("H3: 95% fill not reachable in enough resamples; NOT SUPPORTED")
    say(f"\nruntime {time.time() - t0:.0f}s")
    OUT.write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
