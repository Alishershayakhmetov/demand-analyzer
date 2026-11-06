"""Forward selection of feature groups on the three dev folds only, then one confirmation on the validation fold d1913.

Rules fixed before running:
- start from recent-demand level features + horizon step; candidates are the other feature groups;
- criterion: mean weekly WAPE over the dev folds (d1829, d1857, d1885), default hyper-parameters, seed 42;
- a group is added only if it lowers the criterion by more than max(2 x seed std of the start set, 0.001);
- the validation fold is not looked at until the set is chosen; the test fold is not used.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demand import config as C  # noqa: E402
from demand.evaluate import load_matrices  # noqa: E402
from demand.features import FEATURES, build_rows, training_rows  # noqa: E402
from demand.modeling import fit, inventory_days, moving_average_origins, predict_origins, window_metrics  # noqa: E402

DEV, VAL = C.DEV_CUTOFFS, [C.VAL_CUTOFF]
PRICE = ["price", "price_vs_median", "price_vs_origin"]
WD_HIST = ["wd_mean8", "wd_mean4", "wd_last"]
LEVEL = ["mean7", "mean14", "mean28", "mean56", "mean91", "std28", "nz28", "nz91", "days_since_sale", "age"]
BASE = [*LEVEL, "h"]
GROUPS = {"price": PRICE, "SNAP": ["snap"], "events": ["event"], "month / day of month": ["month", "dom"],
          "weekday of target day": ["wday"], "weekday history": WD_HIST, "item type": ["dept", "cat"]}
SEEDS = [42, 1, 2]

m = load_matrices()
N = len(m.ids)
DATA: dict[int, tuple] = {}
CACHE: dict[tuple, list[np.ndarray]] = {}


def ordered(feats) -> list[str]:
    return [f for f in FEATURES if f in set(feats)]


def forecasts(feats, seed: int, cutoff: int) -> list[np.ndarray]:
    key = (tuple(ordered(feats)), seed, cutoff)
    if key not in CACHE:
        if cutoff not in DATA:
            DATA[cutoff] = (training_rows(m, cutoff), [build_rows(m, cutoff + 7 * k, with_target=False) for k in range(4)])
        train, origins = DATA[cutoff]
        CACHE[key] = predict_origins(fit(train, ordered(feats), seed), origins, ordered(feats), N)
    return CACHE[key]


def score(feats, seed: int, folds: list[int]) -> dict[str, float]:
    per_fold = [window_metrics(m, c, forecasts(feats, seed, c)[0]) for c in folds]
    return {k: float(np.mean([p[k] for p in per_fold])) for k in per_fold[0]}


def inventory(feats, seed: int, folds: list[int]) -> float:
    return inventory_days(m, {c: forecasts(feats, seed, c) for c in folds}, folds)


def main() -> None:
    out = []

    def say(s: str = "") -> None:
        print(s, flush=True)
        out.append(s)

    t0 = time.time()
    base_scores = [score(BASE, s, DEV)["weekly_wape"] for s in SEEDS]
    thr = max(2 * np.std(base_scores, ddof=1), 0.001)
    say(f"start set (recent-demand level + horizon): dev weekly WAPE per seed {np.round(base_scores, 4)}, "
        f"threshold for adding a group {thr:.4f}\n")

    current, selected, rounds = list(BASE), [], []
    cur = score(current, 42, DEV)["weekly_wape"]
    while True:
        cand = {}
        for g, fs in GROUPS.items():
            if g not in selected:
                s = score([*current, *fs], 42, DEV)
                cand[g] = s
                rounds.append(dict(round=len(selected) + 1, group=g, **s))
        if not cand:
            break
        best = min(cand, key=lambda g: cand[g]["weekly_wape"])
        say(f"round {len(selected) + 1}: current weekly {cur:.4f}; " + ", ".join(f"{g} {s['weekly_wape']:.4f}" for g, s in sorted(cand.items(), key=lambda kv: kv[1]["weekly_wape"])))
        if cur - cand[best]["weekly_wape"] > thr:
            say(f"  -> add '{best}' ({cur - cand[best]['weekly_wape']:+.4f})")
            selected.append(best)
            current += GROUPS[best]
            cur = cand[best]["weekly_wape"]
        else:
            say(f"  -> stop, best candidate '{best}' improves by only {cur - cand[best]['weekly_wape']:+.4f} (< {thr:.4f})")
            break
    pd.DataFrame(rounds).to_csv(C.REPORTS / "feature_selection_rounds.csv", index=False)
    chosen = ordered(current)
    say(f"\nselected groups: {selected or 'none (level + horizon only)'}\nselected features ({len(chosen)}): {chosen}")
    json.dump(dict(features=chosen, groups=selected, threshold=thr), open(C.REPORTS / "selected_features.json", "w"), indent=1)

    sets = {"full (24 features)": FEATURES, "level + horizon": BASE, f"selected ({len(chosen)})": chosen}
    say("\nSeed-averaged results (3 seeds, mean +- std of the seeds); the validation fold was first looked at here")
    rows = []
    for split, folds in [("dev folds (d1829, d1857, d1885)", DEV), ("validation fold d1913", VAL)]:
        for name, feats in sets.items():
            sc = [score(feats, s, folds) for s in SEEDS]
            inv = [inventory(feats, s, folds) for s in SEEDS]
            rows.append(dict(split=split, model=name, **{k: np.mean([x[k] for x in sc]) for k in sc[0]},
                             weekly_std=np.std([x["weekly_wape"] for x in sc], ddof=1), inv_days_95=np.mean(inv)))
        ref = [moving_average_origins(m, c, DATA[c][1] if c in DATA else [build_rows(m, c + 7 * k, with_target=False) for k in range(4)]) for c in folds]
        per = [window_metrics(m, c, r[0]) for c, r in zip(folds, ref)]
        rows.append(dict(split=split, model="28-day average", **{k: np.mean([p[k] for p in per]) for k in per[0]},
                         weekly_std=np.nan, inv_days_95=inventory_days(m, dict(zip(folds, ref)), folds)))
    res = pd.DataFrame(rows)
    res.to_csv(C.REPORTS / "feature_selection_results.csv", index=False)
    say(res.round(4).to_string(index=False))

    say("\nWeekly WAPE per fold (mean of 3 seeds)")
    tab = {name: [np.mean([window_metrics(m, c, forecasts(f, s, c)[0])["weekly_wape"] for s in SEEDS]) for c in [*DEV, *VAL]] for name, f in sets.items()}
    ma = []
    for c in [*DEV, *VAL]:
        origins = DATA[c][1] if c in DATA else [build_rows(m, c + 7 * k, with_target=False) for k in range(4)]
        ma.append(window_metrics(m, c, moving_average_origins(m, c, origins)[0])["weekly_wape"])
    tab["28-day average"] = ma
    say(pd.DataFrame(tab, index=[f"fold {c}" for c in [*DEV, *VAL]]).T.round(4).to_string())
    say(f"\nruntime {time.time() - t0:.0f}s")
    (C.REPORTS / "feature_selection.txt").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
