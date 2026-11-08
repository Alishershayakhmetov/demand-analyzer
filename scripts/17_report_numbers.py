"""Collect the numbers and figures used in docs/ (report and slides) from the stored results.

Weekly WAPE per model for the dev folds, the validation fold and the test fold; inventory days at 95% fill (lead time 3) for the
owner rule, 28-day average, weekday average and the shipped 11-feature model on the four earlier folds (trained here with
the shipped configuration) and on the test fold (read from reports/test_inventory.csv). Writes reports/report_numbers.json
and docs/assets/*.png. Nothing here touches the test fold's labels beyond reading the already written test result files.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demand import baselines as B  # noqa: E402
from demand import config as C  # noqa: E402
from demand.evaluate import load_matrices  # noqa: E402
from demand.features import FEATURES, build_rows, training_rows  # noqa: E402
from demand.inventory import inventory_at_fill  # noqa: E402
from demand.modeling import fit, inventory_by_item, predict_origins  # noqa: E402

FOLDS = C.DEV_CUTOFFS + [C.VAL_CUTOFF]
ASSETS = C.ROOT / "docs" / "assets"
TEAL, SLATE, LIGHT, AMBER, INK = "#1F8A8A", "#5B6B75", "#A8B3BA", "#E0902A", "#0F2A3D"


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    by_fold = pd.read_csv(C.REPORTS / "results_by_fold.csv")
    fsel = pd.read_csv(C.REPORTS / "feature_selection_results.csv")
    test = pd.read_csv(C.REPORTS / "test_point_metrics.csv").set_index("model")

    names = {"seasonal_naive": ("seasonal naive", "seasonal naive"), "moving_avg_28": ("28-day average", "28-day average"),
             "weekday_mean_8w": ("weekday average 8w", "weekday average, 8 weeks"), "AutoETS": ("ETS", "AutoETS")}
    wape = {}
    for key, (label, test_key) in names.items():
        d = by_fold[by_fold.model == key].set_index("cutoff").weekly_wape
        wape[label] = dict(dev=float(d.loc[C.DEV_CUTOFFS].mean()), val=float(d.loc[C.VAL_CUTOFF]), test=float(test.loc[test_key, "weekly_wape"]))
    sel = fsel[fsel.model == "selected (11)"].set_index("split").weekly_wape
    wape["LightGBM (selected)"] = dict(dev=float(sel.iloc[0]), val=float(sel.iloc[1]),
                                       test=float(test.loc["LightGBM selected 11 (FINAL, stored model)", "weekly_wape"]))

    # inventory on the four earlier folds
    m = load_matrices()
    n, H = len(m.ids), C.HORIZON
    selected = json.loads((C.REPORTS / "selected_features.json").read_text())["features"]
    feats = [f for f in FEATURES if f in selected]
    rolls = {k: {} for k in ("owner", "ma28", "wd8", "lean")}
    for c in FOLDS:
        origins = [build_rows(m, c + 7 * k, with_target=False) for k in range(4)]
        rolls["lean"][c] = predict_origins(fit(training_rows(m, c), feats, C.SEED), origins, feats, n)
        for k in ("owner", "ma28", "wd8"):
            rolls[k][c] = []
        for r in range(4):
            seen = m.Y[:, :c + 7 * r]
            rolls["owner"][c].append(np.repeat(seen[:, -7:].mean(axis=1, keepdims=True), H, axis=1))
            rolls["ma28"][c].append(np.repeat(seen[:, -28:].mean(axis=1, keepdims=True), H, axis=1))
            rolls["wd8"][c].append(B.weekday_mean(seen))
        print("fold", c, "done", flush=True)
    per_item = {k: inventory_by_item(m, v, FOLDS, 3) for k, v in rolls.items()}

    def inv95(k, idx=slice(None)):
        s, st, d = per_item[k]
        return inventory_at_fill(s[:, idx].sum(axis=1) / d[idx].sum(), st[:, idx].sum(axis=1) / d[idx].sum(), 0.95)

    rng = np.random.default_rng(C.SEED)
    boot = rng.integers(0, n, size=(1000, n))
    diff = {}
    for other in ("owner", "ma28", "wd8"):
        r = np.array([inv95("lean", i) / inv95(other, i) - 1 for i in boot])
        r = r[~np.isnan(r)]
        diff[other] = [float(np.median(r)), *map(float, np.percentile(r, [5, 95]))]
    tinv = pd.read_csv(C.REPORTS / "test_inventory.csv")
    tinv = tinv[tinv.lead == 3].set_index("policy")["inv@95%"]
    tmap = {"owner": "owner: last 7 days", "ma28": "28-day average", "wd8": "weekday average, 8 weeks",
            "lean": "LightGBM selected 11 features (final)"}
    inventory = {k: dict(earlier=float(inv95(k)), test=float(tinv[tmap[k]])) for k in rolls}
    out = dict(wape=wape, inventory=inventory, lean_vs_earlier_bootstrap=diff,
               counts=dict(items=n, days=int(m.Y.shape[1]), folds=FOLDS, test_cutoff=C.TEST_CUTOFF))
    (C.REPORTS / "report_numbers.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))

    # figures for the Word report (slides use native charts built from the same json)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": SLATE,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "font.size": 10})
    order = ["seasonal naive", "28-day average", "weekday average 8w", "ETS", "LightGBM (selected)"]
    colors = [LIGHT, SLATE, "#7C8B94", "#93A0A8", TEAL]
    ru = {"seasonal naive": "seasonal naive", "28-day average": "среднее за 28 дней", "weekday average 8w": "среднее по дням недели",
          "ETS": "ETS", "LightGBM (selected)": "LightGBM"}
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x, w = np.arange(3), 0.16
    for i, k in enumerate(order):
        vals = [wape[k]["dev"], wape[k]["val"], wape[k]["test"]]
        bars = ax.bar(x + (i - 2) * w, vals, w, label=ru[k], color=colors[i])
        if k in ("seasonal naive", "28-day average", "LightGBM (selected)"):
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x, ["dev (3 окна)", "validation d1913", "тест d1942–1969"])
    ax.set_ylabel("недельный WAPE (ниже — лучше)")
    ax.set_ylim(0, 0.52)
    ax.legend(ncol=5, fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_wape_by_split.png", dpi=200)
    plt.close(fig)

    labels = {"owner": "«как в прошлую\nнеделю»", "ma28": "среднее\nза 28 дней", "wd8": "среднее по\nдням недели", "lean": "LightGBM"}
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x, w = np.arange(4), 0.36
    ax.bar(x - w / 2, [inventory[k]["earlier"] for k in labels], w, label="4 прежних окна", color=SLATE)
    ax.bar(x + w / 2, [inventory[k]["test"] for k in labels], w, label="тест", color=TEAL)
    for xi, k in zip(x, labels):
        ax.text(xi - w / 2, inventory[k]["earlier"] + 0.12, f"{inventory[k]['earlier']:.2f}", ha="center", fontsize=8, color=INK)
        ax.text(xi + w / 2, inventory[k]["test"] + 0.12, f"{inventory[k]['test']:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, list(labels.values()))
    ax.set_ylabel("запас, дней продаж\n(95% обслуживания, поставка 3 дня)")
    ax.set_ylim(0, 10.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_inventory_by_split.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
