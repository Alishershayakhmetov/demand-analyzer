"""Inventory simulation on the dev + validation folds (test fold not used).

Every forecasting method drives the same order-up-to policy: S = forecast demand over lead time + review period
+ z * sigma. Forecasts are refreshed at every weekly review from the demand observed so far (the LightGBM model itself
stays fixed at the fold cutoff); `lgbm_static` keeps the single forecast made at the cutoff. Sweeping z gives an inventory-service frontier; the question is how much inventory each method needs
for the same fill rate. Fill rate is revenue-weighted, inventory is measured in days of sales (the cost ratio cancels).
Assumptions: review every 7 days, lead time L in {1, 3, 7} days, lost sales, unit demand = observed sales
(real demand is censored by stockouts, which the data does not show), start stock = L days of the 28-day average.
Items with no sale in the 28 days before a fold's cutoff are left out of that fold (dormant: demand that restarts after
a long gap is invisible to every method, and the data cannot tell a stock-out from a delisting).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demand import config as C  # noqa: E402
from demand.evaluate import load_forecasts, load_matrices  # noqa: E402
from demand.baselines import weekday_mean  # noqa: E402
from demand.inventory import extend, inventory_at_fill, simulate  # noqa: E402

LEADS = [1, 3, 7]
Z = np.arange(-1.0, 6.01, 0.25)
TARGETS = [0.90, 0.95, 0.98]
FOLDS = C.DEV_CUTOFFS + [C.VAL_CUTOFF]
POLICIES = {"owner_last7": "owner: last 7 days", "ma28": "28-day average", "wd8": "weekday average, 8 weeks",
            "lgbm_static": "LightGBM, one forecast", "lgbm": "LightGBM, weekly refresh",
            "lgbm_qspread": "LightGBM + own P90 spread"}
N_BOOT = 300


def policy_inputs(name: str, lead: int, cutoff: int, m, d: dict, roll: dict) -> tuple[np.ndarray, np.ndarray]:
    """(reviews, items) mean demand and sigma over the protection interval of each weekly review."""
    actual = d["actual"]
    full = np.concatenate([m.Y[:, :cutoff], actual], axis=1)
    span = lead + 7
    means, sigmas = [], []
    for r in range(4):
        seen = full[:, :cutoff + 7 * r]  # what the shop knows at review r
        sigma = seen[:, -91:].std(axis=1) * np.sqrt(span)
        if name == "owner_last7":
            mean = seen[:, -7:].mean(axis=1) * span
        elif name == "ma28":
            mean = seen[:, -28:].mean(axis=1) * span
        elif name == "wd8":
            mean = weekday_mean(seen)[:, :span].sum(axis=1)
        elif name == "lgbm_static":
            mean = extend(d["lgbm_default"])[:, 7 * r:7 * r + span].sum(axis=1)
        elif name in ("lgbm", "lgbm_qspread"):
            f = roll[f"roll{r}_mean"]
            mean = f[:, :span].sum(axis=1)
            if name == "lgbm_qspread":
                spread = np.clip(roll[f"roll{r}_q90"] - f, 0, None) / 1.2816
                sigma = np.sqrt((spread[:, :span] ** 2).sum(axis=1))
        elif name == "oracle":  # perfect foresight, no safety stock
            mean, sigma = extend(actual)[:, 7 * r:7 * r + span].sum(axis=1), np.zeros(len(actual))
        else:
            raise ValueError(name)
        means.append(mean), sigmas.append(sigma)
    return np.stack(means), np.stack(sigmas)


def run(m, fc, rolling) -> tuple[dict, float]:
    """res[lead][policy] = dict(sold, stock, dem) with arrays (folds, z, items) / (folds, items); also the
    share of demand value left out because the item was dormant before the fold."""
    res = {}
    left_out = []
    for lead in LEADS:
        res[lead] = {}
        for name in [*POLICIES, "oracle"]:
            sold, stock, dem = [], [], []
            for cutoff in FOLDS:
                d, roll = fc[cutoff], rolling[cutoff]
                price = np.nanmedian(m.PR[:, cutoff - 91:cutoff], axis=1)
                assert not np.isnan(price).any()
                active = (m.Y[:, cutoff - 28:cutoff].sum(axis=1) > 0).astype(float)
                init = np.ceil(m.Y[:, cutoff - 28:cutoff].mean(axis=1) * lead)
                mean_p, sig_p = policy_inputs(name, lead, cutoff, m, d, roll)
                s, k = simulate(d["actual"], mean_p, sig_p, Z if name != "oracle" else np.array([0.0]), lead, init)
                demand_value = d["actual"][:, lead:].sum(axis=1) * price
                sold.append(s * price * active), stock.append(k * price * active), dem.append(demand_value * active)
                if name == "oracle" and lead == LEADS[0]:
                    left_out.append(((1 - active) * demand_value).sum() / demand_value.sum())
            res[lead][name] = dict(sold=np.array(sold), stock=np.array(stock), dem=np.array(dem))
    return res, float(np.mean(left_out))


def frontier(r: dict, folds=slice(None), items=slice(None)) -> tuple[np.ndarray, np.ndarray]:
    dem = r["dem"][folds][..., items].sum(axis=(0, -1))
    fill = r["sold"][folds][..., items].sum(axis=(0, -1)) / dem
    inv = r["stock"][folds][..., items].sum(axis=(0, -1)) / dem
    return fill, inv


def main() -> None:
    m = load_matrices()
    fc = load_forecasts("baselines", "lgbm")
    rolling = load_forecasts("rolling")
    res, left_out = run(m, fc, rolling)
    out = []

    def say(s: str = "") -> None:
        print(s)
        out.append(s)

    say(f"dormant items left out: {left_out:.1%} of demand value on average per fold")
    n_active = [(m.Y[:, c - 28:c].sum(axis=1) > 0).sum() for c in FOLDS]
    say(f"active items per fold: {n_active} of {len(m.ids)}\n")

    rows = []
    for lead in LEADS:
        for name, r in res[lead].items():
            fill, inv = frontier(r)
            zz = Z if name != "oracle" else np.array([0.0])
            rows += [dict(lead=lead, policy=name, z=z, fill=f, inv_days=i) for z, f, i in zip(zz, fill, inv)]
    pd.DataFrame(rows).to_csv(C.REPORTS / "inventory_frontier.csv", index=False)

    say("Inventory (days of sales) needed for a revenue-weighted fill rate of 90 / 95 / 98%, pooled over 4 folds")
    for lead in LEADS:
        say(f"\nlead time {lead} days")
        tab = {}
        for name in POLICIES:
            fill, inv = frontier(res[lead][name])
            tab[POLICIES[name]] = [inventory_at_fill(fill, inv, t) for t in TARGETS] + [fill.max()]
        fo, io = frontier(res[lead]["oracle"])
        tab["perfect foresight, no safety stock"] = [np.nan, np.nan, np.nan, float(fo[0])]
        say(pd.DataFrame(tab, index=["inv@90%", "inv@95%", "inv@98%", "max fill"]).T.round(2).to_string())
        say(f"perfect foresight: fill {fo[0]:.3f} at {io[0]:.2f} days of inventory")

    say("\nBootstrap over items (300 resamples, paired): change in inventory needed for the same fill, 90% interval")
    rng = np.random.default_rng(C.SEED)
    n = len(m.ids)
    boot = rng.integers(0, n, size=(N_BOOT, n))
    for lead in LEADS:
        for target in TARGETS:
            vals = {k: [] for k in POLICIES}
            for idx in boot:
                for k in POLICIES:
                    fill, inv = frontier(res[lead][k], items=idx)
                    vals[k].append(inventory_at_fill(fill, inv, target))
            vals = {k: np.array(v) for k, v in vals.items()}
            for a, b in [("lgbm", "owner_last7"), ("lgbm", "ma28"), ("lgbm", "wd8"), ("lgbm", "lgbm_static"), ("lgbm_qspread", "lgbm")]:
                ratio = vals[a] / vals[b] - 1
                ok = ratio[~np.isnan(ratio)]
                if len(ok) < N_BOOT * 0.8:
                    say(f"L={lead} fill {target:.0%}: {a} vs {b}: target often unreachable ({len(ok)}/{N_BOOT} resamples)")
                else:
                    lo, hi = np.percentile(ok, [5, 95])
                    say(f"L={lead} fill {target:.0%}: {POLICIES[a]} vs {POLICIES[b]}: {np.median(ok):+.1%} [{lo:+.1%}, {hi:+.1%}]")

    say("\nStability, lead time 3 days, fill 95%: inventory days per fold")
    tab = {}
    for name in POLICIES:
        tab[POLICIES[name]] = [inventory_at_fill(*frontier(res[3][name], folds=[i]), 0.95) for i in range(len(FOLDS))]
    say(pd.DataFrame(tab, index=[f"fold {c}" for c in FOLDS]).T.round(2).to_string())

    say("\nBy volume stratum, lead time 3 days, fill 95%: inventory days")
    strata = m.items.stratum.to_numpy()
    tab = {}
    for name in POLICIES:
        tab[POLICIES[name]] = {s: inventory_at_fill(*frontier(res[3][name], items=np.where(strata == s)[0]), 0.95) for s in ["high", "mid", "low"]}
    say(pd.DataFrame(tab).T.round(2).to_string())
    (C.REPORTS / "inventory_summary.txt").write_text("\n".join(out), encoding="utf-8")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for a, lead in zip(ax, LEADS):
        for name, label in POLICIES.items():
            fill, inv = frontier(res[lead][name])
            a.plot(inv, fill, label=label, lw=1.8)
        fo, io = frontier(res[lead]["oracle"])
        a.scatter(io, fo, c="k", marker="*", s=80, label="perfect foresight", zorder=5)
        a.axhline(0.95, c="grey", lw=0.6, ls=":")
        a.set_xlim(0, 12)
        a.set_ylim(0.6, 1.0)
        a.set_title(f"lead time {lead} days")
        a.set_xlabel("average inventory, days of sales")
    ax[0].set_ylabel("fill rate (revenue-weighted)")
    ax[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(C.REPORTS / "figures" / "inventory_frontier.png", dpi=110)


if __name__ == "__main__":
    main()
