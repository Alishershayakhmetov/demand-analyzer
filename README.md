# Demand Copilot

Demand forecasting and purchase planning for a small shop. Course project, *AI Essentials: Theory and Applications*.

## Status

| Stage | State |
|---|---|
| Data download, single-store panel, time-based folds | done |
| EDA | done (`reports/eda_summary.txt`, `reports/figures/`) |
| Baselines (naive, seasonal naive, moving average, weekday mean, ETS, Croston-SBA, TSB, empirical quantiles) | done |
| Global LightGBM (Tweedie + quantiles), random-search tuning, daily / weekly / 28-day metrics | done |
| Error analysis | done |
| SHAP, ablation | done |
| Inventory simulator and purchase policies | done (frontier comparison, no profit claim) |
| Streamlit app | not started |
| Test fold evaluation | **not run on purpose**, to be opened once at the end |

## Data

M5 (Walmart) from the GitHub mirror of the Kaggle files, no account needed. We use one store (`CA_1`) and
300 items: 100 each from the low, mid and high volume terciles, picked with data up to day 1829 only.
Daily sales for 1969 days (2011-01-29 to 2016-06-19), prices, calendar events and SNAP days.

The data has **no stock levels or purchase costs**. The inventory part is therefore a simulation with stated
assumptions (to be documented when it is built).

## Evaluation protocol

Time-based only. A fold trains on `d <= cutoff` and forecasts the next 28 days (four Monday-to-Sunday weeks).
Dev folds end at d1829, d1857, d1885; validation fold d1913; **test fold d1941 (d1942..d1969) is unused so far**.
Prices, SNAP and calendar columns of the forecast days are treated as known in advance; demand features use only the past.
A forecast is set to 0 for days without a price (the item is not on sale that week). This applies to every model.

Metrics: WAPE at three aggregation levels (item-day, item-week, item-28-days), bias, RMSSE (scale = in-sample one-step
naive MSE), pinball loss and coverage for quantiles. Ordering decisions are weekly, so weekly WAPE is the headline number.

Tuning (`scripts/06_tune.py`): random search over 16 configurations plus the default, trained on the three dev folds;
selection rule fixed in advance (lowest mean weekly WAPE on dev folds). The validation fold is not used for selection.

## Results (300 items, 28-day horizon)

Mean over the four folds / validation fold only (d1913). The dev folds were used to choose the tuned model, so its
four-fold mean is optimistic; the default LightGBM was never tuned, so its mean is clean.

| Model | daily WAPE | weekly WAPE | 28-day WAPE | weekly WAPE, val fold |
|---|---|---|---|---|
| naive | 0.982 | 0.823 | 0.799 | 0.694 |
| seasonal naive | 0.819 | 0.406 | 0.338 | 0.370 |
| weekday mean 8w | 0.679 | 0.347 | 0.251 | 0.367 |
| AutoETS | 0.684 | 0.370 | 0.280 | 0.339 |
| moving average 28 | 0.682 | 0.342 | 0.247 | 0.350 |
| LightGBM default | 0.666 | 0.343 | 0.244 | 0.342 |
| LightGBM tuned | 0.658 | 0.337 | 0.238 | 0.344 |

What this says:
- Against seasonal naive the LightGBM is clearly better (weekly 0.343 vs 0.406, -16%).
- Against the best trivial baseline (28-day average) the gain is small: 0% weekly, 1.9% daily (default model).
- Tuning improved the dev-fold weekly WAPE from 0.3425 to 0.3350, but on the untouched validation fold the tuned
  model is not better (0.344 vs 0.342). The gain is within noise of a single 28-day window.
- Aggregating over weeks halves the error (daily 0.66, weekly 0.34, 28-day 0.24): item-day demand is mostly noise,
  so purchasing should be judged on weekly or lead-time demand.

Quantiles (pinball, lower is better; coverage target 0.5 / 0.9), mean over 4 folds / validation fold:

| Model | pinball | val pinball | coverage | val coverage |
|---|---|---|---|---|
| weekday quantile q90 | 0.487 | 0.480 | 0.850 | 0.834 |
| LightGBM default q90 | 0.462 | 0.504 | 0.899 | 0.885 |
| LightGBM tuned q90 | 0.458 | 0.481 | 0.895 | 0.880 |

The q90 advantage over the empirical baseline shows in the four-fold mean but not on the validation fold, so it is
not established yet. Coverage of the LightGBM q90 is closer to 0.90 than the baseline (0.85) on average.

## Error analysis (`scripts/08_error_analysis.py`, full tables in `reports/error_analysis.txt`)

Model analysed: the untuned LightGBM, 4 folds, 300 items. Figures: `reports/figures/error_*.png`.

- **Error follows volume, not model quality.** The 30 items (10%) with the largest error carry 34% of it; the high-volume
  tercile has 75% of units and 61% of the error. The 10 worst items are all high-volume, with weekly WAPE 0.10-0.40
  and no level shifts (median shift 0.16 vs 0.28 for all items): the forecast follows level and weekday pattern but
  not daily spikes. Days with zero sales are 52% of item-days but only 30% of the error.
- **Where LightGBM helps** (weekly WAPE vs the 28-day moving average): smooth items (+1% weekly, +9% daily),
  high-volume items (+5% daily), mid-volume items (+5% weekly). Tuesday to Thursday are 6-7% better (daily WAPE), Saturday is not.
- **Where it does not:** low-volume items (weekly WAPE 0.75 vs 0.66 for the average; bias +8%, it over-forecasts them),
  erratic/lumpy items (no gain), and small price cuts (daily WAPE 1.18 vs 0.82, bias +20%). Only 1.4% of units fall
  in promo weeks (31 item-weeks), so the price effect cannot be judged with this data.
- **The edge over a 28-day average is not stable.** Weekly WAPE gain per fold: +4.1%, -2.9%, -4.0%, +2.2%.
- **Bias.** LightGBM under-forecasts the fourth week (-9% on average; in 3 of 4 folds worse than the moving average),
  and the sign of the overall bias changes between folds (-6.6%, -3.9%, +5.2%, -7.7%). That comes from level changes
  of the store between windows and matters for safety stock: a policy built on these forecasts needs the sensitivity analysis.
- **P90 calibration** by segment: LightGBM 0.86-0.92 (target 0.90), weekday empirical quantile 0.81-0.88.
  The LightGBM quantile is better calibrated in every segment; erratic items are the weakest (0.86).

## Inventory simulation (`src/demand/inventory.py`, `scripts/09_rolling_forecasts.py`, `scripts/10_inventory.py`)

The data has no stock levels, so this is a simulation; the point is to compare forecasting methods under identical rules.

Rules (all assumptions): one shop, every item reordered every 7 days, order arrives after a lead time L (1, 3 or 7 days),
order-up-to level S = forecast demand over lead time + review period + z * sigma, lost sales (no backorders), integer units,
start stock = L days of the 28-day average, demand = observed sales. Forecasts are refreshed at each weekly review from
the demand seen so far; the LightGBM model is **not** retrained, only its inputs are refreshed (`lgbm_static` keeps one
forecast per fold for reference). sigma is the standard deviation of the last 91 days times sqrt(L + 7) for every method
except "own P90 spread", which takes it from the model's quantile forecast. Sweeping z traces an inventory-service
frontier (`reports/figures/inventory_frontier.png`). Fill rate is revenue-weighted; inventory is in days of sales, so the
unknown purchase-cost ratio cancels out and no profit is claimed. Items with no sale in the 28 days before a fold are
left out of that fold (3.4% of demand value): demand that restarts after a long gap is invisible to every method.
The logic is covered by `tests/test_inventory.py` (hand-computed example, conservation and monotonicity checks).

Average inventory (days of sales) needed for the same fill rate, lead time 3 days, pooled over 4 folds:

| Policy | 90% fill | 95% fill | 98% fill |
|---|---|---|---|
| owner: last 7 days | 6.12 | 7.78 | 10.17 |
| 28-day average | 5.06 | 6.83 | 9.42 |
| weekday average, 8 weeks | 5.07 | 6.96 | 9.97 |
| LightGBM, one forecast | 4.95 | 6.80 | 9.57 |
| **LightGBM, weekly refresh** | **4.86** | **6.51** | **8.85** |
| LightGBM + own P90 spread | 4.87 | 6.63 | 9.08 |
| perfect foresight (no safety stock) | 3.29 days for 100% fill | | |

Paired bootstrap over items, 90% interval, change in inventory for the same 95% fill:
- vs the owner's last-7-days rule: -16.6% [-18.6%, -14.3%] (-12% at lead time 1, -19% at lead time 7)
- vs the 28-day average: -4.9% [-7.1%, -2.5%]; vs the 8-week weekday average: -6.5% [-9.2%, -3.9%]
- weekly refresh vs a single forecast: -4.3% [-7.5%, -1.6%]
- own P90 spread vs the plain sigma: +1.9% [+0.9%, +3.1%], i.e. the quantile model adds nothing here.

The ranking is the same in every fold (lead time 3, 95% fill: LightGBM refresh 6.96 / 6.30 / 6.27 / 6.49 days vs 28-day
average 7.46 / 6.66 / 6.50 / 6.74) and at all three lead times. The saving over a 28-day average is modest (2-6%)
although the forecast WAPE looked tied. The ablation below rules out the calendar and price features as the reason: a
model with only recent-demand features keeps the advantage (6.59 vs 6.54 days at 95% fill; 28-day average 6.83).
Limits: one store, 4 windows of 28 days, intervals only cover item sampling, demand is censored by real stock-outs,
the test fold is not used.

## SHAP and feature ablation (`scripts/11_shap.py`, `scripts/12_ablation.py`)

SHAP (exact tree SHAP from LightGBM, forecast rows of fold d1913, log scale of the Tweedie link; it describes the model,
not causal effects). Details in `reports/shap_summary.txt`, figures `shap_beeswarm.png`, `shap_groups.png`.
- Recent demand level carries 71% of the attribution (mean of the last 7/14/28/56/91 days, zero share, days since last
  sale); calendar 10%, item type 5%, weekday history 4%, price 4%, horizon 3%, events 1.5%, SNAP 0.8%.
- Weekday: Saturday/Sunday x1.21-1.23, Tuesday-Thursday x0.87-0.88. SNAP days only x1.02. Price cuts below 0.98 of the
  median give x1.56-1.59, but from only 139 forecast rows; price rises give x1.09, which is not causal (the model has
  no reason to raise demand when the price rises), so price attributions should not be read as elasticities.
- The "horizon" attribution is confounded with the weekday: every forecast origin is a Sunday, so step h fixes the
  weekday of the target day. Looking only at Sundays (steps 7, 14, 21, 28), the effect grows from x0.98 to x1.06.

Ablation: each feature group is removed, the model is retrained on every fold and scored on the same 4 folds
(default hyper-parameters, mean over folds). The full model is trained with 3 seeds; the seed noise is 0.07% (daily
WAPE), 0.24% (weekly), 0.48% (28-day) and 0.6% (inventory days), so smaller differences are not interpretable.

| Variant | daily WAPE | weekly WAPE | 28-day WAPE | inventory days, 95% fill, L=3 |
|---|---|---|---|---|
| full model (3 seeds) | 0.6663 | 0.3432 | 0.2428 | 6.54 |
| no price | 0.6721 (+0.9%) | 0.3483 (+1.5%) | 0.2451 (+1.0%) | 6.51 |
| no month / day of month | 0.6735 (+1.1%) | 0.3488 (+1.6%) | 0.2497 (+2.8%) | 6.55 |
| no long history (56/91 d, age) | 0.6715 (+0.8%) | 0.3475 (+1.3%) | 0.2456 (+1.1%) | 6.57 |
| no item type | 0.6700 (+0.6%) | 0.3460 (+0.8%) | 0.2456 (+1.1%) | 6.59 |
| no SNAP / no events / no weekday / no horizon step | within +-0.8% | within +-0.8% | within +-0.9% | 6.50-6.55 |
| no known-future covariates (price, SNAP, events, calendar) | 0.6621 (-0.6%) | **0.3348 (-2.4%)** | **0.2337 (-3.7%)** | 6.53 |
| recent-demand level + horizon only | 0.6649 (-0.2%) | 0.3374 (-1.7%) | 0.2388 (-1.6%) | 6.59 |
| 28-day average (reference) | 0.6817 | 0.3424 | 0.2474 | 6.83 |

What this says:
- Price, month/day of month, long history and item type each help a little (1-3%, above the noise).
- **The known-future covariates as a whole do not help; removing them all improves the weekly and 28-day WAPE by 2-4%,
  beyond the noise.** The model overfits them at this data size (300 items, one store). Single groups do not show it
  because the remaining covariates compensate. The weekday features are redundant by construction (see above).
- The inventory result is flat: no variant differs from the full model by more than ~1%, while all stay about 4% below
  the 28-day average. LightGBM's inventory advantage therefore comes from the adaptive recent-level features and the weekly
  refresh, not from price, SNAP or calendar.
- This was measured on the same four folds that were used for earlier comparisons, so choosing the leaner model on
  this evidence alone would be selection on those folds. To be settled with dev folds only, then checked on d1913.
- Consequence for the app: the model only saw Sunday origins. It must be run on Mondays, or retrained with origins on all
  weekdays, otherwise its step-to-weekday mapping is wrong.

## Run

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python scripts/01_download.py    # ~50 MB from GitHub into data/raw
.venv/Scripts/python scripts/02_prepare.py     # subset + panel -> data/processed
.venv/Scripts/python scripts/03_eda.py
.venv/Scripts/python scripts/04_baselines.py   # also stores forecasts
.venv/Scripts/python scripts/05_lightgbm.py    # first untuned version
.venv/Scripts/python scripts/06_tune.py        # ~15 min; stores forecasts_lgbm.npz
.venv/Scripts/python scripts/07_report.py      # daily / weekly / 28-day tables -> reports/
.venv/Scripts/python scripts/08_error_analysis.py
.venv/Scripts/python scripts/09_rolling_forecasts.py   # weekly refreshed LightGBM forecasts
.venv/Scripts/python scripts/10_inventory.py           # simulation, frontier, bootstrap
.venv/Scripts/python scripts/11_shap.py
.venv/Scripts/python scripts/12_ablation.py            # ~10 min
.venv/Scripts/python -m unittest discover -s tests
```

`data/` is not tracked by git; everything in it is rebuilt by the scripts.
