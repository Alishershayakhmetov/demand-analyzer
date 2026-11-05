# Demand Copilot

Demand forecasting and purchase planning for a small shop. Course project, *AI Essentials: Theory and Applications*.

## Status

| Stage | State |
|---|---|
| Data download, single-store panel, time-based folds | done |
| EDA | done (`reports/eda_summary.txt`, `reports/figures/`) |
| Baselines (naive, seasonal naive, moving average, weekday mean, ETS, Croston-SBA, TSB, empirical quantiles) | done |
| Global LightGBM (Tweedie + quantiles), random-search tuning, daily / weekly / 28-day metrics | done |
| Error analysis, SHAP, ablation | not started |
| Inventory simulator and purchase policies | not started |
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
```

`data/` is not tracked by git; everything in it is rebuilt by the scripts.
