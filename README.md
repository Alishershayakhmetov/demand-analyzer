# Demand Copilot

Demand forecasting and purchase planning for a small shop. Course project, *AI Essentials: Theory and Applications*.

## Status

| Stage | State |
|---|---|
| Data download, single-store panel, time-based folds | done |
| EDA | done (`reports/eda_summary.txt`, `reports/figures/`) |
| Baselines (naive, seasonal naive, moving average, weekday mean, ETS, Croston-SBA, TSB, empirical quantiles) | done |
| Global LightGBM (Tweedie + quantiles) | first version, untuned |
| Inventory simulator and purchase policies | not started |
| Error analysis, SHAP, ablation | not started |
| Streamlit app | not started |
| Test fold evaluation | **not run on purpose**, to be opened once at the end |

## Results so far (mean over the 4 dev/validation folds, 300 items, 28-day horizon)

| Model | WAPE | RMSSE | bias |
|---|---|---|---|
| naive | 0.982 | 0.974 | +0.286 |
| seasonal naive | 0.819 | 0.945 | -0.023 |
| moving average 28 | 0.682 | 0.724 | -0.034 |
| weekday mean 8w | 0.679 | 0.747 | -0.049 |
| AutoETS | 0.684 | 0.719 | -0.021 |
| **LightGBM (Tweedie, untuned)** | **0.666** | 0.717 | -0.033 |

LightGBM is 18.7% better than seasonal naive but only 1.9% better than the best simple baseline (weekday mean).
Item-day demand is very noisy (54% of on-sale item-days are zero), so the gap to simple baselines is small at this level.

Quantile forecasts (pinball loss, lower is better): LightGBM q50 0.465 vs 0.510 and q90 0.462 vs 0.487 for the
weekday empirical quantile; q90 coverage 0.899 (target 0.90) vs 0.850.

## Data

M5 (Walmart) from the GitHub mirror of the Kaggle files, no account needed. We use one store (`CA_1`) and
300 items: 100 each from the low, mid and high volume terciles, picked with data up to day 1829 only.
Daily sales for 1969 days (2011-01-29 to 2016-06-19), prices, calendar events and SNAP days.

The data has **no stock levels or purchase costs**. The inventory part is therefore a simulation with stated
assumptions (to be documented when it is built).

## Evaluation protocol

Time-based only. A fold trains on `d <= cutoff` and forecasts the next 28 days.
Dev folds end at d1829, d1857, d1885; validation fold d1913; **test fold d1941 (d1942..d1969) is unused so far**.
Prices, SNAP and calendar columns of the forecast days are treated as known in advance; demand features use only the past.
A forecast is set to 0 for days without a price (the item is not on sale that week). This applies to every model.

Metrics: WAPE, bias, RMSSE (scale = in-sample one-step naive MSE), pinball loss and coverage for quantiles.

## Run

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python scripts/01_download.py    # ~50 MB from GitHub into data/raw
.venv/Scripts/python scripts/02_prepare.py     # subset + panel -> data/processed
.venv/Scripts/python scripts/03_eda.py
.venv/Scripts/python scripts/04_baselines.py
.venv/Scripts/python scripts/05_lightgbm.py
```

`data/` is not tracked by git; everything in it is rebuilt by the scripts.
