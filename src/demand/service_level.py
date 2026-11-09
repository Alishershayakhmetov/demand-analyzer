"""Safety factor z calibrated on the file's own backtest instead of on M5.

The forecasts the chosen method made for the last backtest windows are run through the order-up-to simulator against what really sold;
z is the smallest value that reaches the target revenue-agnostic fill rate (units) over those windows. If the target is not reached or there is
no backtest, the caller falls back to the M5 calibration.
"""
import numpy as np

from .config import HORIZON
from .inventory import extend, interval_sums, inventory_at_fill, simulate
from .modeling import Z


def file_frontier(Y: np.ndarray, bt: list, lead: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Fill rate and inventory (units-days) for every z in Z, pooled over the backtest windows; None without usable windows."""
    sold = np.zeros(len(Z)); stock = np.zeros(len(Z)); dem = 0.0
    span = lead + 7
    for o, f in bt:
        hist, actual = Y[:, :o], Y[:, o:o + HORIZON]
        active = hist[:, -28:].sum(axis=1) > 0
        if not active.any():
            continue
        sig = np.tile(hist[:, -91:].std(axis=1) * np.sqrt(span), (4, 1))
        s, k = simulate(actual, interval_sums(extend(f), lead), sig, Z, lead, np.ceil(hist[:, -28:].mean(axis=1) * lead))
        sold += s[:, active].sum(axis=1); stock += k[:, active].sum(axis=1); dem += actual[active][:, lead:].sum()
    if dem <= 0:
        return None
    return sold / dem, stock / dem


def file_z(Y: np.ndarray, bt: list, lead: int, target: float) -> float | None:
    fr = file_frontier(Y, bt, lead)
    if fr is None:
        return None
    fill = np.maximum.accumulate(fr[0])
    if target > fill[-1]:
        return None
    return float(np.interp(target, fill, Z))
