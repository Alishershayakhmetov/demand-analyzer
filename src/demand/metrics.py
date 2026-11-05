import numpy as np


def wape(y: np.ndarray, f: np.ndarray) -> float:
    return float(np.abs(y - f).sum() / y.sum())


def bias(y: np.ndarray, f: np.ndarray) -> float:
    """Relative systematic error: positive means over-forecasting."""
    return float((f - y).sum() / y.sum())


def rmsse(y: np.ndarray, f: np.ndarray, history: np.ndarray) -> float:
    """Root mean squared scaled error of one series; scale is the in-sample one-step naive MSE."""
    scale = np.mean(np.diff(history) ** 2)
    if scale == 0:
        return np.nan
    return float(np.sqrt(np.mean((y - f) ** 2) / scale))


def block_sums(a: np.ndarray, size: int = 7) -> np.ndarray:
    """(items, days) -> (items, days // size) sums, e.g. daily to weekly demand per item."""
    n, days = a.shape
    return a.reshape(n, days // size, size).sum(axis=2)


def pinball(y: np.ndarray, q_forecast: np.ndarray, q: float) -> float:
    """Mean pinball (quantile) loss normalised by mean demand, so it is comparable across series."""
    diff = y - q_forecast
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)) / y.mean()) if y.mean() > 0 else np.nan


def coverage(y: np.ndarray, q_forecast: np.ndarray) -> float:
    """Share of observations at or below the quantile forecast; should equal q if calibrated."""
    return float(np.mean(y <= q_forecast))
