"""Periodic-review, order-up-to inventory simulator (lost sales, integer units, fixed lead time).

An order is placed at the start of every review day r, arrives L days later, and brings the inventory position
(on hand + on order) up to S_r = mean_r + z * sigma_r, where mean_r is the forecast demand over the protection
interval [r, r + L + R) and sigma_r its standard deviation. Sweeping z traces an inventory-service frontier.
"""
import numpy as np

REVIEW = 7


def simulate(demand: np.ndarray, mean_p: np.ndarray, sigma_p: np.ndarray, z: np.ndarray, lead: int,
             init: np.ndarray, review: int = REVIEW) -> tuple[np.ndarray, np.ndarray]:
    """Run one window for all items and all z at once.

    demand: (items, T) actual demand. mean_p, sigma_p: (reviews, items), row r belongs to review day r * review.
    Returns per item and z: units sold and the sum of end-of-day on-hand units, both over days t >= lead only
    (the first order arrives on day `lead`, so earlier days are identical for every policy and are left out).
    """
    n, T = demand.shape
    nz = len(z)
    on_hand = np.tile(init.astype(float), (nz, 1))
    pipeline = np.zeros((nz, n, T + lead + 1))
    sold = np.zeros((nz, n))
    stock = np.zeros((nz, n))
    for t in range(T):
        on_hand += pipeline[:, :, t]
        if t % review == 0:
            r = t // review
            target = mean_p[r][None] + z[:, None] * sigma_p[r][None]
            position = on_hand + pipeline[:, :, t + 1:].sum(axis=2)
            pipeline[:, :, t + lead] += np.ceil(np.maximum(target - position, 0))
        sales = np.minimum(on_hand, demand[:, t][None])
        on_hand -= sales
        if t >= lead:
            sold += sales
            stock += on_hand
    return sold, stock


def extend(f: np.ndarray, extra: int = 7) -> np.ndarray:
    """Append a repeat of the last `extra` days so the last review cycle has a full protection interval."""
    return np.concatenate([f, f[:, -extra:]], axis=1)


def interval_sums(daily: np.ndarray, lead: int, review: int = REVIEW, n_reviews: int = 4) -> np.ndarray:
    """(items, days) -> (reviews, items) sums over [r*review, r*review + lead + review)."""
    return np.stack([daily[:, r * review: r * review + lead + review].sum(axis=1) for r in range(n_reviews)])


def interval_rss(daily_sigma: np.ndarray, lead: int, review: int = REVIEW, n_reviews: int = 4) -> np.ndarray:
    """Same windows, combining independent daily standard deviations as sqrt(sum of squares)."""
    return np.sqrt(interval_sums(daily_sigma ** 2, lead, review, n_reviews))


def inventory_at_fill(fill: np.ndarray, inv_days: np.ndarray, target: float) -> float:
    """Inventory (days of sales) needed for a fill rate `target`, interpolated on the frontier; NaN outside it."""
    order = np.argsort(inv_days)
    f = np.maximum.accumulate(fill[order])
    x = inv_days[order]
    if target > f[-1] or target < f[0]:
        return float("nan")
    return float(np.interp(target, f, x))
