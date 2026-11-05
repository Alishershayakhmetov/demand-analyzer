import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demand.inventory import extend, interval_rss, interval_sums, inventory_at_fill, simulate  # noqa: E402


class SimulatorTest(unittest.TestCase):
    def test_hand_computed_example(self):
        demand = np.full((1, 4), 2.0)
        sold, stock = simulate(demand, np.array([[8.0]]), np.array([[0.0]]), np.array([0.0]), lead=1, init=np.array([2]))
        # t=0 orders 6 (arrive t=1); from t=1: sold 2+2+2, end-of-day stock 4+2+0... plus t=3 -> 4, 2, 0
        self.assertEqual(sold[0, 0], 6)
        self.assertEqual(stock[0, 0], 6)

    def test_no_forecast_no_stock_means_no_sales(self):
        demand = np.full((2, 28), 3.0)
        sold, stock = simulate(demand, np.zeros((4, 2)), np.zeros((4, 2)), np.array([0.0, 2.0]), lead=3, init=np.zeros(2))
        self.assertEqual(sold.sum(), 0)
        self.assertEqual(stock.sum(), 0)

    def test_constant_demand_perfect_forecast_is_fully_served(self):
        demand = np.full((1, 28), 5.0)
        mean_p = np.full((4, 1), 5.0 * (3 + 7))
        sold, _ = simulate(demand, mean_p, np.zeros((4, 1)), np.array([0.0]), lead=3, init=np.array([15]))
        self.assertEqual(sold[0, 0], 5.0 * (28 - 3))

    def test_more_safety_never_sells_less_and_holds_more(self):
        rng = np.random.default_rng(0)
        demand = rng.poisson(4, size=(50, 28)).astype(float)
        mean_p = np.tile(demand.mean(axis=1) * 10, (4, 1))
        sigma_p = np.tile(demand.std(axis=1) * np.sqrt(10), (4, 1))
        z = np.linspace(-1, 4, 11)
        sold, stock = simulate(demand, mean_p, sigma_p, z, lead=3, init=np.ceil(demand.mean(axis=1) * 3))
        self.assertTrue((np.diff(sold.sum(axis=1)) >= 0).all())
        self.assertTrue((np.diff(stock.sum(axis=1)) >= 0).all())
        self.assertTrue((sold <= demand[:, 3:].sum(axis=1)[None] + 1e-9).all())

    def test_helpers(self):
        f = np.arange(28, dtype=float)[None]
        self.assertEqual(extend(f).shape, (1, 35))
        self.assertEqual(interval_sums(extend(f), 3)[0, 0], f[0, :10].sum())
        self.assertEqual(interval_rss(np.full((1, 35), 2.0), 3)[0, 0], np.sqrt(10 * 4))
        fill, inv = np.array([0.5, 0.9, 1.0]), np.array([1.0, 3.0, 5.0])
        self.assertAlmostEqual(inventory_at_fill(fill, inv, 0.95), 4.0)
        self.assertTrue(np.isnan(inventory_at_fill(fill, inv, 1.2)))


if __name__ == "__main__":
    unittest.main()
