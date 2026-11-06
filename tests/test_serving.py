import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demand import config as C  # noqa: E402
from demand.serving import Shop, build_features, item_status, purchase_plan, to_shop, z_for_service  # noqa: E402


def long_table(days: int, end: str, items=("A", "B"), rate=3.0, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=end, periods=days)
    return pd.DataFrame([(d, i, rng.poisson(rate), 2.0) for i in items for d in dates], columns=["date", "item", "units", "price"])


class CleaningTest(unittest.TestCase):
    def test_history_is_cut_at_last_sunday(self):
        shop = to_shop(long_table(200, "2024-03-13"), "date", "item", "units")  # a Wednesday
        self.assertEqual(shop.dates[-1], pd.Timestamp("2024-03-10"))
        self.assertEqual(shop.dates[-1].weekday(), 6)
        self.assertTrue(any("last Sunday" in n for n in shop.notes))

    def test_sunday_end_is_kept_and_odd_rows_are_reported(self):
        df = long_table(120, "2024-03-10")
        df.loc[0, "units"] = -4
        df = pd.concat([df, df.iloc[[5]]], ignore_index=True)
        df.loc[1, "date"] = "not a date"
        shop = to_shop(df, "date", "item", "units", "price")
        self.assertEqual(shop.dates[-1], pd.Timestamp("2024-03-10"))
        notes = " ".join(shop.notes)
        self.assertIn("unreadable", notes)
        self.assertIn("negative", notes)
        self.assertIn("duplicate", notes)
        self.assertEqual(shop.Y.shape, (2, 120))
        self.assertTrue((shop.Y >= 0).all())

    def test_missing_days_are_zero_and_short_history_is_rejected(self):
        df = long_table(120, "2024-03-10")
        df = df[~((df.item == "A") & (df.date.dt.day == 5))]
        shop = to_shop(df, "date", "item", "units")
        self.assertTrue((shop.Y[0][shop.dates.day == 5] == 0).all())
        with self.assertRaises(ValueError):
            to_shop(long_table(60, "2024-03-10"), "date", "item", "units")

    def test_status_flags(self):
        T = 120
        Y = np.zeros((3, T))
        Y[0] = 2                      # ok
        Y[1, -10:] = 5                # new: first sale 10 days ago
        Y[2, :60] = 1                 # dormant: nothing in the last 28 days
        first = np.array([0, T - 10, 0])
        shop = Shop(["a", "b", "c"], pd.date_range(end="2024-03-10", periods=T), Y, None, first)
        self.assertEqual(list(item_status(shop)), ["ok", "new", "dormant"])


class PlanTest(unittest.TestCase):
    def test_order_up_to_arithmetic(self):
        T = 120
        Y = np.tile(np.array([[4.0], [0.0]]), (1, T))          # item a: constant 4/day (sigma 0); item b: dormant
        shop = Shop(["a", "b"], pd.date_range(end="2024-03-10", periods=T), Y, np.array([2.0, 2.0]), np.array([0, T]))
        f = np.full((2, C.HORIZON), 4.0)
        plan = purchase_plan(shop, f, lead=3, z=1.0, on_hand=np.array([10.0, 0.0]), on_order=np.array([6.0, 0.0]))
        # lead + 7 = 10 days * 4 = 40, safety 0, minus 10 on hand and 6 on order
        self.assertEqual(plan.order_qty[0], 24)
        self.assertEqual(plan.order_qty[1], 0)
        self.assertEqual(plan.order_value[0], 48.0)

    def test_service_level_to_z(self):
        cal = {"3": {"z": [0.0, 1.0, 2.0], "fill": [0.8, 0.9, 0.99]}}
        self.assertAlmostEqual(z_for_service(cal, 3, 0.9), 1.0)
        self.assertAlmostEqual(z_for_service(cal, 3, 0.945), 1.5, places=2)


@unittest.skipUnless((C.PROC / f"panel_{C.STORE}.parquet").exists(), "needs data/processed from scripts/02_prepare.py")
class TrainServeConsistencyTest(unittest.TestCase):
    def test_features_match_training_features(self):
        from demand.evaluate import load_matrices
        from demand.features import build_rows
        m = load_matrices()
        cutoff = C.VAL_CUTOFF
        panel = pd.read_parquet(C.PROC / f"panel_{C.STORE}.parquet", columns=["item_id", "date", "d", "sales", "sell_price"])
        panel = panel[panel.d <= cutoff]
        shop = to_shop(panel, "date", "item_id", "sales", "sell_price")
        self.assertEqual(shop.ids, m.ids)
        served = build_features(shop)
        trained = build_rows(m, cutoff, with_target=False)
        cols = [c for c in served.columns if c in trained.columns and c not in ("age", "origin") and not c.startswith(("price", "on_sale"))
                and c not in ("snap", "wday", "month", "dom", "event", "dept", "cat", "item")]
        self.assertIn("mean14", cols)
        self.assertIn("days_since_sale", cols)
        np.testing.assert_allclose(served[cols].to_numpy(float), trained[cols].to_numpy(float), rtol=1e-9, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
