import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demand import webapi  # noqa: E402


def make_csv(sep: str = ";", end: str = "2024-03-13") -> str:
    rng = np.random.default_rng(1)
    dates = pd.date_range(end=end, periods=200)
    rows = [(d.strftime("%Y-%m-%d"), "milk", int(rng.poisson(8))) for d in dates] + [(d.strftime("%Y-%m-%d"), "bread", int(rng.poisson(3))) for d in dates] \
        + [(d.strftime("%Y-%m-%d"), "new", int(rng.poisson(2))) for d in dates[-10:]]
    return pd.DataFrame(rows, columns=["Дата", "SKU", "Qty"]).to_csv(index=False, sep=sep)


class CsvTest(unittest.TestCase):
    def test_delimiter_is_detected(self):
        for sep in (",", ";", "\t"):
            df = webapi.read_csv_text(make_csv(sep))
            self.assertEqual(list(df.columns), ["Дата", "SKU", "Qty"])
            self.assertEqual(len(df), 410)

    def test_bom_is_ignored(self):
        df = webapi.read_csv_text("﻿date,item,sales\n2024-01-01,a,1\n")
        self.assertEqual(list(df.columns), ["date", "item", "sales"])


@unittest.skipUnless(webapi.model_available(), "needs models/ from scripts/14_train_final.py")
class AnalyzeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = dict(source="csv", csv=make_csv(), cols=dict(date="Дата", item="SKU", sales="Qty"))
        cls.res = webapi.analyze(cls.payload)

    def test_structure_and_json(self):
        r = self.res
        json.dumps(r)  # must be serialisable as is
        self.assertEqual(r["n_items"], 3)
        self.assertEqual(r["forecast_from"], "2024-03-11")  # history was cut at Sunday 2024-03-10
        self.assertTrue(any("last Sunday" in n for n in r["notes"]))
        self.assertEqual(r["n_new"], 1)
        it = r["items"][0]
        self.assertEqual(it["id"], "milk")  # busiest first
        self.assertEqual((len(it["hist"]), len(it["fc"]), len(it["past"]), len(it["ahead"])), (84, 28, 4, 4))

    def test_plan_responds_to_parameters(self):
        a = webapi.plan(self.res["token"], 3, 95)
        b = webapi.plan(self.res["token"], 7, 98)
        self.assertGreater(b["units"], a["units"])
        self.assertGreater(b["z"], a["z"])
        self.assertEqual(len(a["rows"]), 3)
        json.dumps(a)
        csv = webapi.plan_csv(self.res["token"], 3, 95).decode("utf-8")
        self.assertTrue(csv.startswith("item_id,status"))

    def test_stock_csv_is_used(self):
        p = dict(self.payload, stock_csv="item_id,on_hand,on_order\nmilk,1000,0\nbread,0,0\n")
        r = webapi.analyze(p)
        plan = webapi.plan(r["token"], 3, 95)
        qty = {row["item_id"]: row["order_qty"] for row in plan["rows"]}
        self.assertEqual(qty["milk"], 0)  # a thousand units on hand cover the target level
        self.assertGreater(qty["bread"], 0)

    def test_user_errors(self):
        with self.assertRaises(webapi.UserError):
            webapi.analyze(dict(source="csv", csv=make_csv(), cols=dict(date="nope", item="SKU", sales="Qty")))
        with self.assertRaises(webapi.UserError):
            webapi.plan(self.res["token"], 5, 95)
        with self.assertRaises(webapi.UserError):
            webapi.plan("unknown-token", 3, 95)
        with self.assertRaises(webapi.UserError):
            webapi.analyze(dict(source="csv", csv="a,b,c\n2024-01-01,x,1\n", cols=dict(date="a", item="b", sales="c")))


if __name__ == "__main__":
    unittest.main()
