"""Train the model that the app serves and calibrate the safety factor z.

- final model: the selected 11-feature LightGBM trained with forecast origins up to d1913, i.e. on days <= d1941
  (labels of the test fold d1942..d1969 are not used);
- calibration: for lead times 1, 3, 7 the fill rate reached for each z on the dev + validation folds, using models trained
  per fold (same weekly-refresh simulator as scripts/10_inventory.py). The app maps a target service level to z with it.
Writes models/lean_model.txt, models/meta.json.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demand import config as C  # noqa: E402
from demand.evaluate import load_matrices  # noqa: E402
from demand.features import FEATURES, build_rows, training_rows  # noqa: E402
from demand.modeling import Z, fit, inventory_frontier, level_range, predict_origins  # noqa: E402
from demand.serving import forecast, to_shop  # noqa: E402

LEADS = [1, 3, 7]
FOLDS = C.DEV_CUTOFFS + [C.VAL_CUTOFF]


def main() -> None:
    selected = json.loads((C.REPORTS / "selected_features.json").read_text())["features"]
    feats = [f for f in FEATURES if f in selected]
    m = load_matrices()
    n = len(m.ids)
    t0 = time.time()

    rolls = {}
    for cutoff in FOLDS:
        origins = [build_rows(m, cutoff + 7 * k, with_target=False) for k in range(4)]
        rolls[cutoff] = predict_origins(fit(training_rows(m, cutoff), feats), origins, feats, n)
        print(f"calibration fold {cutoff} done", flush=True)
    calibration = {}
    for lead in LEADS:
        fill, inv = inventory_frontier(m, rolls, FOLDS, lead)
        calibration[str(lead)] = dict(z=Z.tolist(), fill=fill.tolist(), inv_days=inv.tolist())
        print(f"lead {lead}: fill at z=0 {np.interp(0, Z, fill):.3f}, z=1 {np.interp(1, Z, fill):.3f}, z=2 {np.interp(2, Z, fill):.3f}")

    model = fit(training_rows(m, C.TEST_CUTOFF), feats)
    (C.ROOT / "models").mkdir(exist_ok=True)
    model.booster_.save_model(str(C.ROOT / "models" / "lean_model.txt"))
    meta = dict(features=feats, trained_through_day=C.TEST_CUTOFF, trained_through_date="2016-05-22",
                data="M5 store CA_1, 300 items", calibration=calibration, level_range=level_range(m))
    meta["sha256"] = hashlib.sha256((C.ROOT / "models" / "lean_model.txt").read_bytes()).hexdigest()
    (C.ROOT / "models" / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    # Forecast of the untouched test window (d1942..d1969), written before that fold is ever scored.
    panel = pd.read_parquet(C.PROC / f"panel_{C.STORE}.parquet", columns=["item_id", "date", "d", "sales", "sell_price"])
    shop = to_shop(panel[panel.d <= C.TEST_CUTOFF], "date", "item_id", "sales", "sell_price")
    f = forecast(shop, model.booster_, feats)
    pd.DataFrame(f, index=pd.Index(shop.ids, name="item_id"), columns=range(1, C.HORIZON + 1)).round(4).to_csv(
        C.REPORTS / "test_window_forecast.csv")
    print(f"saved model ({len(feats)} features) in {time.time() - t0:.0f}s; sha256 {meta['sha256'][:12]}")


if __name__ == "__main__":
    main()
