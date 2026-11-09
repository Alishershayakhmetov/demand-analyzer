"""Train the weekly-sum LightGBM (M5 CA_1, all days up to d1941). Writes models/weekly_model.txt."""
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demand import config as C  # noqa: E402
from demand import serving as S  # noqa: E402
from demand import universal as U  # noqa: E402

p = pd.read_parquet(C.PROC / f"panel_{C.STORE}.parquet", columns=["item_id", "date", "d", "sales"])
shop = S.to_shop(p[p.d <= C.TEST_CUTOFF], "date", "item_id", "sales")
frame = U.week_training_frame(shop)
model = lgb.LGBMRegressor(**{**U.PARAMS, "min_child_samples": 50}).fit(frame[U.WEEK_FEATURES], frame.y)
model.booster_.save_model(str(S.MODEL_DIR / "weekly_model.txt"))
print("saved", len(frame), "rows")
