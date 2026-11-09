"""Train the scale-normalised LightGBM used as a candidate by the selector (M5 CA_1, all days up to d1941). Writes models/norm_model.txt."""
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
frame = U.training_frame(shop)
model = lgb.LGBMRegressor(**U.PARAMS).fit(frame[U.FEATURES], frame.y)
model.booster_.save_model(str(S.MODEL_DIR / "norm_model.txt"))
print("saved", len(frame), "rows")
