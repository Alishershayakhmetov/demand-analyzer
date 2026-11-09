"""Does the file-specific z hit the target fill rate better than the M5-calibrated z? End to end on the open datasets: the selector forecasts the
window, z is taken from the backtest before it (file) or from the M5 table, and the order-up-to simulator is run on the window that really followed.
Declared before the run: file z is better if the mean absolute gap between realised and target fill (95%, lead 3 d, three windows per dataset) is lower
than with the M5 z, and its inventory is at most 10% higher on average."""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demand import config as C  # noqa: E402
from demand import selector as SEL  # noqa: E402
from demand import service_level as SL  # noqa: E402
from demand import serving as S  # noqa: E402
from demand import webapi  # noqa: E402
from demand.inventory import extend, interval_sums, simulate  # noqa: E402

booster, meta = S.load_model()
limit = meta["level_range"]["mean28_p999"]
norm = lgb.Booster(model_file=str(S.MODEL_DIR / "norm_model.txt")); weekly = lgb.Booster(model_file=str(S.MODEL_DIR / "weekly_model.txt"))
LEAD, TARGET = 3, 0.95
rows = []
for name, c in webapi.EXAMPLES.items():
    df = webapi.read_csv_text((C.ROOT / "examples" / f"{name}.csv").read_text(encoding="utf-8"))
    cols = dict(date_col=c["date"], item_col=c["item"], sales_col=c["sales"], price_col=None)
    full = S.to_shop(df, **cols)
    d = pd.to_datetime(df[c["date"]], errors="coerce").dt.normalize()
    for k in range(3):
        cut = full.dates[-1] - pd.Timedelta(days=28 * (k + 1))
        try:
            hist = S.to_shop(df[d <= cut], **cols)
        except ValueError:
            continue
        ids = [i for i in hist.ids if i in set(full.ids)]
        fi = {i: j for j, i in enumerate(full.ids)}; hi = {i: j for j, i in enumerate(hist.ids)}
        pos = full.dates.get_loc(cut) + 1
        actual = np.stack([full.Y[fi[i], pos:pos + 28] for i in ids])
        sub = S.Shop(ids, hist.dates, np.stack([hist.Y[hi[i]] for i in ids]), None, np.array([hist.first_idx[hi[i]] for i in ids]))
        f, _, info = SEL.select(sub, booster, meta["features"], limit, norm, weekly, True)
        z_m5 = S.z_for_service(meta["calibration"], LEAD, TARGET)
        z_file = SL.file_z(sub.Y, info["_bt"], LEAD, TARGET) if info.get("_bt") else None
        T = sub.Y.shape[1]; hh = sub.Y; active = hh[:, -28:].sum(axis=1) > 0
        for label, z in [("M5 z", z_m5), ("file z", z_file if z_file is not None else z_m5)]:
            sig = np.tile(hh[:, -91:].std(axis=1) * np.sqrt(LEAD + 7), (4, 1))
            s, st = simulate(actual, interval_sums(extend(f), LEAD), sig, np.array([z]), LEAD, np.ceil(hh[:, -28:].mean(axis=1) * LEAD))
            dem = actual[active][:, LEAD:].sum()
            rows.append(dict(dataset=name, window=k, variant=label, z=z, fill=s[0, active].sum() / dem, inv=st[0, active].sum() / dem, file_z_used=z_file is not None))
r = pd.DataFrame(rows)
r["gap"] = (r.fill - TARGET).abs()
agg = r.groupby("variant")[["gap", "fill", "inv", "z"]].mean()
per = r.groupby(["dataset", "variant"])[["fill", "inv", "z"]].mean().round(3)
ok = agg.loc["file z", "gap"] < agg.loc["M5 z", "gap"] and agg.loc["file z", "inv"] <= 1.10 * agg.loc["M5 z", "inv"]
text = per.to_string() + "\n\n" + agg.round(3).to_string() + f"\n\nfile z better (smaller gap, inventory <= +10%): {'YES' if ok else 'NO'}"
print(text); (C.REPORTS / "service_level_check.txt").write_text(text, encoding="utf-8")
