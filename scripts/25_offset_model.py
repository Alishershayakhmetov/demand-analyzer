"""Fix for the scale-normalised model on tiny volumes: counts as targets with log(level) as an offset (exposure) instead of ratio targets.
Leave-one-dataset-out, same data and windows as scripts/20 (no Chronos). Declared before the run: offset-multi is a success if on the bakery and on M5
its weekly WAPE is within +10% of the best simple baseline and its bias is within +-30% (norm-multi had +347% and +60%), and on the other four datasets
its mean weekly WAPE is not worse than norm-multi's."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
spec = importlib.util.spec_from_file_location("lodo", Path(__file__).with_name("20_universal_lodo.py"))
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
from demand import baselines as B  # noqa: E402
from demand import config as C  # noqa: E402
from demand import metrics as M  # noqa: E402
from demand import universal as U  # noqa: E402

shops = L.load_all()
frames = {k: U.training_frame(v) for k, v in shops.items()}
early = U.training_frame(shops["m5_CA_1"], last_origin=C.TEST_CUTOFF - 112)
norm_m5, off_m5 = L.fit(early), U.fit_offset(early)
rows = []
for held in L.EVAL:
    shop, T = shops[held], shops[held].Y.shape[1]
    group = [k for k in frames if k != held and not (held == "m5_CA_1" and k.startswith("m5_"))]
    big = pd.concat([frames[k] for k in group], ignore_index=True)
    norm_multi, off_multi = L.fit(big), U.fit_offset(big)
    per = {}
    for k in range(3):
        o = T - 28 * (k + 1)
        if o < U.MIN_HISTORY:
            continue
        hist, first, act = shop.Y[:, :o], np.minimum(shop.first_idx, o), shop.Y[:, o:o + 28]
        f = {"seasonal naive": B.seasonal_naive(hist), "28-day average": B.moving_average(hist), "weekday average 8w": B.weekday_mean(hist),
             "norm-multi (ratio)": U.predict_normalised(norm_multi, hist, first), "offset-M5": U.predict_offset(off_m5, hist, first), "offset-multi": U.predict_offset(off_multi, hist, first)}
        for m, fc in f.items():
            per.setdefault(m, []).append((M.wape(M.block_sums(act), M.block_sums(fc)), M.bias(act, fc)))
    for m, v in per.items():
        a = np.array(v); rows.append(dict(dataset=held, method=m, weekly=a[:, 0].mean(), bias=a[:, 1].mean()))
    print(held, "done", flush=True)
r = pd.DataFrame(rows)
w = r.pivot(index="method", columns="dataset", values="weekly")[L.EVAL]; b = r.pivot(index="method", columns="dataset", values="bias")[L.EVAL]
base = w.loc[["seasonal naive", "28-day average", "weekday average 8w"]].min()
ok = all((w.loc["offset-multi", d] <= base[d] * 1.10) and abs(b.loc["offset-multi", d]) <= 0.30 for d in ["bakery_daily", "m5_CA_1"])
others = [d for d in L.EVAL if d not in ("bakery_daily", "m5_CA_1")]
ok2 = w.loc["offset-multi", others].mean() <= w.loc["norm-multi (ratio)", others].mean()
text = "Weekly WAPE\n" + w.round(3).to_string() + "\n\nBias\n" + b.round(3).to_string() + f"\n\nbest simple baseline: {base.round(3).to_dict()}\nrule 1 (bakery and M5 within +10%, |bias|<=30%): {'PASSED' if ok else 'NOT PASSED'}\nrule 2 (other four: mean WAPE not worse than ratio model): {'PASSED' if ok2 else 'NOT PASSED'}"
print(text); (C.REPORTS / "offset_model.txt").write_text(text, encoding="utf-8")
