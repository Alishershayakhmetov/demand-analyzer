"""Logic behind the local web app (app/server.py): analyse a sales history, return the forecast and the purchase plan.

No web framework is involved; the functions take and return plain dicts so they can be tested directly.
"""
import io
import threading
import uuid
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as C
from . import serving as S

_LOCK = threading.Lock()
_STATE: dict[str, dict] = {}
_MAX_STATES = 6
DEMO_COLS = dict(date_col="date", item_col="item_id", sales_col="sales", price_col="sell_price")


class UserError(ValueError):
    """Problem with what the user sent; its message is shown in the interface."""


@lru_cache(maxsize=1)
def _model():
    return S.load_model()


@lru_cache(maxsize=1)
def _demo_table() -> pd.DataFrame:
    df = pd.read_parquet(C.PROC / f"panel_{C.STORE}.parquet", columns=["item_id", "date", "d", "sales", "sell_price"])
    return df[df.d <= C.TEST_CUTOFF].drop(columns="d")  # nothing after the model's training window is shown


def demo_available() -> bool:
    return (C.PROC / f"panel_{C.STORE}.parquet").exists()


def model_available() -> bool:
    return (S.MODEL_DIR / "lean_model.txt").exists() and (S.MODEL_DIR / "meta.json").exists()


def read_csv_text(text: str) -> pd.DataFrame:
    """Read an uploaded CSV; the delimiter (comma, semicolon or tab) is taken from the header line."""
    text = text.lstrip("﻿")
    header = text.split("\n", 1)[0]
    sep = max([",", ";", "\t"], key=header.count)
    try:
        return pd.read_csv(io.StringIO(text), sep=sep)
    except Exception as e:
        raise UserError(f"cannot read the CSV: {e}") from e


def meta() -> dict:
    out = dict(model=model_available(), demo=demo_available(), name="Demand Copilot")
    res = C.REPORTS / "feature_selection_results.csv"
    if res.exists():
        r = pd.read_csv(res)
        r = r[r.split.str.startswith("validation")].drop(columns=["split", "weekly_std"]).round(4)
        out["quality_val"] = r.to_dict("records")
    tst = C.REPORTS / "test_point_metrics.csv"
    if tst.exists():
        t = pd.read_csv(tst)
        keep = ["seasonal naive", "28-day average", "weekday average, 8 weeks", "AutoETS", "LightGBM selected 11 (FINAL, stored model)"]
        out["quality_test"] = t[t.model.isin(keep)][["model", "daily_wape", "weekly_wape", "wape_28d"]].round(4).to_dict("records")
    out["frontier_image"] = (C.REPORTS / "figures" / "inventory_frontier.png").exists()
    return out


def _stock(payload: dict, shop: S.Shop) -> tuple[np.ndarray | None, np.ndarray | None]:
    n = len(shop.ids)
    if payload.get("stock_csv"):
        df = read_csv_text(payload["stock_csv"])
        if not {"item_id", "on_hand"} <= set(df.columns):
            raise UserError("the stock CSV needs the columns item_id and on_hand (on_order is optional)")
        s = df.set_index(df.item_id.astype(str))
        on_hand = pd.to_numeric(s.on_hand, errors="coerce").reindex(shop.ids).fillna(0).to_numpy(float)
        on_order = pd.to_numeric(s.on_order, errors="coerce").reindex(shop.ids).fillna(0).to_numpy(float) if "on_order" in s.columns else np.zeros(n)
        return on_hand, on_order
    if payload.get("synthetic_stock"):
        rng = np.random.default_rng(0)
        return np.round(shop.Y[:, -28:].mean(axis=1) * rng.uniform(1, 5, n)), np.zeros(n)
    return None, None


def analyze(payload: dict) -> dict:
    """payload: source 'demo' | 'csv'; for csv also csv text and cols {date, item, sales, price}; optional stock_csv or synthetic_stock."""
    if not model_available():
        raise UserError("no_model")
    if payload.get("source") == "demo":
        if not demo_available():
            raise UserError("no_demo")
        df, cols = _demo_table(), DEMO_COLS
    else:
        df = read_csv_text(payload.get("csv") or "")
        c = payload.get("cols") or {}
        for key in ("date", "item", "sales"):
            if c.get(key) not in df.columns:
                raise UserError(f"column for '{key}' is not in the file")
        cols = dict(date_col=c["date"], item_col=c["item"], sales_col=c["sales"], price_col=c.get("price") if c.get("price") in df.columns else None)
    try:
        shop = S.to_shop(df, **cols)
    except ValueError as e:
        raise UserError(str(e)) from e
    booster, meta_ = _model()
    f = S.forecast(shop, booster, meta_["features"])
    on_hand, on_order = _stock(payload, shop)
    status = S.item_status(shop)
    n = len(shop.ids)
    order = np.argsort(-shop.Y[:, -28:].sum(axis=1), kind="stable")
    first = shop.dates[-1] + pd.Timedelta(days=1)
    items = []
    for i in order:
        items.append(dict(id=shop.ids[i], status=str(status[i]), hist=[int(v) if float(v).is_integer() else round(float(v), 2) for v in shop.Y[i, -84:]],
                          fc=[round(float(v), 2) for v in f[i]],
                          past=[round(float(v), 1) for v in shop.Y[i, -28:].reshape(4, 7).sum(axis=1)],
                          ahead=[round(float(v), 1) for v in f[i].reshape(4, 7).sum(axis=1)]))
    token = uuid.uuid4().hex
    with _LOCK:
        while len(_STATE) >= _MAX_STATES:
            _STATE.pop(next(iter(_STATE)))
        _STATE[token] = dict(shop=shop, f=f, on_hand=on_hand, on_order=on_order, calibration=meta_["calibration"])
    return dict(token=token, notes=shop.notes, n_items=n, n_days=len(shop.dates), hist_start=str(shop.dates[-84].date()), forecast_from=str(first.date()),
                forecast_to=str((first + pd.Timedelta(days=C.HORIZON - 1)).date()), next7=float(f[:, :7].sum()), last7=float(shop.Y[:, -7:].sum()),
                n_new=int((status == "new").sum()), n_dormant=int((status == "dormant").sum()), has_stock=on_hand is not None,
                has_price=shop.price is not None, items=items)


def _state(token: str) -> dict:
    with _LOCK:
        st = _STATE.get(token)
    if st is None:
        raise UserError("session expired, calculate again")
    return st


def plan_frame(token: str, lead: int, service: int) -> tuple[pd.DataFrame, float]:
    if lead not in (1, 3, 7) or service not in (90, 95, 98):
        raise UserError("lead time must be 1, 3 or 7 days and service level 90, 95 or 98")
    st = _state(token)
    z = S.z_for_service(st["calibration"], lead, service / 100)
    return S.purchase_plan(st["shop"], st["f"], lead, z, st["on_hand"], st["on_order"]), z


def plan(token: str, lead: int, service: int) -> dict:
    df, z = plan_frame(token, lead, service)
    sort_col = "order_value" if "order_value" in df else "order_qty"
    rows = df.sort_values(sort_col, ascending=False, kind="stable")
    return dict(z=round(z, 3), n_order=int((df.order_qty > 0).sum()), units=int(df.order_qty.sum()),
                value=float(df.order_value.sum()) if "order_value" in df else None, flagged=int((df.status != "ok").sum()),
                columns=list(df.columns), rows=rows.astype(object).where(rows.notna(), None).to_dict("records"))


def plan_csv(token: str, lead: int, service: int) -> bytes:
    df, _ = plan_frame(token, lead, service)
    return df.to_csv(index=False).encode("utf-8")
