"""Demand Copilot: upload a sales history, get a 28-day forecast and a weekly purchase plan.

Run from the project root:  .venv/Scripts/streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demand import config as C  # noqa: E402
from demand import serving as S  # noqa: E402

TXT = {
    "ru": {
        "title": "Demand Copilot: прогноз спроса и план закупок",
        "intro": "Загрузите историю продаж магазина (по строке на товар и день) и получите прогноз на 28 дней и недельный план закупок.",
        "lang": "Язык", "source": "Данные", "demo": "Демо: магазин M5 CA_1 (300 товаров)", "upload": "Загрузить CSV",
        "file": "CSV с продажами", "col_date": "Колонка с датой", "col_item": "Колонка с товаром", "col_sales": "Колонка с продажами (шт.)",
        "col_price": "Колонка с ценой (необязательно)", "none": "— нет —", "stock_file": "CSV с остатками (item_id, on_hand, on_order), необязательно",
        "demo_stock": "Синтетические остатки для демо (случайно, 1–5 дней продаж)", "params": "Параметры закупок",
        "lead": "Срок поставки, дней", "service": "Целевой уровень обслуживания, %",
        "service_help": "Доля выручки, которая обслуживается из наличия. Пересчёт в запас калиброван на бэктестах M5; на ваших данных результат может отличаться.",
        "review": "Заказ делается раз в 7 дней.", "no_data": "Выберите демо или загрузите CSV, чтобы начать.",
        "no_model": "Модель не найдена. Выполните `python scripts/14_train_final.py` (см. README).",
        "no_demo": "Демо-данные не найдены. Выполните `python scripts/01_download.py` и `python scripts/02_prepare.py`.",
        "bad_input": "Не удалось обработать данные:", "tab_fc": "Прогноз", "tab_plan": "План закупок", "tab_quality": "Качество и ограничения",
        "data_check": "Проверка данных", "items": "Товаров", "history": "Истории, дней", "start": "Прогноз с", "end": "по",
        "next7": "Прогноз на 7 дней, шт.", "last7": "Продано за последние 7 дней, шт.", "pick": "Товар", "daily": "Прогноз по дням",
        "weekly": "Прогноз по неделям", "week": "Неделя", "units": "шт.", "hist_series": "история", "fc_series": "прогноз",
        "n_order": "Товаров к заказу", "units_order": "Штук к заказу", "value_order": "Сумма заказа", "flags": "Товары с пометками",
        "only_order": "Показывать только товары к заказу", "download": "Скачать план (CSV)",
        "plan_note": "Целевой уровень = прогноз на срок поставки + 7 дней + запас. Заказ = целевой уровень − остаток − уже заказанное. "
                     "Остатки не загружены, поэтому приняты равными нулю." ,
        "plan_note_stock": "Целевой уровень = прогноз на срок поставки + 7 дней + запас. Заказ = целевой уровень − остаток − уже заказанное.",
        "st_ok": "ок", "st_new": "новый (<28 дней)", "st_dormant": "нет продаж 28 дней",
        "cols": {"item_id": "Товар", "status": "Статус", "forecast_next_7d": "Прогноз 7 дн.", "target_level": "Целевой уровень",
                 "safety_stock": "Страх. запас", "on_hand": "Остаток", "on_order": "В пути", "order_qty": "Заказать", "order_value": "Сумма"},
        "chart_note": "Линия справа: ожидаемый (средний) спрос в день, поэтому значения дробные: 0,6 значит «в среднем 6 продаж за 10 дней». "
                      "Срок поставки и уровень обслуживания на этот график не влияют: они меняют только рекомендацию ниже и вкладку «План закупок».",
        "item_plan": "Рекомендация по товару (срок поставки {lead} дн., обслуживание {service}%)",
        "m_target": "Целевой уровень, шт.", "m_onhand": "Остаток, шт.", "m_onorder": "В пути, шт.", "m_order": "Заказать, шт.",
        "week_actual": "факт", "week_forecast": "прогноз", "kind": "Тип",
        "flat_note": "Почему недели почти одинаковы: модель прогнозирует текущий уровень продаж и привычный рисунок по дням недели. "
                     "Тренда, праздников, акций и цен она не знает (на проверках такие признаки не улучшали точность), поэтому прогноз "
                     "меняется от недели к неделе лишь на 1–2% в среднем, а реальные недельные продажи колеблются на десятки процентов. "
                     "Новые продажи учитываются при следующей загрузке данных: прогноз пересчитывается каждую неделю.",
        "quality_note": "Таблицы ниже посчитаны заранее на исторических окнах M5 (не на ваших данных).",
        "quality_tbl": "Проверка выбранной модели на фолде d1913 (после выбора признаков по dev-фолдам)",
        "test_tbl": "Итоговое тестовое окно d1942–1969 (оценено один раз; ошибка выше, чем на прежних окнах)",
        "limits": "#### Ограничения и риски\n"
                  "- **Модель обучена на одном магазине (M5, CA_1)**. Для другого бизнеса качество не проверено; перед использованием сравните с вашим текущим методом.\n"
                  "- **Остатков и закупочных цен в данных нет**, поэтому план закупок проверен только на симуляции (запас при одинаковом уровне обслуживания), прибыль не оценивалась.\n"
                  "- **Прогноз не знает про акции, цены и праздники**: на этих данных такие признаки не улучшили точность, поэтому модель использует только историю продаж.\n"
                  "- Выигрыш над простым средним за 28 дней небольшой и нестабилен: на предыдущих окнах 3–5% по запасу, на итоговом тестовом окне его нет (модель требует на ~6% больше запаса, интервал включает 0). Над правилом «как за прошлую неделю» на тестовом окне выигрыш всего 0–4%.\n"
                  "- Продажи ≠ спрос: если товара не было на полке, спрос не виден. Товары без продаж 28 дней не планируются; новые (<28 дней) получают только среднее.\n"
                  "- История обрезается по последнее воскресенье: модель обучена на прогнозах, начинающихся с понедельника.\n"
                  "- План — рекомендация. Решение о заказе принимает человек, особенно для дорогих и скоропортящихся товаров.",
        "warn_last": "Совет: загружайте данные по воскресенье включительно, тогда прогноз начнётся с ближайшего понедельника.",
    },
    "en": {
        "title": "Demand Copilot: demand forecast and purchase plan",
        "intro": "Upload a shop's sales history (one row per item and day) to get a 28-day forecast and a weekly purchase plan.",
        "lang": "Language", "source": "Data", "demo": "Demo: M5 store CA_1 (300 items)", "upload": "Upload CSV",
        "file": "Sales CSV", "col_date": "Date column", "col_item": "Item column", "col_sales": "Sales column (units)",
        "col_price": "Price column (optional)", "none": "— none —", "stock_file": "Stock CSV (item_id, on_hand, on_order), optional",
        "demo_stock": "Synthetic stock for the demo (random, 1-5 days of sales)", "params": "Purchasing parameters",
        "lead": "Lead time, days", "service": "Target service level, %",
        "service_help": "Share of revenue served from stock. The conversion to safety stock is calibrated on M5 backtests; your data may behave differently.",
        "review": "Orders are placed every 7 days.", "no_data": "Choose the demo or upload a CSV to start.",
        "no_model": "Model not found. Run `python scripts/14_train_final.py` (see README).",
        "no_demo": "Demo data not found. Run `python scripts/01_download.py` and `python scripts/02_prepare.py`.",
        "bad_input": "Could not process the data:", "tab_fc": "Forecast", "tab_plan": "Purchase plan", "tab_quality": "Quality and limits",
        "data_check": "Data check", "items": "Items", "history": "History, days", "start": "Forecast from", "end": "to",
        "next7": "Forecast for 7 days, units", "last7": "Sold in the last 7 days, units", "pick": "Item", "daily": "Daily forecast",
        "weekly": "Weekly forecast", "week": "Week", "units": "units", "hist_series": "history", "fc_series": "forecast",
        "n_order": "Items to order", "units_order": "Units to order", "value_order": "Order value", "flags": "Flagged items",
        "only_order": "Show only items to order", "download": "Download plan (CSV)",
        "plan_note": "Target level = forecast over lead time + 7 days + safety stock. Order = target level - on hand - already on order. "
                     "No stock file was loaded, so stock is assumed to be zero.",
        "plan_note_stock": "Target level = forecast over lead time + 7 days + safety stock. Order = target level - on hand - already on order.",
        "st_ok": "ok", "st_new": "new (<28 days)", "st_dormant": "no sales for 28 days",
        "cols": {"item_id": "Item", "status": "Status", "forecast_next_7d": "Forecast 7d", "target_level": "Target level",
                 "safety_stock": "Safety stock", "on_hand": "On hand", "on_order": "On order", "order_qty": "Order", "order_value": "Value"},
        "chart_note": "The line on the right is the expected (average) demand per day, so values are fractional: 0.6 means 'about 6 sales in 10 days'. "
                      "Lead time and service level do not change this chart: they only change the recommendation below and the Purchase plan tab.",
        "item_plan": "Recommendation for this item (lead time {lead} d, service {service}%)",
        "m_target": "Target level, units", "m_onhand": "On hand, units", "m_onorder": "On order, units", "m_order": "Order, units",
        "week_actual": "actual", "week_forecast": "forecast", "kind": "Type",
        "flat_note": "Why the weeks look alike: the model forecasts the current sales level and the usual weekday pattern. "
                     "It does not know about trend, holidays, promotions or prices (such features did not improve accuracy in the checks), so the forecast "
                     "changes from week to week by only 1-2% on average, while real weekly sales swing by tens of percent. "
                     "New sales are used the next time data is uploaded: the forecast is recomputed every week.",
        "quality_note": "The tables below were computed in advance on historical M5 windows (not on your data).",
        "quality_tbl": "Check of the selected model on fold d1913 (after feature selection on the dev folds)",
        "test_tbl": "Final test window d1942-1969 (scored once; errors are higher than on the earlier windows)",
        "limits": "#### Limits and risks\n"
                  "- **The model was trained on one store (M5, CA_1)**. Quality on another business is not verified; compare with your current method first.\n"
                  "- **The data has no stock levels or purchase costs**, so the plan was only tested in a simulation (inventory at equal service level); profit was not estimated.\n"
                  "- **The forecast does not know about promotions, prices or holidays**: such features did not improve accuracy on this data, so the model uses sales history only.\n"
                  "- The gain over a plain 28-day average is small and unstable: 3-5% less inventory on earlier windows, none on the final test window (about 6% more inventory, interval includes 0). Over the 'same as last week' rule the test-window gain is only 0-4%.\n"
                  "- Sales are not demand: stock-outs hide demand. Items with no sales for 28 days are not planned; new items (<28 days) only get their average.\n"
                  "- History is cut at the last Sunday: the model was trained on forecasts starting on a Monday.\n"
                  "- The plan is a recommendation. A person decides the order, especially for expensive or perishable items.",
        "warn_last": "Tip: upload data through a Sunday so the forecast starts on the next Monday.",
    },
}


@st.cache_resource
def get_model():
    return S.load_model()


@st.cache_data(show_spinner="...")
def compute(df: pd.DataFrame, date_col: str, item_col: str, sales_col: str, price_col: str | None):
    shop = S.to_shop(df, date_col, item_col, sales_col, price_col)
    booster, meta = get_model()
    return shop, S.forecast(shop, booster, meta["features"])


@st.cache_data
def demo_table() -> pd.DataFrame:
    path = C.PROC / f"panel_{C.STORE}.parquet"
    df = pd.read_parquet(path, columns=["item_id", "date", "d", "sales", "sell_price"])
    return df[df.d <= C.TEST_CUTOFF].drop(columns="d")  # nothing after the model's training window is shown


def guess(columns, words, default=0):
    for i, c in enumerate(columns):
        if any(w in c.lower() for w in words):
            return i
    return default


HIST_COLOR, FC_COLOR = "#9FB1C9", "#3A6EA5"
# Fallback for the Streamlit toolbar (Deploy button and main menu); client.toolbarMode in .streamlit/config.toml does the same.
HIDE_CHROME = """<style>
[data-testid="stToolbar"], [data-testid="stDeployButton"], .stAppDeployButton, #MainMenu, footer {display: none !important;}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {display: none !important;}
</style>"""


def main() -> None:
    st.set_page_config(page_title="Demand Copilot", page_icon="📦", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(HIDE_CHROME, unsafe_allow_html=True)
    left, right = st.columns([3.2, 1], gap="large")  # the controls are a plain panel on the right, not a collapsible sidebar
    panel = right.container(border=True)
    lang = panel.radio("Language / Язык", ["ru", "en"], horizontal=True)
    t = TXT[lang]
    with left:
        st.title(t["title"])
        st.caption(t["intro"])
        render(t, panel)


def render(t: dict, panel) -> None:
    try:
        booster, meta = get_model()
    except Exception:
        st.error(t["no_model"])
        st.stop()
    calibration = meta["calibration"]

    panel.subheader(t["source"])
    source = panel.radio(t["source"], [t["demo"], t["upload"]], label_visibility="collapsed")
    stock_df = None
    if source == t["demo"]:
        if not (C.PROC / f"panel_{C.STORE}.parquet").exists():
            st.error(t["no_demo"])
            st.stop()
        df = demo_table()
        cols = dict(date_col="date", item_col="item_id", sales_col="sales", price_col="sell_price")
        use_synthetic_stock = panel.checkbox(t["demo_stock"], value=True)
    else:
        up = panel.file_uploader(t["file"], type=["csv"])
        if up is None:
            st.info(t["no_data"])
            st.stop()
        df = pd.read_csv(up)
        c = list(df.columns)
        cols = dict(
            date_col=panel.selectbox(t["col_date"], c, guess(c, ["date", "day", "дата"])),
            item_col=panel.selectbox(t["col_item"], c, guess(c, ["item", "sku", "product", "товар"], 1 % len(c))),
            sales_col=panel.selectbox(t["col_sales"], c, guess(c, ["sale", "qty", "unit", "quant", "прод"], 2 % len(c))))
        price = panel.selectbox(t["col_price"], [t["none"], *c], 0)
        cols["price_col"] = None if price == t["none"] else price
        use_synthetic_stock = False
        stock_up = panel.file_uploader(t["stock_file"], type=["csv"])
        if stock_up is not None:
            stock_df = pd.read_csv(stock_up)

    panel.subheader(t["params"])
    lead = panel.selectbox(t["lead"], [1, 3, 7], index=1)
    service = panel.selectbox(t["service"], [90, 95, 98], index=1, help=t["service_help"])
    panel.caption(t["review"])

    try:
        shop, f = compute(df, **cols)
    except Exception as e:  # shown to the user, not swallowed
        st.error(f"{t['bad_input']} {e}")
        st.stop()
    n = len(shop.ids)
    last_date, first_forecast = shop.dates[-1], shop.dates[-1] + pd.Timedelta(days=1)
    status = S.item_status(shop)

    on_hand = on_order = None
    if stock_df is not None and {"item_id", "on_hand"} <= set(stock_df.columns):
        s = stock_df.set_index(stock_df.item_id.astype(str))
        on_hand = s.on_hand.reindex(shop.ids).fillna(0).to_numpy(float)
        on_order = (s.on_order.reindex(shop.ids).fillna(0).to_numpy(float) if "on_order" in s.columns else np.zeros(n))
    elif use_synthetic_stock:
        rng = np.random.default_rng(0)
        on_hand = np.round(shop.Y[:, -28:].mean(axis=1) * rng.uniform(1, 5, n))
        on_order = np.zeros(n)
    z = S.z_for_service(calibration, lead, service / 100)
    plan = S.purchase_plan(shop, f, lead, z, on_hand, on_order)

    tab_fc, tab_plan, tab_q = st.tabs([t["tab_fc"], t["tab_plan"], t["tab_quality"]])

    with tab_fc:
        with st.expander(t["data_check"], expanded=bool(shop.notes)):
            a, b, c4 = st.columns(3)
            a.metric(t["items"], n)
            b.metric(t["history"], len(shop.dates))
            c4.metric(t["start"], f"{first_forecast.date()}  {t['end']}  {(first_forecast + pd.Timedelta(days=27)).date()}")
            for note in shop.notes:
                st.warning(note)
            if not shop.notes:
                st.caption(t["warn_last"])
            n_new, n_dormant = int((status == "new").sum()), int((status == "dormant").sum())
            if n_new or n_dormant:
                st.info(f"{t['st_new']}: {n_new}; {t['st_dormant']}: {n_dormant}")
        k1, k2 = st.columns(2)
        k1.metric(t["next7"], f"{f[:, :7].sum():,.0f}")
        k2.metric(t["last7"], f"{shop.Y[:, -7:].sum():,.0f}")
        order = np.argsort(-shop.Y[:, -28:].sum(axis=1))
        pick = st.selectbox(t["pick"], [shop.ids[i] for i in order])
        i = shop.ids.index(pick)
        hist = pd.Series(shop.Y[i, -84:], index=shop.dates[-84:], name=t["hist_series"])
        fut = pd.Series(f[i], index=pd.date_range(first_forecast, periods=C.HORIZON), name=t["fc_series"])
        st.subheader(t["daily"])
        st.caption(t["chart_note"])
        st.line_chart(pd.concat([hist, fut], axis=1), height=320, color=[HIST_COLOR, FC_COLOR])

        row = plan.iloc[i]
        st.subheader(t["item_plan"].format(lead=lead, service=service))
        r1, r2, r3, r4 = st.columns(4)
        r1.metric(t["m_target"], f"{row.target_level:,.0f}")
        r2.metric(t["m_onhand"], f"{row.on_hand:,.0f}")
        r3.metric(t["m_onorder"], f"{row.on_order:,.0f}")
        r4.metric(t["m_order"], f"{int(row.order_qty):,}")

        past = shop.Y[i, -28:].reshape(4, 7).sum(axis=1)
        ahead = f[i].reshape(4, 7).sum(axis=1)
        starts = [first_forecast + pd.Timedelta(days=7 * (w - 4)) for w in range(8)]
        labels = [f"{s:%m-%d}" for s in starts]
        bars = pd.DataFrame({t["week_actual"]: [*past, *[np.nan] * 4], t["week_forecast"]: [*[np.nan] * 4, *ahead]}, index=labels)
        st.subheader(t["weekly"])
        st.bar_chart(bars, height=240, color=[HIST_COLOR, FC_COLOR])
        fmt = lambda v: f"{v:,.0f}" if v >= 10 else f"{v:.1f}"  # noqa: E731
        weekly = pd.DataFrame({t["week"]: [f"{s:%Y-%m-%d}" for s in starts],
                               t["kind"]: [*[t["week_actual"]] * 4, *[t["week_forecast"]] * 4],
                               t["units"]: [fmt(v) for v in [*past, *ahead]]})
        st.dataframe(weekly, hide_index=True, width=420)
        st.info(t["flat_note"])

    with tab_plan:
        todo = plan[plan.order_qty > 0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(t["n_order"], len(todo))
        m2.metric(t["units_order"], f"{int(plan.order_qty.sum()):,}")
        if "order_value" in plan:
            m3.metric(t["value_order"], f"{plan.order_value.sum():,.0f}")
        m4.metric(t["flags"], int((status != "ok").sum()))
        st.caption(t["plan_note_stock"] if on_hand is not None else t["plan_note"])
        only = st.checkbox(t["only_order"], value=True)
        view = (todo if only else plan).copy()
        sort_col = "order_value" if "order_value" in view else "order_qty"
        view = view.sort_values(sort_col, ascending=False)
        view["status"] = view.status.map({"ok": t["st_ok"], "new": t["st_new"], "dormant": t["st_dormant"]})
        shown = [c for c in t["cols"] if c in view.columns]
        st.dataframe(view[shown].rename(columns=t["cols"]), hide_index=True, width="stretch", height=480)
        st.download_button(t["download"], plan.to_csv(index=False).encode("utf-8"), "purchase_plan.csv", "text/csv")

    with tab_q:
        st.caption(t["quality_note"])
        res_path = C.REPORTS / "feature_selection_results.csv"
        if res_path.exists():
            res = pd.read_csv(res_path)
            res = res[res.split.str.startswith("validation")].drop(columns=["split", "weekly_std"]).round(4)
            st.markdown(f"**{t['quality_tbl']}**")
            st.dataframe(res.rename(columns={"model": "model", "inv_days_95": "inventory days @95% fill"}), hide_index=True, width="stretch")
        test_path = C.REPORTS / "test_point_metrics.csv"
        if test_path.exists():
            tst = pd.read_csv(test_path)
            keep = ["seasonal naive", "28-day average", "weekday average, 8 weeks", "AutoETS", "LightGBM selected 11 (FINAL, stored model)"]
            st.markdown(f"**{t['test_tbl']}**")
            st.dataframe(tst[tst.model.isin(keep)][["model", "daily_wape", "weekly_wape", "wape_28d"]], hide_index=True, width="stretch")
        fig_path = C.REPORTS / "figures" / "inventory_frontier.png"
        if fig_path.exists():
            st.image(str(fig_path), caption="Inventory vs fill rate, M5 backtests (lower and to the left is better)")
        st.markdown(t["limits"])


main()
