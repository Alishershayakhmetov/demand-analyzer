// Builds docs/Demand_Copilot_slides.pptx from reports/report_numbers.json (run: npm run slides, see package.json).
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.APPLY_THEME || "apply_theme.js");

const ROOT = path.resolve(__dirname, "..", "..");
const NUM = JSON.parse(fs.readFileSync(path.join(ROOT, "reports", "report_numbers.json"), "utf8"));
const OUT = path.join(ROOT, "docs", "Demand_Copilot_slides.pptx");

const THEME = {
  name: "DemandCopilot", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "1F2D45", lt1: "FFFFFF", dk2: "5D6B82", lt2: "EEF2F8", accent1: "3A6EA5", accent2: "E8765B", accent3: "C5D0DE",
            accent4: "7A8CA5", accent5: "C8472E", accent6: "238A70", hlink: "3A6EA5", folHlink: "5D6B82" },
};
const HEX = { ink: "1F2D45", blue: "3A6EA5", coral: "E8765B", slate: "5D6B82", light: "C5D0DE", grid: "DDE4EE", brick: "C8472E" };

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Demand Copilot: прогноз спроса и план закупок";
pres.author = "[Имя Фамилия], [Имя Фамилия], [Имя Фамилия]";
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "CONTENT", background: { color: "FFFFFF" },
  slideNumber: { x: 9.2, y: 5.22, w: 0.5, h: 0.3, fontFace: "Calibri", fontSize: 10, color: HEX.slate },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.28, w: 9.0, h: 0.95, fontFace: "Cambria", fontSize: 26,
    bold: true, color: C.text1, align: "left", valign: "middle", margin: 0 }, text: "" } }],
});
pres.defineSlideMaster({
  title: "LIGHTTITLE", background: { color: "EEF2F8" },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.4, w: 8.8, h: 0.9, fontFace: "Cambria", fontSize: 28,
    bold: true, color: C.text1, align: "left", valign: "middle", margin: 0 }, text: "" } }],
});

const body = { fontFace: "Calibri", color: C.text1, margin: 0, isTextBox: true, valign: "top" };
const content = (title, section) => {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
};
const card = (s, x, y, w, h, name, fill = C.background2) => s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: HEX.grid, width: 0.75 }, objectName: name });
const bullets = (items, size = 14, after = 6) => items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, fontSize: size, paraSpaceAfter: after } }));
const pct = (a, b) => `${a / b - 1 >= 0 ? "+" : "−"}${Math.abs((a / b - 1) * 100).toFixed(1)}%`;
const W = NUM.wape, I = NUM.inventory;

// 1. Title
pres.addSection({ title: "Введение" });
{
  const s = pres.addSlide({ masterName: "LIGHTTITLE", sectionTitle: "Введение" });
  s.addShape(pres.ShapeType.ellipse, { x: 6.6, y: 0.5, w: 3.4, h: 3.4, fill: { color: C.accent1, transparency: 72 }, line: { color: C.accent1, width: 0 }, objectName: "Круг 1" });
  s.addShape(pres.ShapeType.ellipse, { x: 8.0, y: 2.5, w: 2.4, h: 2.4, fill: { color: C.accent2, transparency: 55 }, line: { color: C.accent2, width: 0 }, objectName: "Круг 2" });
  s.addShape(pres.ShapeType.ellipse, { x: 5.7, y: 3.3, w: 1.6, h: 1.6, fill: { color: C.accent3, transparency: 45 }, line: { color: C.accent3, width: 0 }, objectName: "Круг 3" });
  s.addText("Demand Copilot", { x: 0.6, y: 1.2, w: 6.2, h: 0.9, fontFace: "Cambria", fontSize: 44, bold: true, color: C.text1, margin: 0, isTextBox: true });
  s.addText("Прогноз спроса и план закупок для малого магазина", { x: 0.6, y: 2.2, w: 5.6, h: 0.9, fontFace: "Calibri", fontSize: 22, color: C.text2, margin: 0, isTextBox: true, valign: "top" });
  s.addText("AI Essentials: Theory and Applications · финальный проект", { x: 0.6, y: 3.7, w: 6, h: 0.35, fontFace: "Calibri", fontSize: 14, color: C.text2, margin: 0, isTextBox: true });
  s.addText("[Имя Фамилия] · [Имя Фамилия] · [Имя Фамилия]", { x: 0.6, y: 4.1, w: 6, h: 0.35, fontFace: "Calibri", fontSize: 16, bold: true, color: C.text1, margin: 0, isTextBox: true });
  s.addNotes("~15 с. Представьтесь, назовите тему. Проект: система, которая по истории продаж магазина прогнозирует спрос на 28 дней и предлагает недельный план закупок. Дальше: задача, данные, метод, честные результаты, демо.");
}

// 2. Problem
pres.addSection({ title: "Задача и данные" });
{
  const s = content("Задача: закупать по прогнозу, а не на глаз", "Задача и данные");
  s.addText([
    { text: "Для кого", options: { bold: true, color: C.accent1, fontSize: 16, breakLine: true, paraSpaceAfter: 4 } },
    { text: "Владелец небольшого магазина, который закупает «на глаз» или по правилу «как в прошлую неделю».", options: { fontSize: 15, breakLine: true, paraSpaceAfter: 14 } },
    { text: "Что делаем", options: { bold: true, color: C.accent1, fontSize: 16, breakLine: true, paraSpaceAfter: 4 } },
    { text: "Прогноз продаж каждого товара на 28 дней и недельный план закупок: что и сколько заказать.", options: { fontSize: 15, breakLine: true, paraSpaceAfter: 14 } },
    { text: "Почему ML", options: { bold: true, color: C.accent1, fontSize: 16, breakLine: true, paraSpaceAfter: 4 } },
    { text: "Простое правило не видит одновременно уровень спроса, рисунок по дням недели и редкие продажи.", options: { fontSize: 15 } },
  ], { ...body, x: 0.5, y: 1.45, w: 4.3, h: 3.6 });
  s.addText("Как измеряем успех", { ...body, x: 5.2, y: 1.45, w: 4.3, h: 0.35, fontSize: 16, bold: true, color: C.accent1 });
  const goals = [["1", "Прогноз точнее seasonal naive минимум на 10–15% (WAPE)"], ["2", "При том же уровне обслуживания нужно меньше запаса, чем по правилу «как в прошлую неделю»"], ["3", "Проверка на окне, которое не использовалось при разработке"]];
  goals.forEach(([n, t], i) => {
    const y = 1.9 + i * 1.07;
    card(s, 5.2, y, 4.3, 0.95, `Цель ${n}`);
    s.addShape(pres.ShapeType.ellipse, { x: 5.35, y: y + 0.2, w: 0.55, h: 0.55, fill: { color: C.accent1 }, line: { color: C.accent1, width: 0 }, objectName: `Номер ${n}` });
    s.addText(n, { x: 5.35, y: y + 0.2, w: 0.55, h: 0.55, fontFace: "Calibri", fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText(t, { ...body, x: 6.05, y: y + 0.08, w: 3.35, h: 0.8, fontSize: 14, valign: "middle" });
  });
  s.addNotes("~30 с. Пользователь — владелец небольшого магазина. Результат — прогноз на 28 дней и недельный план закупок. Три измеримых критерия успеха: точность относительно seasonal naive, запас при том же уровне обслуживания и проверка на окне, которое не использовалось при разработке. ML нужен, потому что простое правило не учитывает сразу уровень, дни недели и разреженность.");
}

// 3. Data
{
  const s = content("Данные: магазин M5 (Walmart), 300 товаров", "Задача и данные");
  const stats = [["300", "товаров: по 100 из трёх групп по объёму продаж"], ["1 969", "дней истории: 2011-01-29 … 2016-06-19"], ["54%", "дней без продаж у товара, который был в наличии"]];
  stats.forEach(([big, label], i) => {
    const x = 0.5 + i * 3.05;
    card(s, x, 1.45, 2.9, 1.65, `Показатель ${i + 1}`);
    s.addText(big, { x: x + 0.15, y: 1.55, w: 2.6, h: 0.8, fontFace: "Cambria", fontSize: 40, bold: true, color: C.accent1, margin: 0, isTextBox: true, valign: "middle" });
    s.addText(label, { ...body, x: x + 0.15, y: 2.35, w: 2.6, h: 0.7, fontSize: 14 });
  });
  s.addText(bullets([
    "Открытый датасет M5, скачивается без регистрации; в нём продажи, цены и календарь.",
    "Подвыборка (магазин CA_1) выбрана только по данным до 1829-го дня, чтобы не подсмотреть будущее.",
    "Остатков и закупочных цен в данных нет: закупки проверяем симуляцией с явными допущениями.",
  ], 15, 8), { ...body, x: 0.5, y: 3.4, w: 9.0, h: 1.7 });
  s.addNotes("~35 с. Данные: открытый датасет M5, один магазин, 300 товаров, почти 2000 дней. У трети дневных значений… точнее: у 54% дней товар не продавался — разреженный спрос. Главное ограничение: в открытых данных нет остатков и закупочных цен, поэтому закупки мы проверяем симуляцией и честно об этом говорим.");
}

// 4. Protocol
{
  const s = content("Проверка: по времени, тест открыт один раз", "Задача и данные");
  const y = 1.55, h = 0.8;
  const segs = [["обучение", 0.5, 4.2, "9FB1C9", "1F2D45"], ["dev ×3", 4.75, 1.75, HEX.blue, "FFFFFF"], ["val", 6.55, 0.95, "238A70", "FFFFFF"], ["тест", 7.55, 1.95, HEX.brick, "FFFFFF"]];
  segs.forEach(([t, x, w, fill, tc], i) => {
    s.addShape(pres.ShapeType.rect, { x, y, w, h, fill: { color: fill }, line: { color: "FFFFFF", width: 1 }, objectName: `Отрезок ${i + 1}` });
    s.addText(t, { x, y, w, h, fontFace: "Calibri", fontSize: 16, bold: true, color: tc, align: "center", valign: "middle", margin: 0, isTextBox: true });
  });
  s.addText("d1 … d1829", { ...body, x: 0.5, y: y + h + 0.05, w: 4.2, h: 0.3, fontSize: 12, color: C.text2, align: "center" });
  s.addText("d1829, d1857, d1885", { ...body, x: 4.75, y: y + h + 0.05, w: 1.75, h: 0.3, fontSize: 11, color: C.text2, align: "center" });
  s.addText("d1913", { ...body, x: 6.55, y: y + h + 0.05, w: 0.95, h: 0.3, fontSize: 12, color: C.text2, align: "center" });
  s.addText("d1942–1969", { ...body, x: 7.55, y: y + h + 0.05, w: 1.95, h: 0.3, fontSize: 12, color: C.text2, align: "center" });
  s.addText("Схема не в масштабе: каждое окно — 28 дней прогноза после даты отсечения", { ...body, x: 0.5, y: 2.75, w: 9, h: 0.3, fontSize: 12, italic: true, color: C.text2 });
  const rules = [["Только прошлое", "Признаки строятся из истории до даты прогноза; цены и календарь будущих дней известны заранее."],
                 ["Выбор — на dev", "Гиперпараметры и признаки выбираются только на трёх dev-окнах; validation — проверка после выбора."],
                 ["Тест — один раз", "Гипотезы и правила записаны до запуска; скрипт сверяет хеш модели и не запускается повторно."]];
  rules.forEach(([h1, t], i) => {
    const x = 0.5 + i * 3.05;
    card(s, x, 3.2, 2.9, 1.85, `Правило ${i + 1}`);
    s.addText(h1, { ...body, x: x + 0.15, y: 3.3, w: 2.6, h: 0.35, fontSize: 16, bold: true, color: C.accent1 });
    s.addText(t, { ...body, x: x + 0.15, y: 3.7, w: 2.6, h: 1.3, fontSize: 14 });
  });
  s.addNotes("~35 с. Разбиение только по времени, без перемешивания. Три dev-окна — для выбора признаков и параметров, validation — проверка после выбора, тест — последнее окно, которое никто не смотрел. Гипотезы для теста записаны до запуска, скрипт не запускается второй раз. Это то, что отличает честную оценку от подгонки.");
}

// 5. Method
pres.addSection({ title: "Метод и результаты" });
{
  const s = content("Метод: от простых правил к LightGBM", "Метод и результаты");
  const cols = [["1. История продаж", "Строка на товар и день: дата, товар, продано штук (цена — по желанию)."],
                ["2. Прогноз", "Одна общая модель LightGBM (Tweedie) на 11 признаках: средние за 7–91 дней, доля нулей, дни с последней продажи, шаг горизонта. Сравнение с naive, seasonal naive, средним за 28 дней, средним по дням недели, ETS, Croston/TSB."],
                ["3. План закупок", "Цель = прогноз на срок поставки + 7 дней + z·σ. Заказ = цель − остаток − в пути. Коэффициент z подбирается под нужный уровень обслуживания."]];
  const xs = [0.5, 3.55, 6.6], ws = [2.9, 2.9, 2.9];
  cols.forEach(([h1, t], i) => {
    card(s, xs[i], 1.45, ws[i], 2.95, `Этап ${i + 1}`);
    s.addText(h1, { ...body, x: xs[i] + 0.15, y: 1.55, w: 2.6, h: 0.4, fontSize: 17, bold: true, color: C.accent1 });
    s.addText(t, { ...body, x: xs[i] + 0.15, y: 2.05, w: 2.6, h: 2.1, fontSize: 14 });
  });
  s.addText("Все модели сравниваются на одинаковых окнах и метриках: WAPE (по дням, неделям, за 28 дней), смещение, RMSSE. Для закупок: запас при одинаковом уровне обслуживания.",
    { ...body, x: 0.5, y: 4.55, w: 9.0, h: 0.6, fontSize: 15, bold: true });
  [3.42, 6.47].forEach((x, i) => s.addText("▶", { x, y: 2.6, w: 0.2, h: 0.4, fontFace: "Calibri", fontSize: 16, color: C.accent2, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `Стрелка ${i + 1}` }));
  s.addNotes("~40 с. Лестница от простого к сложному: naive, seasonal naive, средние, ETS, Croston, затем одна общая модель LightGBM на всех товарах. Признаков 11: в основном недавний уровень продаж. Над прогнозом — слой закупок: целевой уровень запаса покрывает срок поставки плюс неделю плюс страховой запас. Закупки проверены в симуляции склада.");
}

// 6. Forecast accuracy (native chart)
{
  const s = content("Прогноз лучше seasonal naive, но не лучше среднего", "Метод и результаты");
  const cats = ["dev (3 окна)", "validation d1913", "тест d1942–1969"];
  const series = [["seasonal naive", "seasonal naive", "C5D0DE"], ["28-day average", "среднее за 28 дней", HEX.slate], ["ETS", "ETS", HEX.coral], ["LightGBM (selected)", "LightGBM", HEX.blue]];
  s.addChart(pres.charts.BAR, series.map(([k, name]) => ({ name, labels: cats, values: [W[k].dev, W[k].val, W[k].test].map((v) => +v.toFixed(3)) })), {
    x: 0.4, y: 1.35, w: 6.0, h: 3.8, barDir: "col", barGrouping: "clustered", chartColors: series.map((x) => x[2]),
    showTitle: true, title: "Недельный WAPE, ниже — лучше", titleFontSize: 13, titleColor: HEX.ink, titleFontFace: "+mn-lt",
    showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: "+mn-lt", legendColor: HEX.ink,
    showValue: true, dataLabelFontSize: 9, dataLabelFormatCode: "0.00", dataLabelPosition: "outEnd", dataLabelColor: HEX.ink, dataLabelFontFace: "+mn-lt",
    valAxisMinVal: 0, valAxisMaxVal: 0.5, valAxisLabelFontSize: 10, valAxisLabelColor: HEX.slate, valAxisLabelFontFace: "+mn-lt",
    catAxisLabelFontSize: 11, catAxisLabelColor: HEX.ink, catAxisLabelFontFace: "+mn-lt",
    valGridLine: { color: HEX.grid, size: 0.5 }, catGridLine: { style: "none" }, barGapWidthPct: 60,
  });
  card(s, 6.6, 1.45, 2.9, 3.55, "Выводы");
  s.addText([
    { text: "К seasonal naive", options: { bold: true, color: C.accent1, fontSize: 14, breakLine: true } },
    { text: `${pct(W["LightGBM (selected)"].dev, W["seasonal naive"].dev)} dev, ${pct(W["LightGBM (selected)"].val, W["seasonal naive"].val)} val, ${pct(W["LightGBM (selected)"].test, W["seasonal naive"].test)} тест`, options: { fontSize: 14, breakLine: true, paraSpaceAfter: 10 } },
    { text: "К среднему за 28 дней", options: { bold: true, color: C.accent1, fontSize: 14, breakLine: true } },
    { text: `${pct(W["LightGBM (selected)"].dev, W["28-day average"].dev)} dev, ${pct(W["LightGBM (selected)"].val, W["28-day average"].val)} val, ${pct(W["LightGBM (selected)"].test, W["28-day average"].test)} тест`, options: { fontSize: 14, breakLine: true, paraSpaceAfter: 10 } },
    { text: "Тестовое окно труднее для всех методов, а разрыв с простым средним мал.", options: { fontSize: 14 } },
  ], { ...body, x: 6.75, y: 1.6, w: 2.6, h: 3.3 });
  s.addNotes("~45 с. Недельная ошибка WAPE по трём типам окон. На dev и validation модель заметно лучше seasonal naive, но почти не отличается от простого среднего за 28 дней. На тесте окно труднее для всех методов: LightGBM лучше seasonal naive на 6% и лучше среднего за 28 дней на 1%. Это честный результат: преимущество над простым средним мало.");
}

// 7. Error analysis (native chart)
{
  const s = content("Где модель помогает, а где нет", "Метод и результаты");
  const cats = ["высокий объём", "средний объём", "малый объём"];
  s.addChart(pres.charts.BAR, [
    { name: "среднее за 28 дней", labels: cats, values: [0.3846, 0.5206, 0.5908] },
    { name: "LightGBM", labels: cats, values: [0.3843, 0.4898, 0.6037] }], {
    x: 0.4, y: 1.35, w: 4.9, h: 3.8, barDir: "col", barGrouping: "clustered", chartColors: [HEX.slate, HEX.blue],
    showTitle: true, title: "Тест: недельный WAPE по группам объёма", titleFontSize: 13, titleColor: HEX.ink, titleFontFace: "+mn-lt",
    showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: "+mn-lt", legendColor: HEX.ink,
    showValue: true, dataLabelFontSize: 10, dataLabelFormatCode: "0.00", dataLabelPosition: "outEnd", dataLabelColor: HEX.ink, dataLabelFontFace: "+mn-lt",
    valAxisMinVal: 0, valAxisMaxVal: 0.7, valAxisLabelFontSize: 10, valAxisLabelColor: HEX.slate, valAxisLabelFontFace: "+mn-lt",
    catAxisLabelFontSize: 11, catAxisLabelColor: HEX.ink, catAxisLabelFontFace: "+mn-lt",
    valGridLine: { color: HEX.grid, size: 0.5 }, catGridLine: { style: "none" }, barGapWidthPct: 70,
  });
  s.addText(bullets([
    "Ошибка идёт за объёмом: 30 товаров (10%) дают 34% всей ошибки.",
    "Лучше у товаров среднего объёма (−6%) и «комковатых» (−9%), хуже у малообъёмных (+2%).",
    "Небольшие скидки модель переоценивает (смещение +20%), но таких недель мало.",
    "Выигрыш над средним за 28 дней по окнам: +4,1%, −2,9%, −4,0%, +2,2%, то есть нестабилен.",
  ], 14, 8), { ...body, x: 5.5, y: 1.45, w: 4.0, h: 3.6 });
  s.addNotes("~35 с. Ошибка пропорциональна объёму: самые продаваемые товары дают основную долю ошибки. Модель лучше на товарах среднего объёма и со скачками, хуже на малообъёмных, где она завышает прогноз. Выигрыш над простым средним на разных окнах меняет знак — на нём нельзя строить громких заявлений.");
}

// 8. Inventory (native chart)
{
  const s = content("Закупки: экономия запаса была, на тесте исчезла", "Метод и результаты");
  const keys = ["owner", "ma28", "wd8", "lean"];
  const cats = ["«как в прошлую неделю»", "среднее за 28 дней", "среднее по дням недели", "LightGBM"];
  s.addChart(pres.charts.BAR, [
    { name: "4 прежних окна", labels: cats, values: keys.map((k) => +I[k].earlier.toFixed(2)) },
    { name: "тест", labels: cats, values: keys.map((k) => +I[k].test.toFixed(2)) }], {
    x: 0.4, y: 1.35, w: 5.3, h: 3.8, barDir: "col", barGrouping: "clustered", chartColors: [HEX.slate, HEX.blue],
    showTitle: true, title: "Запас, дней продаж (95% обслуживания, поставка 3 дня)", titleFontSize: 12, titleColor: HEX.ink, titleFontFace: "+mn-lt",
    showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: "+mn-lt", legendColor: HEX.ink,
    showValue: true, dataLabelFontSize: 10, dataLabelFormatCode: "0.0", dataLabelPosition: "outEnd", dataLabelColor: HEX.ink, dataLabelFontFace: "+mn-lt",
    valAxisMinVal: 0, valAxisMaxVal: 11, valAxisLabelFontSize: 10, valAxisLabelColor: HEX.slate, valAxisLabelFontFace: "+mn-lt",
    catAxisLabelFontSize: 10, catAxisLabelColor: HEX.ink, catAxisLabelFontFace: "+mn-lt",
    valGridLine: { color: HEX.grid, size: 0.5 }, catGridLine: { style: "none" }, barGapWidthPct: 60,
  });
  s.addText([
    { text: "Прежние окна", options: { bold: true, color: C.accent1, fontSize: 15, breakLine: true } },
    { text: `${pct(I.lean.earlier, I.owner.earlier)} к правилу «как в прошлую неделю» (интервал −18…−13%), ${pct(I.lean.earlier, I.ma28.earlier)} к среднему за 28 дней (−6…−1%).`, options: { fontSize: 14, breakLine: true, paraSpaceAfter: 12 } },
    { text: "Тест", options: { bold: true, color: C.accent5, fontSize: 15, breakLine: true } },
    { text: `${pct(I.lean.test, I.owner.test)} к правилу и ${pct(I.lean.test, I.ma28.test)} к среднему за 28 дней; интервалы включают ноль.`, options: { fontSize: 14, breakLine: true, paraSpaceAfter: 12 } },
    { text: "Вывод: заявлять, что LightGBM экономит запас по сравнению с простыми правилами, нельзя.", options: { fontSize: 14, bold: true } },
  ], { ...body, x: 5.9, y: 1.45, w: 3.6, h: 3.6 });
  s.addNotes("~40 с. Закупки проверены в симуляции склада: еженедельный заказ, срок поставки 3 дня, уровень обслуживания 95% выручки. На четырёх прежних окнах модель требовала на 15% меньше запаса, чем правило «как в прошлую неделю», и на 4% меньше, чем среднее за 28 дней. На тесте это не воспроизвелось: на 2,5% меньше первого и на 7% больше второго, оба интервала включают ноль. Мы не можем утверждать экономию запаса.");
}

// 9. Test hypotheses
{
  const s = content("Тест: три гипотезы, заданные до запуска", "Метод и результаты");
  const cell = (t, o = {}) => ({ text: t, options: { fontFace: "Calibri", fontSize: 13, color: C.text1, valign: "middle", ...o } });
  const head = (t) => cell(t, { bold: true, color: C.background1, fill: { color: C.accent1 }, fontSize: 13 });
  const ok = (t) => cell(t, { bold: true, color: C.background1, fill: { color: C.accent6 }, align: "center" });
  const no = (t) => cell(t, { bold: true, color: C.background1, fill: { color: C.accent5 }, align: "center" });
  const rows = [
    [head("Гипотеза"), head("Результат"), head("Итог")],
    [cell("H1. Недельный WAPE минимум на 10% ниже seasonal naive"), cell(`${pct(W["LightGBM (selected)"].test, W["seasonal naive"].test)}: 0,4221 против 0,4484`), no("не подтверждена")],
    [cell("H2. Недельный WAPE не хуже среднего за 28 дней более чем на 3%"), cell(`${pct(W["LightGBM (selected)"].test, W["28-day average"].test)}: 0,4221 против 0,4274`), ok("подтверждена")],
    [cell("H3. Запас при 95% обслуживания: минимум на 10% меньше, чем по правилу «как в прошлую неделю», и меньше, чем по среднему за 28 дней (интервалы ниже нуля)"),
      cell(`${pct(I.lean.test, I.owner.test)} и ${pct(I.lean.test, I.ma28.test)}; оба интервала включают ноль`), no("не подтверждена")],
  ];
  s.addTable(rows, { x: 0.5, y: 1.45, w: 9.0, colW: [4.6, 2.9, 1.5], rowH: [0.4, 0.6, 0.6, 1.0], border: { type: "solid", pt: 0.75, color: HEX.grid }, margin: [0.05, 0.1, 0.05, 0.1], objectName: "Таблица гипотез" });
  s.addText(bullets([
    "Окно 23.05–19.06.2016 труднее для всех методов: недельный WAPE 0,41–0,45 против 0,34–0,37 на прежних окнах.",
    "Результат записан как есть: после запуска ничего не менялось, повторный запуск скрипт не допускает.",
  ], 14, 6), { ...body, x: 0.5, y: 4.35, w: 9.0, h: 0.85 });
  s.addNotes("~40 с. Три гипотезы мы записали до запуска теста. H2 подтверждена: модель не хуже простого среднего. H1 и H3 не подтверждены: выигрыш над seasonal naive 6% вместо 10%, а экономии запаса нет. Мы показываем это без смягчений — после запуска ничего не менялось, а скрипт отказывается запускаться повторно.");
}

// 10. What was tried
{
  const s = content("Что проверили и отвергли", "Метод и результаты");
  const items = [["SHAP", "71% вклада даёт недавний уровень продаж; календарь 10%, цена 4%, SNAP меньше 1%."],
                 ["Ablation", "Все «известные заранее» признаки вместе ухудшают недельную ошибку на 2–4%, поэтому модель использует только историю."],
                 ["Отбор на dev", "Ни одна группа признаков не прошла порог 0,0029: выбран набор из 11 признаков."],
                 ["Календарь после тюнинга", "С календарём и ценой хуже на dev (0,342 и 0,339 против 0,334). Прогноз колеблется между неделями на 4–7% при реальных 47%."]];
  items.forEach(([h1, t], i) => {
    const x = 0.5 + (i % 2) * 4.6, y = 1.45 + Math.floor(i / 2) * 1.6;
    card(s, x, y, 4.4, 1.45, `Проверка ${i + 1}`);
    s.addText(h1, { ...body, x: x + 0.15, y: y + 0.1, w: 4.1, h: 0.35, fontSize: 16, bold: true, color: C.accent1 });
    s.addText(t, { ...body, x: x + 0.15, y: y + 0.5, w: 4.1, h: 0.9, fontSize: 14 });
  });
  s.addText("Поэтому прогноз почти плоский от недели к неделе: недельные скачки продаж по имеющимся данным не предсказать.", { ...body, x: 0.5, y: 4.7, w: 9.0, h: 0.5, fontSize: 15, bold: true });
  s.addNotes("~35 с. Мы не ограничились одной моделью: SHAP показал, что почти всё даёт недавний уровень продаж. Ablation и отбор признаков на dev показали, что календарь и цена не помогают, а после отдельного тюнинга календарь тоже оказался хуже. Следствие, которое видно в демо: прогноз почти одинаков от недели к неделе, и это не ошибка, а свойство данных.");
}

// 11. Demo
pres.addSection({ title: "Демо и итоги" });
{
  const s = content("Демо: загрузил продажи — получил план", "Демо и итоги");
  const imgH = 3.3;
  s.addImage({ path: path.join(ROOT, "docs", "assets", "app_forecast.jpg"), x: 0.5, y: 1.4, w: imgH * 800 / 882, h: imgH, altText: "Окно приложения: прогноз по товару, рекомендация и недельный график факта и прогноза; справа панель параметров" });
  s.addImage({ path: path.join(ROOT, "docs", "assets", "app_plan_crop.jpg"), x: 0.5 + imgH * 800 / 882 + 0.2, y: 1.4, w: imgH * 550 / 570, h: imgH, altText: "Вкладка «План закупок»: показатели и таблица заказа по товарам" });
  s.addText("Прогноз по товару", { ...body, x: 0.5, y: 4.75, w: 3.0, h: 0.3, fontSize: 12, color: C.text2 });
  s.addText("План закупок", { ...body, x: 0.5 + imgH * 800 / 882 + 0.2, y: 4.75, w: 3.0, h: 0.3, fontSize: 12, color: C.text2 });
  const steps = [["Загрузить CSV", "или открыть демо на данных M5"], ["Выбрать параметры", "срок поставки, обслуживание"], ["Смотреть товар", "прогноз и рекомендация"], ["Скачать план", "таблица закупок в CSV"]];
  steps.forEach(([h1, t], i) => {
    const y = 1.4 + i * 0.92;
    s.addShape(pres.ShapeType.ellipse, { x: 7.1, y: y + 0.08, w: 0.45, h: 0.45, fill: { color: C.accent1 }, line: { color: C.accent1, width: 0 }, objectName: `Шаг ${i + 1}` });
    s.addText(String(i + 1), { x: 7.1, y: y + 0.08, w: 0.45, h: 0.45, fontFace: "Calibri", fontSize: 15, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText([{ text: h1, options: { bold: true, fontSize: 15, color: C.accent1, breakLine: true } }, { text: t, options: { fontSize: 14 } }], { ...body, x: 7.7, y, w: 1.85, h: 0.85 });
  });
  s.addNotes("Демо ~5 мин (см. сценарий в отчёте). Запуск: streamlit run app/streamlit_app.py (порт 8501). На слайде — скриншот на случай сбоя. Шаги: открыть демо; показать данные и проверку; выбрать товар; показать график и рекомендацию; изменить срок поставки до 7 дней и уровень до 98% и показать, как меняется рекомендация; открыть вкладку плана закупок и скачать CSV; показать вкладку ограничений.");
}

// 12. Limits
{
  const s = content("Ограничения и ответственное использование", "Демо и итоги");
  card(s, 0.5, 1.45, 4.4, 3.6, "Ограничения");
  s.addText("Ограничения", { ...body, x: 0.65, y: 1.55, w: 4.1, h: 0.35, fontSize: 16, bold: true, color: C.accent5 });
  s.addText(bullets([
    "Один магазин: на другом бизнесе точность не проверена.",
    "Нет остатков и закупочных цен: закупки проверены на симуляции, прибыль не оценивалась.",
    "Продажи — не спрос: нехватка товара скрывает спрос.",
    "Прогноз не знает акций и праздников.",
  ], 15, 8), { ...body, x: 0.65, y: 1.95, w: 4.1, h: 3.0 });
  card(s, 5.1, 1.45, 4.4, 3.6, "Меры");
  s.addText("Меры", { ...body, x: 5.25, y: 1.55, w: 4.1, h: 0.35, fontSize: 16, bold: true, color: C.accent1 });
  s.addText(bullets([
    "Решение о заказе принимает человек, план — рекомендация.",
    "Статусы «новый товар» и «нет продаж 28 дней»: такие товары не планируются.",
    "Перед внедрением сравнить с текущим методом закупок.",
    "Только данные о продажах, без персональной информации; код и результаты воспроизводимы.",
  ], 15, 8), { ...body, x: 5.25, y: 1.95, w: 4.1, h: 3.0 });
  s.addNotes("~30 с. Ограничения называем сами: один магазин, нет остатков и цен, продажи не равны спросу, прогноз не знает акций. Меры: человек принимает решение, проблемные товары помечены, перед внедрением нужно сравнить с текущим методом. Персональных данных система не использует.");
}

// 13. Closing
{
  const s = pres.addSlide({ masterName: "LIGHTTITLE", sectionTitle: "Демо и итоги" });
  s.addText("Итоги", { placeholder: "title" });
  s.addText(bullets([
    "Собрана вся цепочка: данные, прогноз, закупки, приложение.",
    "Проверка честная: тест открыт один раз, итог скромный: выигрыш над простым средним мал и нестабилен.",
    "Дальше: данные реального магазина, остатки и акции.",
  ], 16, 10).map((r) => ({ ...r, options: { ...r.options, color: C.text1 } })), { x: 0.6, y: 1.5, w: 4.7, h: 3.3, fontFace: "Calibri", margin: 0, isTextBox: true, valign: "top" });
  card(s, 5.6, 1.5, 3.8, 2.6, "Вклад участников", C.background1);
  s.addText("Вклад участников", { ...body, x: 5.75, y: 1.6, w: 3.5, h: 0.35, fontSize: 16, bold: true, color: C.accent1 });
  s.addText([
    { text: "[Имя Фамилия]", options: { bold: true, fontSize: 14, breakLine: true } },
    { text: "[данные, эксперименты, метрики]", options: { fontSize: 13, color: C.text2, breakLine: true, paraSpaceAfter: 8 } },
    { text: "[Имя Фамилия]", options: { bold: true, fontSize: 14, breakLine: true } },
    { text: "[модель, закупки, симуляция]", options: { fontSize: 13, color: C.text2, breakLine: true, paraSpaceAfter: 8 } },
    { text: "[Имя Фамилия]", options: { bold: true, fontSize: 14, breakLine: true } },
    { text: "[приложение, инженерия, отчёт]", options: { fontSize: 13, color: C.text2 } },
  ], { ...body, x: 5.75, y: 2.05, w: 3.5, h: 2.7 });
  s.addText("Вопросы?", { x: 0.6, y: 4.75, w: 4.7, h: 0.5, fontFace: "Cambria", fontSize: 24, bold: true, color: C.accent5, margin: 0, isTextBox: true });
  s.addNotes("~20 с. Итог: собрана вся цепочка и проверена честно. Скромный результат — тоже результат: мы знаем, где выигрыш есть, а где нет. Заполните вклад участников и переходите к вопросам.");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("written", OUT);
})();
