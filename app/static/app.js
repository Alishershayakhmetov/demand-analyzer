"use strict";
// Demand Copilot front end: plain JavaScript, no libraries; charts are drawn as SVG.
const HIST = "#9FB1C9", FC = "#3A6EA5";
const state = { lang: "ru", source: "demo", meta: null, csvText: null, csvName: "", header: [], cols: {}, stockText: null, stockName: "",
  synthetic: true, lead: 3, service: 95, data: null, plan: null, tab: "fc", item: null, onlyOrder: true, busy: false, error: null, openChecks: null };
const T = () => I18N[state.lang];
const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const loc = () => (state.lang === "ru" ? "ru-RU" : "en-US");
const fmt = (v, d = 0) => Number(v).toLocaleString(loc(), { maximumFractionDigits: d, minimumFractionDigits: 0 });
const bold = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
const DAY = 86400000;
const isoToMs = (iso) => Date.parse(iso + "T00:00:00Z");
const dayLabel = (ms) => new Date(ms).toLocaleDateString(loc(), { day: "2-digit", month: "short", timeZone: "UTC" });
const shortLabel = (ms) => new Date(ms).toISOString().slice(5, 10);

// ---------- data loading ----------
async function api(path, opts) {
  const r = await fetch(path, opts);
  let body = {};
  try { body = await r.json(); } catch (e) { /* not JSON */ }
  if (!r.ok) throw new Error(body.error || `${T().server_error} ${r.status}`);
  return body;
}
function errorText(e) {
  const m = String(e.message || e);
  if (m === "no_model") return T().no_model;
  if (m === "no_demo") return T().no_demo;
  return `${T().bad_input} ${m}`;
}
async function analyze() {
  if (state.source === "csv" && !state.csvText) { state.data = null; state.plan = null; state.error = null; return render(); }
  state.busy = true; state.error = null; render();
  try {
    const payload = state.source === "demo"
      ? { source: "demo", synthetic_stock: state.synthetic }
      : { source: "csv", csv: state.csvText, cols: state.cols, stock_csv: state.stockText };
    const res = await api("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    state.data = res;
    state.item = res.items.length ? res.items[0].id : null;
    state.plan = await api(planUrl("/api/plan"));
  } catch (e) {
    state.data = null; state.plan = null; state.error = errorText(e);
  }
  state.busy = false; render();
}
const planUrl = (base) => `${base}?token=${state.data.token}&lead=${state.lead}&service=${state.service}`;
async function reloadPlan() {
  if (!state.data) return;
  try { state.plan = await api(planUrl("/api/plan")); state.error = null; } catch (e) { state.error = errorText(e); }
  render();
}

// ---------- csv helpers ----------
function splitHeader(text) {
  const line = text.replace(/^﻿/, "").split(/\r?\n/, 1)[0];
  const sep = [",", ";", "\t"].map((s) => [s, line.split(s).length]).sort((a, b) => b[1] - a[1])[0][0];
  const re = new RegExp(sep.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + '(?=(?:[^"]*"[^"]*")*[^"]*$)');
  return line.split(re).map((c) => c.trim().replace(/^"|"$/g, ""));
}
function guess(cols, words, fallback) {
  const i = cols.findIndex((c) => words.some((w) => c.toLowerCase().includes(w)));
  return cols[i >= 0 ? i : Math.min(fallback, cols.length - 1)];
}
function setHeader(text) {
  state.header = splitHeader(text);
  const h = state.header;
  state.cols = { date: guess(h, ["date", "day", "дата"], 0), item: guess(h, ["item", "sku", "product", "товар", "article"], 1),
                 sales: guess(h, ["sale", "qty", "unit", "quant", "прод", "кол"], 2), price: "" };
  const p = h.find((c) => /price|цена/i.test(c));
  if (p) state.cols.price = p;
}

// ---------- panel ----------
function renderPanel() {
  const t = T(), m = state.meta || {};
  const opt = (v, label) => `<option value="${v}"${String(state.lead) === String(v) ? " selected" : ""}>${label}</option>`;
  const sel = (id, label, options, cur, extra = "") => `<label class="field" for="${id}">${esc(label)}</label><select id="${id}" ${extra}>` +
    options.map((o) => `<option value="${esc(o)}"${o === cur ? " selected" : ""}>${esc(o)}</option>`).join("") + "</select>";
  let data = `<label class="opt${m.demo ? "" : " disabled"}"><input type="radio" name="src" value="demo" ${state.source === "demo" ? "checked" : ""} ${m.demo ? "" : "disabled"}> <span>${esc(t.demo)}</span></label>` +
    `<label class="opt"><input type="radio" name="src" value="csv" ${state.source === "csv" ? "checked" : ""}> <span>${esc(t.upload)}</span></label>`;
  if (state.source === "demo") {
    data += `<label class="opt"><input type="checkbox" id="synthetic" ${state.synthetic ? "checked" : ""}> <span>${esc(t.demo_stock)}</span></label>`;
  } else {
    data += `<label class="field">${esc(t.file)}</label><button class="btn" id="pick-csv" type="button">${esc(t.file)}…</button><input id="file-csv" type="file" accept=".csv,text/csv,text/plain" hidden>` +
      `<span class="filename">${esc(state.csvName || t.no_file)}</span>`;
    if (state.header.length) {
      const h = state.header;
      data += sel("col-date", t.col_date, h, state.cols.date) + sel("col-item", t.col_item, h, state.cols.item) + sel("col-sales", t.col_sales, h, state.cols.sales) +
        sel("col-price", t.col_price, [t.none, ...h], state.cols.price || t.none);
    }
    data += `<label class="field">${esc(t.stock_file)}</label><button class="btn" id="pick-stock" type="button">${esc(t.stock_file.split(" (")[0])}…</button>` +
      `<input id="file-stock" type="file" accept=".csv,text/csv,text/plain" hidden><span class="filename">${esc(state.stockName || t.no_file)}</span>`;
  }
  $("#panel").innerHTML =
    `<label class="field" style="margin-top:0">Language / Язык</label><div class="seg" role="group"><button data-lang="ru" aria-pressed="${state.lang === "ru"}">ru</button><button data-lang="en" aria-pressed="${state.lang === "en"}">en</button></div>` +
    `<h3>${esc(t.source)}</h3>${data}<h3>${esc(t.params)}</h3>` +
    `<label class="field" for="lead">${esc(t.lead)}</label><select id="lead">${[1, 3, 7].map((v) => opt(v, v)).join("")}</select>` +
    `<label class="field" for="service" title="${esc(t.service_help)}">${esc(t.service)} ⓘ</label><select id="service" title="${esc(t.service_help)}">` +
    [90, 95, 98].map((v) => `<option value="${v}"${state.service === v ? " selected" : ""}>${v}</option>`).join("") + "</select>" +
    `<p class="note">${esc(t.review)}</p>`;
  bindPanel();
}
function bindPanel() {
  const p = $("#panel");
  p.querySelectorAll("[data-lang]").forEach((b) => b.addEventListener("click", () => { state.lang = b.dataset.lang; document.documentElement.lang = state.lang; render(); }));
  p.querySelectorAll("input[name=src]").forEach((r) => r.addEventListener("change", () => { state.source = r.value; renderPanel(); analyze(); }));
  const syn = $("#synthetic"); if (syn) syn.addEventListener("change", () => { state.synthetic = syn.checked; analyze(); });
  const pick = (btn, input, onText) => {
    const b = $(btn), i = $(input); if (!b) return;
    b.addEventListener("click", () => i.click());
    i.addEventListener("change", async () => { const f = i.files[0]; if (f) { onText(f.name, await f.text()); } });
  };
  pick("#pick-csv", "#file-csv", (name, text) => { state.csvName = name; state.csvText = text; setHeader(text); renderPanel(); analyze(); });
  pick("#pick-stock", "#file-stock", (name, text) => { state.stockName = name; state.stockText = text; renderPanel(); analyze(); });
  [["#col-date", "date"], ["#col-item", "item"], ["#col-sales", "sales"]].forEach(([id, key]) => { const e = $(id); if (e) e.addEventListener("change", () => { state.cols[key] = e.value; analyze(); }); });
  const price = $("#col-price"); if (price) price.addEventListener("change", () => { state.cols.price = price.value === T().none ? "" : price.value; analyze(); });
  $("#lead").addEventListener("change", (e) => { state.lead = Number(e.target.value); reloadPlan(); });
  $("#service").addEventListener("change", (e) => { state.service = Number(e.target.value); reloadPlan(); });
}

// ---------- charts ----------
function niceTicks(max, n = 5) {
  if (max <= 0) return [0, 1];
  const raw = max / n, mag = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / mag;
  const step = (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * mag, out = [];
  for (let v = 0; v <= max + step * 0.999; v += step) out.push(+v.toFixed(10));
  return out;
}
const tipEl = () => $("#tip");
function showTip(ev, text) { const t = tipEl(); t.textContent = text; t.hidden = false; t.style.left = ev.clientX + 12 + "px"; t.style.top = ev.clientY - 30 + "px"; }
function hideTip() { tipEl().hidden = true; }

function lineChart(hist, fc, histStart, fcStart) {
  const W = 760, H = 300, L = 44, R = 12, Tp = 10, B = 28;
  const x0 = isoToMs(histStart), pts = hist.map((v, i) => [x0 + i * DAY, v]), f0 = isoToMs(fcStart), fpts = fc.map((v, i) => [f0 + i * DAY, v]);
  const xmin = x0, xmax = fpts[fpts.length - 1][0];
  const ticks = niceTicks(Math.max(...hist, ...fc) * 1.05), ymax = ticks[ticks.length - 1];
  const X = (ms) => L + ((ms - xmin) / (xmax - xmin)) * (W - L - R), Y = (v) => Tp + (1 - v / ymax) * (H - Tp - B);
  const path = (a) => a.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(T().daily)}"><g class="axis">`;
  svg += `<rect x="${X(f0 - DAY / 2)}" y="${Tp}" width="${W - R - X(f0 - DAY / 2)}" height="${H - Tp - B}" fill="#EEF2F8"/>`;
  ticks.forEach((v) => { svg += `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="#DDE4EE"/><text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${fmt(v, v < 10 ? 1 : 0)}</text>`; });
  for (let ms = x0 + 7 * DAY; ms <= xmax; ms += 14 * DAY) svg += `<text x="${X(ms)}" y="${H - 8}" text-anchor="middle">${dayLabel(ms)}</text>`;
  svg += `</g><path d="${path(pts)}" fill="none" stroke="${HIST}" stroke-width="1.6"/><path d="${path(fpts)}" fill="none" stroke="${FC}" stroke-width="2.2"/>`;
  svg += `<line id="cursor" y1="${Tp}" y2="${H - B}" stroke="#5D6B82" stroke-dasharray="3 3" visibility="hidden"/><rect id="hit" x="${L}" y="${Tp}" width="${W - L - R}" height="${H - Tp - B}" fill="transparent"/></svg>`;
  const html = `<div class="chart">${svg}<div class="legend"><span><i class="dot" style="background:${HIST}"></i>${esc(T().hist_series)}</span><span><i class="dot" style="background:${FC}"></i>${esc(T().fc_series)}</span></div></div>`;
  return { html, bind(root) {
    const hit = $("#hit", root), cur = $("#cursor", root), svgEl = $("svg", root), all = pts.concat(fpts);
    hit.addEventListener("mousemove", (ev) => {
      const r = svgEl.getBoundingClientRect(), vx = ((ev.clientX - r.left) / r.width) * W, ms = xmin + ((vx - L) / (W - L - R)) * (xmax - xmin);
      const k = Math.max(0, Math.min(all.length - 1, Math.round((ms - xmin) / DAY))), p = all[k];
      cur.setAttribute("x1", X(p[0])); cur.setAttribute("x2", X(p[0])); cur.setAttribute("visibility", "visible");
      showTip(ev, `${dayLabel(p[0])}: ${fmt(p[1], 2)} ${k >= pts.length ? "(" + T().fc_series + ")" : ""}`);
    });
    hit.addEventListener("mouseleave", () => { hideTip(); cur.setAttribute("visibility", "hidden"); });
  } };
}
function barChart(labels, past, ahead) {
  const W = 760, H = 230, L = 44, R = 12, Tp = 18, B = 28, n = labels.length;
  const vals = [...past, ...ahead], ticks = niceTicks(Math.max(...vals) * 1.1), ymax = ticks[ticks.length - 1];
  const Y = (v) => Tp + (1 - v / ymax) * (H - Tp - B), bw = (W - L - R) / n;
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(T().weekly)}"><g class="axis">`;
  ticks.forEach((v) => { svg += `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="#DDE4EE"/><text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${fmt(v, v < 10 ? 1 : 0)}</text>`; });
  vals.forEach((v, i) => {
    const x = L + i * bw + bw * 0.15, w = bw * 0.7, y = Y(v), col = i < past.length ? HIST : FC;
    svg += `<rect x="${x}" y="${y}" width="${w}" height="${H - B - y}" fill="${col}" rx="2"><title>${labels[i]}: ${fmt(v, 1)}</title></rect>` +
      `<text x="${x + w / 2}" y="${y - 4}" text-anchor="middle">${fmt(v, v < 10 ? 1 : 0)}</text><text x="${x + w / 2}" y="${H - 8}" text-anchor="middle">${labels[i]}</text>`;
  });
  svg += "</g></svg>";
  return `<div class="chart">${svg}<div class="legend"><span><i class="dot" style="background:${HIST}"></i>${esc(T().week_actual)}</span><span><i class="dot" style="background:${FC}"></i>${esc(T().week_forecast)}</span></div></div>`;
}

// ---------- main view ----------
const metric = (label, value, sm = false) => `<div class="metric"><div class="label">${esc(label)}</div><div class="value${sm ? " sm" : ""}">${value}</div></div>`;
const statusName = (s) => ({ ok: T().st_ok, new: T().st_new, dormant: T().st_dormant }[s] || s);
const tag = (s) => `<span class="tag ${s}">${esc(statusName(s))}</span>`;

function forecastTab() {
  const t = T(), d = state.data, it = d.items.find((x) => x.id === state.item) || d.items[0];
  const checks = `<details class="card"${state.openChecks ?? d.notes.length ? " open" : ""}><summary>${esc(t.data_check)}</summary><div class="metrics">` +
    metric(t.items, fmt(d.n_items)) + metric(t.history, fmt(d.n_days)) + metric(t.start, `${d.forecast_from} ${t.end} ${d.forecast_to}`, true) + "</div>" +
    d.notes.map((n) => `<div class="msg warn">${esc(n)}</div>`).join("") + (d.notes.length ? "" : `<p class="note">${esc(t.warn_last)}</p>`) +
    (d.n_new || d.n_dormant ? `<div class="msg info">${esc(t.st_new)}: ${d.n_new}; ${esc(t.st_dormant)}: ${d.n_dormant}</div>` : "") + "</details>";
  const chart = lineChart(it.hist, it.fc, d.hist_start, d.forecast_from);
  const row = state.plan && state.plan.rows.find((r) => r.item_id === it.id);
  const rec = row ? `<h2>${esc(t.item_plan.replace("{lead}", state.lead).replace("{service}", state.service))}</h2><div class="metrics">` +
    metric(t.m_target, fmt(row.target_level)) + metric(t.m_onhand, fmt(row.on_hand)) + metric(t.m_onorder, fmt(row.on_order)) + metric(t.m_order, fmt(row.order_qty)) + "</div>" : "";
  const starts = Array.from({ length: 8 }, (_, w) => isoToMs(d.forecast_from) + (w - 4) * 7 * DAY);
  const vals = [...it.past, ...it.ahead];
  const weeklyRows = starts.map((ms, i) => `<tr><td>${new Date(ms).toISOString().slice(0, 10)}</td><td>${esc(i < 4 ? t.week_actual : t.week_forecast)}</td><td>${fmt(vals[i], vals[i] < 10 ? 1 : 0)}</td></tr>`).join("");
  return checks + `<div class="metrics">${metric(t.next7, fmt(d.next7))}${metric(t.last7, fmt(d.last7))}</div>` +
    `<label class="field" for="item">${esc(t.pick)}</label><select id="item">${d.items.map((x) => `<option${x.id === it.id ? " selected" : ""}>${esc(x.id)}</option>`).join("")}</select>` +
    `<h2>${esc(t.daily)}</h2><p class="note">${esc(t.chart_note)}</p>${chart.html}${rec}<h2>${esc(t.weekly)}</h2>${barChart(starts.map(shortLabel), it.past, it.ahead)}` +
    `<table class="t" style="max-width:420px;margin-top:10px"><thead><tr><th>${esc(t.week)}</th><th>${esc(t.kind)}</th><th>${esc(t.units)}</th></tr></thead><tbody>${weeklyRows}</tbody></table>` +
    `<div class="msg info">${esc(t.flat_note)}</div>`;
}
function planTab() {
  const t = T(), p = state.plan, d = state.data;
  if (!p) return "";
  const rows = state.onlyOrder ? p.rows.filter((r) => r.order_qty > 0) : p.rows;
  const cols = Object.keys(t.cols).filter((c) => p.columns.includes(c));
  const cell = (c, r) => c === "status" ? tag(r.status) : c === "item_id" ? esc(r[c]) : fmt(r[c], c === "forecast_next_7d" || c === "target_level" || c === "safety_stock" || c === "order_value" ? 1 : 0);
  return `<div class="metrics">${metric(t.n_order, fmt(p.n_order))}${metric(t.units_order, fmt(p.units))}${p.value != null ? metric(t.value_order, fmt(p.value)) : ""}${metric(t.flags, fmt(p.flagged))}</div>` +
    `<p class="note">${esc(d.has_stock ? t.plan_note_stock : t.plan_note)}</p>` +
    `<label class="opt"><input type="checkbox" id="only-order" ${state.onlyOrder ? "checked" : ""}> <span>${esc(t.only_order)}</span></label>` +
    `<div class="scroll"><table class="t"><thead><tr>${cols.map((c) => `<th>${esc(t.cols[c])}</th>`).join("")}</tr></thead><tbody>` +
    rows.map((r) => `<tr>${cols.map((c) => `<td>${cell(c, r)}</td>`).join("")}</tr>`).join("") + "</tbody></table></div>" +
    `<p><a class="btn" href="${planUrl("/api/plan.csv")}" download="purchase_plan.csv">${esc(t.download)}</a></p>`;
}
function qualityTab() {
  const t = T(), m = state.meta || {};
  const head = `<tr><th>${esc(t.model)}</th><th>${esc(t.daily_wape)}</th><th>${esc(t.weekly_wape)}</th><th>${esc(t.wape_28d)}</th>`;
  const tbl = (rows, inv) => `<table class="t"><thead>${head}${inv ? `<th>${esc(t.inv)}</th>` : ""}</tr></thead><tbody>` + rows.map((r) =>
    `<tr><td>${esc(r.model)}</td><td>${r.daily_wape}</td><td>${r.weekly_wape}</td><td>${r.wape_28d}</td>${inv ? `<td>${r.inv_days_95}</td>` : ""}</tr>`).join("") + "</tbody></table>";
  return `<p class="note">${esc(t.quality_note)}</p>` +
    (m.quality_val ? `<h3>${esc(t.quality_tbl)}</h3>${tbl(m.quality_val, true)}` : "") +
    (m.quality_test ? `<h3>${esc(t.test_tbl)}</h3>${tbl(m.quality_test, false)}` : "") +
    (m.frontier_image ? `<p><img class="fig" src="/reports/inventory_frontier.png" alt="${esc(t.quality_img)}"><span class="small">${esc(t.quality_img)}</span></p>` : "") +
    `<h2>${esc(t.limits_title)}</h2><ul class="limits">${t.limits.map((l) => `<li>${bold(l)}</li>`).join("")}</ul>`;
}
function renderMain() {
  const t = T();
  let html = `<h1>${esc(t.title)}</h1><p class="lead">${esc(t.intro)}</p>`;
  if (state.error) html += `<div class="msg error">${esc(state.error)}</div>`;
  if (state.busy) html += `<div class="msg spin"><span class="spinner"></span>${esc(t.calculating)}</div>`;
  else if (!state.data && !state.error) html += `<div class="msg info">${esc(t.no_data)}</div>`;
  if (state.data && !state.busy) {
    const tabs = [["fc", t.tab_fc], ["plan", t.tab_plan], ["q", t.tab_quality]];
    html += `<div class="tabs" role="tablist">${tabs.map(([k, l]) => `<button class="tab" role="tab" data-tab="${k}" aria-selected="${state.tab === k}">${esc(l)}</button>`).join("")}</div>`;
    html += state.tab === "fc" ? forecastTab() : state.tab === "plan" ? planTab() : qualityTab();
  } else if (!state.data) {
    html += `<div class="tabs"><button class="tab" data-tab="q" aria-selected="true">${esc(t.tab_quality)}</button></div>` + qualityTab();
  }
  const root = $("#main"), scrollY = window.scrollY;
  root.innerHTML = html;
  root.querySelectorAll("[data-tab]").forEach((b) => b.addEventListener("click", () => { state.tab = b.dataset.tab; render(); }));
  const item = $("#item", root); if (item) item.addEventListener("change", () => { state.item = item.value; render(); });
  const only = $("#only-order", root); if (only) only.addEventListener("change", () => { state.onlyOrder = only.checked; render(); });
  const det = $("details.card", root); if (det) det.addEventListener("toggle", () => { state.openChecks = det.open; });
  if (state.data && !state.busy && state.tab === "fc") {
    const it = state.data.items.find((x) => x.id === state.item) || state.data.items[0];
    lineChart(it.hist, it.fc, state.data.hist_start, state.data.forecast_from).bind(root);
  }
  window.scrollTo(0, scrollY);
}
function render() { document.title = T().app_name; renderPanel(); renderMain(); }

(async function init() {
  render();
  try { state.meta = await api("/api/meta"); } catch (e) { state.error = errorText(e); }
  if (state.meta && !state.meta.demo) state.source = "csv";
  render();
  if (state.meta && state.meta.model && state.source === "demo") analyze();
  else if (state.meta && !state.meta.model) { state.error = T().no_model; render(); }
})();
