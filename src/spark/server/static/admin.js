import { api, bayMap, clock, el, liveStream, money, svg, updateBayMap } from "./common.js";

const select = document.getElementById("lot");
const tokenInput = document.getElementById("token");
let current = null;
let map = null;
let refreshTimer = null;

try { tokenInput.value = sessionStorage.getItem("spark-token") ?? ""; } catch {}
tokenInput.addEventListener("change", () => {
  try { sessionStorage.setItem("spark-token", tokenInput.value); } catch {}
});

function kpi(label, value, sub) {
  return el("div", { class: "card kpi" },
    el("div", { class: "label" }, label), el("div", { class: "value" }, value),
    sub ? el("div", { class: "sub" }, sub) : null);
}

function pct(x) { return x == null ? "–" : `${Math.round(x * 100)}%`; }

function renderKpis(s, a) {
  document.getElementById("kpis").replaceChildren(
    kpi("Free now", `${s.free} / ${s.capacity}`, `updated ${clock(s.updated, true)}`),
    kpi("Price", money(s.price, s.currency), "per hour"),
    kpi("Time full", pct(a.share_of_time_full), "share of snapshots at 95% or more"),
    kpi("Median stay", a.dwell_minutes.median == null ? "–" : `${Math.round(a.dwell_minutes.median)} min`,
      `${a.dwell_minutes.n} completed stays`),
    kpi("Turnover", a.turnover_per_bay_per_day.toFixed(2), "arrivals per bay per day"),
    kpi("Observed", `${a.days_observed.toFixed(1)} days`, a.peak ? `peak ${pct(a.peak.occupancy)} on ${clock(a.peak.t, true)}` : ""),
  );
}

function renderTrend(history, prices, policy) {
  const root = document.getElementById("trend");
  root.replaceChildren();
  if (history.length < 2) {
    root.append(svg("text", { x: 10, y: 24, class: "tick" }, document.createTextNode("Waiting for data")));
    return;
  }
  const W = 640, H = 220, L = 40, R = 44, T = 10, B = 24;
  root.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const t0 = history[0].t, t1 = history.at(-1).t;
  const x = (t) => L + ((t - t0) / Math.max(t1 - t0, 1)) * (W - L - R);
  const y = (v) => T + (1 - v) * (H - T - B);
  const maxP = Math.max(policy.max_price, ...prices.map((p) => p.price));
  const yp = (v) => T + (1 - v / maxP) * (H - T - B);
  root.append(svg("rect", { class: "band", x: L, width: W - L - R,
    y: y(policy.target_high), height: y(policy.target_low) - y(policy.target_high) }));
  for (const v of [0, 0.5, 1]) {
    root.append(svg("line", { class: "axis", x1: L, x2: W - R, y1: y(v), y2: y(v) }));
    root.append(svg("text", { class: "tick", x: 4, y: y(v) + 4 }, document.createTextNode(pct(v))));
  }
  root.append(svg("text", { class: "tick", x: W - R + 6, y: yp(maxP) + 10 },
    document.createTextNode(maxP.toFixed(2))));
  // Break the line where observations stop (nights, days without footage) instead of
  // drawing a straight line across the gap.
  const gaps = history.slice(1).map((h, i) => h.t - history[i].t).sort((a, b) => a - b);
  const maxGap = Math.max(4 * (gaps[Math.floor(gaps.length / 2)] ?? 60), 2 * 3600);
  const runs = [[]];
  history.forEach((h, i) => {
    if (i && h.t - history[i - 1].t > maxGap) runs.push([]);
    runs.at(-1).push([x(h.t), y(h.known ? h.occupied / h.known : 0)]);
  });
  for (const pts of runs.filter((r) => r.length > 1)) {
    root.append(svg("path", { class: "area",
      d: `M${pts[0][0]},${y(0)} ` + pts.map((p) => `L${p[0]},${p[1]}`).join(" ") + ` L${pts.at(-1)[0]},${y(0)} Z` }));
    root.append(svg("path", { class: "occ", d: "M" + pts.map((p) => p.join(",")).join(" L") }));
  }
  const inRange = prices.filter((p) => p.t <= t1);
  if (inRange.length) {
    let d = "";
    let last = null;
    for (const p of inRange) {
      const px = Math.max(x(p.t), L);
      d += last == null ? `M${px},${yp(p.price)}` : ` L${px},${yp(last)} L${px},${yp(p.price)}`;
      last = p.price;
    }
    d += ` L${x(t1)},${yp(last)}`;
    root.append(svg("path", { class: "price", d }));
  }
  const ticks = 5;
  const multiDay = t1 - t0 > 2 * 86400;
  const fmt = (t) => multiDay
    ? new Intl.DateTimeFormat(undefined, { day: "numeric", month: "short", timeZone: "UTC" }).format(new Date(t * 1000))
    : clock(t);
  for (let i = 0; i <= ticks; i++) {
    const t = t0 + ((t1 - t0) * i) / ticks;
    root.append(svg("text", { class: "tick", x: x(t) - 16, y: H - 6 }, document.createTextNode(fmt(t))));
  }
}

const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
function renderHeat(profile) {
  const root = document.getElementById("heat");
  const cells = [el("div", { class: "lab" })];
  for (let h = 0; h < 24; h++) cells.push(el("div", { class: "lab" }, h % 6 === 0 ? h : ""));
  profile.forEach((row, d) => {
    cells.push(el("div", { class: "lab" }, DAYS[d]));
    for (let h = 0; h < 24; h++) {
      const v = row[h];
      const style = v == null
        ? "background:var(--line)"
        : `background:color-mix(in srgb, var(--busy) ${Math.round(v * 100)}%, var(--free))`;
      cells.push(el("div", { style, title: `${DAYS[d]} ${h}:00 · ${v == null ? "no data" : pct(v)}` }));
    }
  });
  root.replaceChildren(...cells);
}

function renderPrices(p) {
  const pol = p.policy;
  document.getElementById("policy").textContent =
    `${pol.mode} · target ${pct(pol.target_low)}–${pct(pol.target_high)} · step ${pol.step} · every ${Math.round(pol.review_every_s / 60)} min`;
  document.getElementById("prices").replaceChildren(
    ...p.history.slice().reverse().slice(0, 100).map((r) =>
      el("tr", {}, el("td", {}, clock(r.t, true)), el("td", { class: "num" }, money(r.price, p.currency)),
        el("td", { class: "num" }, pct(r.occupancy)), el("td", {}, r.reason))));
}

async function load(id) {
  current = id;
  const [s, geo, a, hist, p] = await Promise.all([
    api(`/api/lots/${id}`), api(`/api/lots/${id}/geometry`), api(`/api/lots/${id}/analytics`),
    api(`/api/lots/${id}/history?limit=4000`), api(`/api/lots/${id}/prices`),
  ]);
  renderKpis(s, a);
  renderTrend(hist, p.history, p.policy);
  renderHeat(a.hour_of_week);
  renderPrices(p);
  map = bayMap(geo, s);
  document.getElementById("bays").replaceChildren(map);
  document.getElementById("bays-line").textContent = `${s.free} free, ${s.occupied} occupied, ${s.unknown} without a reading`;
}

document.getElementById("review").addEventListener("click", async () => {
  const headers = tokenInput.value ? { Authorization: `Bearer ${tokenInput.value}` } : {};
  try {
    await api(`/api/lots/${current}/prices/review`, { method: "POST", headers });
    await load(current);
  } catch (e) {
    alert(`Review failed: ${e.message}`);
  }
});

select.addEventListener("change", () => load(select.value));

liveStream(async (update) => {
  if (update.id !== current) return;
  const s = await api(`/api/lots/${current}`);
  if (map) updateBayMap(map, s);
  document.getElementById("bays-line").textContent = `${s.free} free, ${s.occupied} occupied, ${s.unknown} without a reading`;
  // Throttle, not debounce: under a steady stream a debounce would never fire.
  if (!refreshTimer) {
    refreshTimer = setTimeout(() => { refreshTimer = null; load(current); }, 1500);
  }
}, document.getElementById("live"));

const lots = await api("/api/lots");
select.replaceChildren(...lots.map((l) => el("option", { value: l.id }, l.name)));
if (lots.length) await load(lots[0].id);
else document.getElementById("kpis").replaceChildren(el("p", { class: "card empty" }, "No car parks registered."));
