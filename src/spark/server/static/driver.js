import { api, bayMap, clock, el, liveStream, money, updateBayMap } from "./common.js";

const list = document.getElementById("list");
const detail = document.getElementById("detail");
const announce = document.getElementById("announce");
let lots = [];
let here = null;
let open = null; // { id, status, geometry, map }

function state(lot) {
  if (lot.free === 0) return "full";
  return lot.free / lot.capacity < 0.15 ? "tight" : "ok";
}

function lotCard(lot) {
  const occ = lot.capacity ? (lot.occupied / lot.capacity) * 100 : 0;
  const where = lot.distance_m != null ? ` · ${(lot.distance_m / 1000).toFixed(1)} km` : "";
  return el(
    "button",
    { class: `card lot ${state(lot)}`, type: "button", onclick: () => show(lot.id),
      "aria-label": `${lot.name}: ${lot.free} of ${lot.capacity} free` },
    el("span", { class: "name" }, lot.name),
    el("span", { class: "free" }, el("b", {}, lot.free), el("span", {}, `of ${lot.capacity} free`)),
    el("span", { class: "muted small" },
      `${money(lot.price, lot.currency)}/h${where} · updated ${clock(lot.updated)}`),
    el("span", { class: "bar", "aria-hidden": "true" }, el("i", { style: `width:${occ}%` })),
  );
}

function renderList() {
  list.replaceChildren(
    ...(lots.length ? lots.map(lotCard) : [el("p", { class: "card empty" }, "No car parks are reporting yet.")]),
  );
}

async function loadList() {
  lots = here
    ? await api(`/api/nearby?lat=${here.lat}&lon=${here.lon}&radius_m=20000`)
    : await api("/api/lots");
  renderList();
}

async function show(id) {
  const [status, geometry, transit] = await Promise.all([
    api(`/api/lots/${id}`), api(`/api/lots/${id}/geometry`), api(`/api/lots/${id}/transit`),
  ]);
  const map = bayMap(geometry, status);
  open = { id, status, geometry, map };
  const others = lots.filter((l) => l.id !== id && l.free > 0).slice(0, 3);
  detail.replaceChildren(...[
    el("button", { class: "back", type: "button", onclick: hide }, "← All car parks"),
    el("div", { class: "card" },
      el("h1", { tabindex: "-1" }, status.name),
      el("p", { class: "muted", id: "detail-line" }, line(status)),
      map,
      el("div", { class: "legend" },
        el("span", {}, el("i", { style: "background:var(--free)" }), "free"),
        el("span", {}, el("i", { style: "background:var(--busy)" }), "occupied"),
        el("span", {}, el("i", { style: "background:var(--unknown)" }), "no reading")),
    ),
    el("div", { class: "card" },
      el("h2", {}, "Leave the car and ride"),
      transit.length
        ? el("ul", { class: "plain" }, transit.map((s) =>
            el("li", {}, el("span", { class: "grow" }, s.name,
              s.routes.length ? el("div", { class: "muted small" }, s.routes.join(" · ")) : null),
              el("span", { class: "muted small num" }, `${s.walk_min} min walk`))))
        : el("p", { class: "muted" }, "No transit stops loaded for this car park.")),
    others.length
      ? el("div", { class: "card" }, el("h2", {}, "Other car parks with space"),
          el("ul", { class: "plain" }, others.map((o) =>
            el("li", {}, el("span", { class: "grow" }, o.name),
              el("b", { class: "num" }, `${o.free} free`)))))
      : null,
  ].filter(Boolean));
  list.hidden = true;
  detail.hidden = false;
  history.pushState({ id }, "", `#${encodeURIComponent(id)}`);
  detail.querySelector("h1").focus();
}

function line(s) {
  return `${s.free} of ${s.capacity} free · ${money(s.price, s.currency)} per hour · updated ${clock(s.updated)}`;
}

function hide() {
  open = null;
  detail.hidden = true;
  list.hidden = false;
  if (location.hash) history.pushState({}, "", location.pathname);
}

window.addEventListener("popstate", () => {
  const id = decodeURIComponent(location.hash.slice(1));
  if (id) show(id); else hide();
});

document.getElementById("locate").addEventListener("click", () => {
  if (!navigator.geolocation) return;
  navigator.geolocation.getCurrentPosition(
    (pos) => { here = { lat: pos.coords.latitude, lon: pos.coords.longitude }; loadList(); },
    () => { announce.textContent = "Location unavailable; showing every car park."; },
    { enableHighAccuracy: false, timeout: 8000 },
  );
});

liveStream(async (update) => {
  const i = lots.findIndex((l) => l.id === update.id);
  if (i >= 0) lots[i] = { ...lots[i], ...update };
  else lots.push(update);
  renderList();
  if (open && open.id === update.id) {
    open.status = await api(`/api/lots/${update.id}`);
    updateBayMap(open.map, open.status);
    document.getElementById("detail-line").textContent = line(open.status);
    if (update.free === 0) announce.textContent = `${update.name} is now full.`;
  }
}, document.getElementById("live"));

await loadList();
if (location.hash) show(decodeURIComponent(location.hash.slice(1)));
if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js").catch(() => {});
