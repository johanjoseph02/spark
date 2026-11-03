// Shared helpers for the driver app and the operator dashboard. No dependencies.

export async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

const SVG = "http://www.w3.org/2000/svg";
export function svg(tag, attrs = {}, ...children) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) node.setAttribute(k, v);
  for (const c of children.flat()) if (c != null) node.append(c);
  return node;
}

export function money(value, currency) {
  try {
    return new Intl.NumberFormat(undefined, { style: "currency", currency }).format(value);
  } catch {
    return `${value.toFixed(2)} ${currency}`;
  }
}

// Observation times are lot-local seconds since the epoch, so format them as UTC.
export function clock(t, withDate = false) {
  if (t == null) return "never";
  const d = new Date(t * 1000);
  const opts = { hour: "2-digit", minute: "2-digit", timeZone: "UTC" };
  if (withDate) Object.assign(opts, { weekday: "short", day: "numeric", month: "short" });
  return new Intl.DateTimeFormat(undefined, opts).format(d);
}

export function liveStream(onLot, indicator) {
  let es;
  const connect = () => {
    es = new EventSource("/api/stream");
    es.addEventListener("open", () => indicator?.classList.add("on"));
    es.addEventListener("error", () => indicator?.classList.remove("on"));
    es.addEventListener("lot", (e) => onLot(JSON.parse(e.data)));
  };
  connect();
  return () => es?.close();
}

// Draw bays as polygons in the camera's own pixel space, so the map matches what the
// camera sees.
export function bayMap(geometry, status) {
  const [w, h] = geometry.frame_size;
  const byId = new Map(status.bays.map((b) => [b.id, b]));
  const root = svg("svg", {
    class: "baymap",
    viewBox: `0 0 ${w} ${h}`,
    role: "img",
    "aria-label": `${status.free} of ${status.capacity} spaces free`,
  });
  for (const bay of geometry.bays) {
    const s = byId.get(bay.id);
    const cls = s?.occupied == null ? "unknown" : s.occupied ? "busy" : "free";
    const poly = svg("polygon", {
      class: `bay ${cls}`,
      "data-bay": bay.id,
      points: bay.polygon.map((p) => p.join(",")).join(" "),
    });
    poly.append(svg("title", {}, document.createTextNode(`Bay ${bay.id}: ${cls}`)));
    root.append(poly);
  }
  return root;
}

export function updateBayMap(root, status) {
  for (const b of status.bays ?? []) {
    const poly = root.querySelector(`[data-bay="${CSS.escape(b.id)}"]`);
    if (!poly) continue;
    const cls = b.occupied == null ? "unknown" : b.occupied ? "busy" : "free";
    poly.setAttribute("class", `bay ${cls}`);
    poly.querySelector("title").textContent = `Bay ${b.id}: ${cls}`;
  }
  root.setAttribute("aria-label", `${status.free} of ${status.capacity} spaces free`);
}
