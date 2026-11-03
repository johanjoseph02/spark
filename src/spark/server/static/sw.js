// Cache the app shell so the driver app opens without a connection; data always comes
// from the network.
const SHELL = "spark-shell-v1";
const FILES = ["/", "/static/styles.css", "/static/common.js", "/static/driver.js", "/static/icon.svg"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(SHELL).then((c) => c.addAll(FILES)));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((keys) =>
    Promise.all(keys.filter((k) => k !== SHELL).map((k) => caches.delete(k)))));
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (url.pathname.startsWith("/api/")) return;
  e.respondWith(fetch(e.request).catch(() => caches.match(e.request)));
});
