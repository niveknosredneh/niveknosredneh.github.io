const fs = require("fs");
const path = require("path");

// 11ty shortcodes for the cycling page (/cycling/).
// Keep these dependency-free so `npm run build:11ty` has no extra installs.

function rideTraceIcon(ride) {
  const pts = ride && ride.trace;
  if (!pts || pts.length < 2) return '<span class="ride-noicon">&mdash;</span>';

  const lats = pts.map((p) => p[0]);
  const lons = pts.map((p) => p[1]);
  const minLat = Math.min(...lats);
  const maxLat = Math.max(...lats);
  const minLon = Math.min(...lons);
  const maxLon = Math.max(...lons);
  const spanLat = Math.max(maxLat - minLat, 1e-6);
  const spanLon = Math.max(maxLon - minLon, 1e-6);
  const scale = 100 / Math.max(spanLat, spanLon);
  const offset = (100 - Math.max(spanLat, spanLon) * scale) / 2;
  const px = pts.map((p) => offset + (p[1] - minLon) * scale);
  const py = pts.map((p) => 100 - offset - (p[0] - minLat) * scale);
  const d = px
    .map((x, i) => (i === 0 ? "M" : "L") + x.toFixed(1) + " " + py[i].toFixed(1))
    .join("");
  const first = { x: px[0].toFixed(1), y: py[0].toFixed(1) };
  const last = { x: px[px.length - 1].toFixed(1), y: py[py.length - 1].toFixed(1) };

  return (
    `<svg class="ride-icon" viewBox="0 0 100 100" aria-hidden="true">` +
    `<path d="${d}" />` +
    `<circle class="ride-start" cx="${first.x}" cy="${first.y}" r="4.5" />` +
    `<circle class="ride-end" cx="${last.x}" cy="${last.y}" r="4.5" />` +
    `</svg>`
  );
}

function formatTime(seconds) {
  if (!seconds || seconds <= 0) return "&mdash;";
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  if (mins === 0) return `${secs}s`;
  return `${mins}min`;
}

function dpDist(p, a, b) {
  const x1 = a[0], y1 = a[1], x2 = b[0], y2 = b[1];
  const dx = x2 - x1, dy = y2 - y1;
  const len2 = dx * dx + dy * dy;
  if (len2 === 0) return Math.hypot(p[0] - x1, p[1] - y1);
  let t = ((p[0] - x1) * dx + (p[1] - y1) * dy) / len2;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(p[0] - (x1 + t * dx), p[1] - (y1 + t * dy));
}

function dpSimplify(pts, tol) {
  if (pts.length < 3) return pts;
  const keep = new Uint8Array(pts.length);
  keep[0] = 1;
  keep[pts.length - 1] = 1;
  const stack = [[0, pts.length - 1]];
  while (stack.length) {
    const [a, b] = stack.pop();
    let dmax = 0, idx = -1;
    for (let i = a + 1; i < b; i++) {
      const d = dpDist(pts[i], pts[a], pts[b]);
      if (d > dmax) { dmax = d; idx = i; }
    }
    if (idx > -1 && dmax > tol) {
      keep[idx] = 1;
      stack.push([a, idx]);
      stack.push([idx, b]);
    }
  }
  const out = [];
  for (let i = 0; i < pts.length; i++) if (keep[i]) out.push(pts[i]);
  return out;
}

function subpath(pts) {
  return pts
    .map(([x, y], i) => (i === 0 ? "M" : "L") + x.toFixed(1) + " " + y.toFixed(1))
    .join("");
}

/* Small single-colour silhouette of Regina's ridden network for the
   #everystreet header. Reads the OSM edges actually ridden (streets AND
   trails) from everystreet/data/rides-thumb.geojson (pre-merged, compact)
   — falling back to the detailed ridden-streets file if that's missing —
   and collapses everything into one shared shape: no per-ride colouring,
   just the coverage. */
function everystreetThumb() {
  const dataDir = path.join(__dirname, "..", "everystreet", "data");
  const lines = loadNetworkLines(dataDir);
  if (!lines.length) return "";

  const PAD = 0.04;
  let minLat = Infinity, maxLat = -Infinity, minLon = Infinity, maxLon = -Infinity;
  lines.forEach((c) =>
    c.forEach(([lon, lat]) => {
      if (lat < minLat) minLat = lat;
      if (lat > maxLat) maxLat = lat;
      if (lon < minLon) minLon = lon;
      if (lon > maxLon) maxLon = lon;
    })
  );
  const latSpan = Math.max(maxLat - minLat, 1e-6);
  const lonSpan = Math.max(maxLon - minLon, 1e-6);
  const lat0 = minLat - latSpan * PAD, lat1 = maxLat + latSpan * PAD;
  const lon0 = minLon - lonSpan * PAD, lon1 = maxLon + lonSpan * PAD;
  const dLat = lat1 - lat0, dLon = lon1 - lon0;
  /* Keep the real-world aspect ratio (lon shrinks with cos(lat)). */
  const cosMid = Math.max(Math.cos(((lat0 + lat1) / 2) * Math.PI / 180), 0.3);
  const W = dLon * cosMid, H = dLat;
  const s = 1000 / Math.max(W, H);
  const vw = W * s, vh = H * s;
  const toXY = (lon, lat) => [
    ((lon - lon0) / dLon) * W * s,
    vh - ((lat - lat0) / dLat) * H * s,
  ];

  const TOL = 1.5; /* viewBox units — collapses collinear trackpoints */
  const d = lines
    .map((c) => detailToSubpath(c, toXY, TOL))
    .join(" ");

  return (
    `<svg class="routes-banner" viewBox="0 0 ${vw.toFixed(1)} ${vh.toFixed(1)}"` +
    ` preserveAspectRatio="xMidYMid meet" role="img"` +
    ` aria-label="Ridden streets and trails across Regina, Saskatchewan">` +
    `<path d="${d}" stroke="#0f9d58" vector-effect="non-scaling-stroke" />` +
    `</svg>`
  );
}

/* Read the network geometry as line-coordinate arrays, preferring the
   pre-merged thumbnail file. Returns [] if nothing is readable. */
function loadNetworkLines(dataDir) {
  for (const name of ["rides-thumb.geojson", "ridden-streets.geojson"]) {
    let fc;
    try {
      fc = JSON.parse(fs.readFileSync(path.join(dataDir, name), "utf8"));
    } catch (e) {
      continue;
    }
    const lines = [];
    for (const f of fc.features || []) {
      const g = f && f.geometry;
      if (!g) continue;
      if (g.type === "MultiLineString") {
        for (const c of g.coordinates) if (c.length >= 2) lines.push(c);
      } else if (g.type === "LineString" && g.coordinates.length >= 2) {
        lines.push(g.coordinates);
      }
    }
    if (lines.length) return lines;
  }
  return [];
}

function detailToSubpath(coords, toXY, tol) {
  const xy = coords.map(([lon, lat]) => {
    const pt = toXY(lon, lat);
    return [Math.round(pt[0] * 10) / 10, Math.round(pt[1] * 10) / 10];
  });
  return subpath(dpSimplify(xy, tol));
}

/* Big "every ride in one web" graphic for the #everystreet header. Each ride
   is its own colour, all normalised to a shared bounding box and drawn as a
   streak with round caps, so the whole city's coverage reads at a glance. */
function rideTable(rides) {
  const cols = [
    ["Date", "date"],
    ["Dist", "dist"],
    ["Elev", "elev"],
    ["Speed", "speed"],
    ["Max", "maxspeed"],
    ["Pace", "pace"],
    ["Route", null],
    ["Time", "time"],
  ];
  const head = cols
    .map(([label, key]) =>
      key
        ? `<th class="sortable" data-sort="${key}"${
            key === "date" ? ' data-default="desc"' : ""
          }>${label}</th>`
        : `<th class="sort-none">${label}</th>`
    )
    .join("");

  if (!rides.length) {
    return (
      `<table class="ride-table" id="ride-table"><thead><tr>${head}</tr></thead>` +
      `<tbody><tr><td colspan="8" class="ride-empty">No rides yet &mdash; drop GPX files into GPX_OUT/.</td></tr></tbody></table>`
    );
  }

  const body = rides
    .map((r, idx) => {
      const date = r.date || "";
      const speed =
        r.avg_speed_kmh != null ? r.avg_speed_kmh.toFixed(1) : "&mdash;";
      const maxSpeed =
        r.max_speed_kmh != null ? r.max_speed_kmh.toFixed(1) : "&mdash;";
      const pace =
        r.pace_min_km != null ? r.pace_min_km.toFixed(1) : "&mdash;";
      return (
        `<tr data-date="${date}" data-year="${date.slice(0, 4)}" data-ride-idx="${idx}">` +
        `<td class="c-date" data-value="${date}">${date}</td>` +
        `<td data-value="${r.distance_km}">${r.distance_km.toFixed(2)}<span class="u">km</span></td>` +
        `<td data-value="${r.elevation_gain_m}">${r.elevation_gain_m.toFixed(1)}<span class="u">m</span></td>` +
        `<td data-value="${r.avg_speed_kmh ?? 0}">${speed}<span class="u">km/h</span></td>` +
        `<td data-value="${r.max_speed_kmh ?? 0}">${maxSpeed}<span class="u">km/h</span></td>` +
        `<td data-value="${r.pace_min_km ?? 0}">${pace}<span class="u">min/km</span></td>` +
        `<td class="c-route">${rideTraceIcon(r)}</td>` +
        `<td data-value="${r.moving_time_s}">${formatTime(r.moving_time_s)}</td>` +
        `</tr>`
      );
    })
    .join("");
  return `<table class="ride-table" id="ride-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function rideSummary(rides) {
  const totals = rides.reduce(
    (acc, r) => ({
      km: acc.km + r.distance_km,
      time: acc.time + r.moving_time_s,
      elev: acc.elev + r.elevation_gain_m,
      rides: acc.rides + 1,
    }),
    { km: 0, time: 0, elev: 0, rides: 0 }
  );
  const cards = [
    ["Distance", "km", totals.km.toFixed(0)],
    ["Rides", "", String(totals.rides)],
    ["Moving time", "hr", (totals.time / 3600).toFixed(1)],
    ["Elevation", "m", String(Math.round(totals.elev))],
  ]
    .map(
      ([label, unit, value]) =>
        `<div class="stat-card"><span class="stat-val">${value}<small>${unit}</small></span><span class="stat-label">${label}</span></div>`
    )
    .join("");

  const byYear = new Map();
  for (const r of rides) {
    const y = (r.date || "").slice(0, 4);
    if (!byYear.has(y)) byYear.set(y, []);
    byYear.get(y).push(r);
  }
  const rows = [...byYear.keys()]
    .sort()
    .reverse()
    .map((y) => {
      const list = byYear.get(y);
      const km = list.reduce((a, r) => a + r.distance_km, 0);
      return `<span class="year-stat"><strong>${y}</strong> ${km.toFixed(0)} km &middot; ${list.length} ride${list.length === 1 ? "" : "s"}</span>`;
    })
    .join("");

  return `<div class="summary-stats">${cards}</div><div class="year-summary">${rows}</div>`;
}

function yearFilter(rides) {
  const years = [
    ...new Set(rides.map((r) => (r.date || "").slice(0, 4))),
  ]
    .filter(Boolean)
    .sort()
    .reverse();
  const buttons = ["Total", ...years]
    .map(
      (y) =>
        `<button type="button" class="year-btn${y === "Total" ? " active" : ""}" data-year="${y}">${y}</button>`
    )
    .join("");
  return `<div class="year-filter" id="year-filter">${buttons}</div>`;
}

function heatmap(rides) {
  const counts = new Map();
  for (const r of rides) {
    counts.set(r.date, (counts.get(r.date) || 0) + 1);
  }

  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const start = new Date(today);
  start.setDate(start.getDate() - 371); // ~53 weeks back
  start.setDate(start.getDate() - start.getDay()); // rewind to Sunday
  const iso = (d) => {
    const m = String(d.getMonth() + 1).padStart(2, "0");
    const day = String(d.getDate()).padStart(2, "0");
    return `${d.getFullYear()}-${m}-${day}`;
  };

  const dayMs = 86400000;
  const weeks = Math.ceil((today - start) / dayMs / 7) + 1;
  const CELL = 10;
  const GAP = 2;
  const STEP = CELL + GAP;
  const width = weeks * STEP - GAP;
  const height = 7 * STEP - GAP;

  const level = (c) => (c >= 8 ? 4 : c >= 4 ? 3 : c >= 2 ? 2 : 1);
  let cells = "";
  for (let i = 0; i < weeks * 7; i++) {
    const day = new Date(+start + i * dayMs);
    const x = Math.floor(i / 7) * STEP;
    const y = (i % 7) * STEP;
    const key = iso(day);
    const future = day > today;
    const c = counts.get(key) || 0;
    const cls = future ? "heat-empty" : c ? `heat-${level(c)}` : "heat-0";
    const title = future ? "" : `${key}: ${c} ride${c === 1 ? "" : "s"}`;
    cells +=
      `<rect x="${x}" y="${y}" width="${CELL}" height="${CELL}" rx="2" class="${cls}"` +
      ` data-date="${future ? "" : key}"><title>${title}</title></rect>`;
  }

  return (
    `<figure class="heatmap-wrap">` +
    `<svg class="heatmap" viewBox="0 0 ${width} ${height}" role="img" aria-label="Ride activity heatmap">${cells}</svg>` +
    `</figure>`
  );
}

function ridesData(rides) {
  return `<script type="application/json" id="rides-data">${JSON.stringify(rides)}</script>`;
}

module.exports = {
  rideTraceIcon,
  everystreetThumb,
  rideTable,
  rideSummary,
  yearFilter,
  heatmap,
  ridesData,
};