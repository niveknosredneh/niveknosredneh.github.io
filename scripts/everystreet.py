#!/usr/bin/env python3
"""#everystreet — track cycled street coverage & plan routes over missing streets.

coverage : download OSM street grid, compare against GPX ride history,
           emit interactive map + GeoPackage + stats.
plan     : Chinese Postman routes covering the missing streets, exported
           as GPX files for OpenTracks plus a routes layer on the map.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import folium
import gpxpy
import gpxpy.gpx
import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
from shapely import LineString, union_all

HOME = Path.home()
DEFAULT_PLACE = "Regina, Saskatchewan, Canada"
DEFAULT_GPX_DIR = HOME / "11blog/eleventy-garden/cycling_page/GPX_OUT"
WORKDIR = HOME / "everystreet"

# OSM highway tags that count as non-vehicular "trails" (foot/cycle paths,
# unpaved tracks, bridleways). Used to tint & filter streets vs trails.
TRAIL_HIGHWAYS = {"footway", "path", "track", "cycleway", "bridleway", "steps"}


def is_trail(highway):
    tags = highway if isinstance(highway, list) else [highway]
    return bool(tags) and any(t in TRAIL_HIGHWAYS for t in tags)


def blog_note_dir(blog_dir):
    return blog_dir / "everystreet"


def cache_dir(args_cache):
    return args_cache or WORKDIR / "cache"

ox.settings.use_cache = True


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_rides(gpx_dir: Path) -> gpd.GeoDataFrame:
    rows = []
    files = sorted(gpx_dir.glob("*.gpx"))
    if not files:
        sys.exit(f"No GPX files found in {gpx_dir}")
    for f in files:
        try:
            gpx = gpxpy.parse(open(f, encoding="utf-8", errors="ignore"))
        except Exception as e:
            log(f"  skipping {f.name}: {e}")
            continue
        pts = []
        for trk in gpx.tracks:
            for seg in trk.segments:
                pts += [(p.longitude, p.latitude) for p in seg.points]
        if len(pts) < 2:
            continue
        line = LineString(pts)
        if not line.is_valid:
            continue
        rows.append({"name": f.stem, "geometry": line, "points": len(pts)})
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    log(f"Loaded {len(gdf)} rides ({gdf['points'].sum():,} trackpoints)")
    return gdf[["name", "geometry"]]


def get_street_graph(place: str, network_type: str, cache: Path,
                     custom_filter: str = None) -> nx.MultiDiGraph:
    cache.mkdir(parents=True, exist_ok=True)
    slug = place.lower().replace(",", "").replace(" ", "-")[:40]
    tag = "custom" if custom_filter else network_type
    fp = cache / f"{slug}-{tag}.graphml"
    if fp.exists():
        log(f"Using cached graph {fp.name}")
        return ox.load_graphml(fp)
    log(f"Downloading {tag} network for '{place}' from OSM...")
    if custom_filter:
        G = ox.graph_from_place(place, custom_filter=custom_filter,
                                simplify=True)
    else:
        G = ox.graph_from_place(place, network_type=network_type,
                                simplify=True)
    ox.save_graphml(G, fp)
    log(f"Graph: {len(G.nodes):,} nodes / {G.number_of_edges():,} edges (cached)")
    return G


def compute_coverage(G, rides_wgs, buffer_m, threshold):
    edges_u = ox.graph_to_gdfs(G, nodes=False, fill_edge_geometry=True).reset_index()
    crs = ox.projection.project_gdf(edges_u.head(1)).crs
    edges_m = edges_u.to_crs(crs)
    del edges_u
    rides_m = rides_wgs.to_crs(crs)
    rs = rides_m.copy()
    rs["geometry"] = [g.simplify(1.0) for g in rides_m.geometry]
    log("Buffering tracks...")
    covered = union_all([g.buffer(buffer_m) for g in rs.geometry])
    log("Overlaying coverage against street edges...")
    idx = edges_m.sindex.query(covered, predicate="intersects")
    fracs = np.zeros(len(edges_m))
    geoms = edges_m.geometry.values
    for i in idx:
        L = geoms[i].length
        if L > 0:
            fracs[i] = min(covered.intersection(geoms[i]).length / L, 1.0)
    edges_m["covered_frac"] = fracs
    edges_m["ridden"] = fracs >= threshold
    return edges_m


def stats_dict(edges_m, rides, place):
    total = edges_m["length"].sum() / 1000
    ridden = edges_m.loc[edges_m.ridden, "length"].sum() / 1000
    return {
        "place": place,
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "rides": int(len(rides)),
        "street_segments": int(len(edges_m)),
        "total_km": round(total, 1),
        "ridden_km": round(ridden, 1),
        "missing_km": round(total - ridden, 1),
        "pct_complete": round(100 * ridden / total, 2) if total else 0,
    }


def _fc(gdf, props=(), style=None):
    feats = []
    names = list(props)
    for row in gdf.itertuples():
        geom = row.geometry
        p = {k: getattr(row, k) for k in names}
        if "name" in p and (p["name"] is None or p["name"] != p["name"]):
            p["name"] = "(unnamed)"
        if "covered_frac" in p:
            p["coverage"] = f"{100 * (p.pop('covered_frac') or 0):.0f}%"
        if style:
            p["style"] = dict(style)
        for part in getattr(geom, "geoms", [geom]):
            coords = [[round(x, 5), round(y, 5)] for x, y in part.coords]
            feats.append({"type": "Feature", "properties": p,
                          "geometry": {"type": "LineString",
                                       "coordinates": coords}})
    return {"type": "FeatureCollection", "features": feats}


def make_map(edges_m, rides_wgs, stats, routes_wgs=None, out_fp=None):
    m = folium.Map(tiles=None, control_scale=True, prefer_canvas=True)
    base_tiles = [
        ("OSM",
         "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
         "&copy; OpenStreetMap contributors",
         {}),
        ("CyclOSM",
         "https://a.tile-cyclosm.openstreetmap.fr/cyclosm/{z}/{x}/{y}.png",
         '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
         ' contributors, rendered by <a href="https://www.cyclosm.org">CyclOSM</a>',
         {"max_zoom": 19}),
        ("Satellite",
         "https://server.arcgisonline.com/ArcGIS/rest/services/"
         "World_Imagery/MapServer/tile/{z}/{y}/{x}",
         "Tiles &copy; Esri",
         {"max_zoom": 19}),
    ]
    for name, tiles, attr, kw in base_tiles:
        folium.TileLayer(tiles=tiles, name=name, attr=attr, **kw).add_to(m)
    street_props = ("name", "highway", "covered_frac")
    tip_style = ("background:#fff;border:1px solid #ccc;border-radius:6px;"
                 "box-shadow:2px 2px 6px #0003;padding:4px 8px;"
                 "font-size:12px;color:#111;")
    layers = [
        ("Missing streets", edges_m[~edges_m.ridden], "#e8442a",
         {"weight": 2.5, "opacity": 0.95}, False, street_props),
        ("Ridden streets", edges_m[edges_m.ridden], "#0f9d58",
         {"weight": 2.5, "opacity": 0.9}, True, street_props),
        ("My GPX tracks", rides_wgs, "#7c3aed",
         {"weight": 1.5, "opacity": 0.65}, False, ("name",)),
    ]
    wgs = []
    for name, gdf, color, style, show, props in layers:
        fg = folium.FeatureGroup(name, show=show)
        gdf = gdf.to_crs("EPSG:4326")
        wgs.append(gdf)
        gj = folium.GeoJson(
            _fc(gdf, props, {**style, "color": color}),
            smooth_factor=1.5,
        )
        folium.GeoJsonTooltip(
            fields=list(props[:1]) + (["coverage"] if "covered_frac" in
                                      gdf.columns else []),
            aliases=(["street", "ridden"] if props == street_props
                     else ["ride"]),
            localize=True, sticky=False, style=tip_style,
        ).add_to(gj)
        gj.add_to(fg)
        fg.add_to(m)
    if routes_wgs is not None and len(routes_wgs):
        fg = folium.FeatureGroup("Planned loops", show=True)
        rw = routes_wgs.to_crs("EPSG:4326")
        wgs.append(rw)
        gj = folium.GeoJson(
            _fc(rw, ("name", "chunk_km"),
                {"color": "#2563eb", "weight": 3.5, "dashArray": "7 9",
                 "opacity": 0.95}),
            smooth_factor=1.5,
        )
        folium.GeoJsonTooltip(
            fields=["name", "chunk_km"], aliases=["loop", "km"],
            localize=True, sticky=False, style=tip_style,
        ).add_to(gj)
        gj.add_to(fg)
        fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(folium.Element(
        "<style>.leaflet-control-layers,"
        ".leaflet-control-layers-expanded{"
        "background:rgba(255,255,255,0.75);border-radius:8px}</style>"))

    legend_items = "".join(
        '<div style="display:flex;align-items:center;margin:4px 0">'
        f'<span style="display:inline-block;width:28px;height:0;'
        f'border-top:{st.get("weight", 2) + 1:.0f}px '
        f'{"dashed" if "dashArray" in st else "solid"} {c};'
        f'margin-right:8px"></span>{n}</div>'
        for n, _, c, st, _, _ in layers)
    legend = f"""<div style="position:fixed;bottom:18px;left:12px;z-index:9999;
        background:rgba(255,255,255,0.75);border-radius:10px;box-shadow:0 2px 10px #0002;
        padding:10px 14px;font-family:-apple-system,sans-serif;
        font-size:13px;color:#111">
        <div style="font-weight:700;margin-bottom:6px">#
        everystreet &mdash; Regina</div>
        {legend_items}
        <div style="margin-top:7px;color:#555">
        {stats['ridden_km']:,.0f} of {stats['total_km']:,.0f} km ridden
        ({stats['pct_complete']}%)</div>
        </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    import numpy as np
    tb = np.array([g.total_bounds for g in wgs])
    m.fit_bounds([[tb[:, 1].min(), tb[:, 0].min()],
                  [tb[:, 3].max(), tb[:, 2].max()]])
    if out_fp:
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        m.save(out_fp)
        log(f"Map saved: {out_fp.name} ({out_fp.stat().st_size/1e6:.1f} MB)")


def export_geojson(edges_m, routes_wgs, bn_dir):
    def clean_name(v):
        # GeoPandas yields NaN floats for unnamed OSM ways; JSON has no NaN,
        # so normalise to a string or the browser fails to parse the file.
        if v is None or v == "" or (isinstance(v, float) and v != v):
            return "(unnamed)"
        return v

    data_dir = bn_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    miss = edges_m[~edges_m.ridden].to_crs("EPSG:4326")
    ride = edges_m[edges_m.ridden].to_crs("EPSG:4326")
    miss_feats = []
    for row in miss.itertuples():
        coords = [[round(x, 5), round(y, 5)] for x, y in row.geometry.coords]
        miss_feats.append({
            "type": "Feature",
            "properties": {"name": clean_name(row.name), "highway": row.highway,
                           "trail": is_trail(row.highway)},
            "geometry": {"type": "LineString", "coordinates": coords},
        })
    ride_feats = []
    for row in ride.itertuples():
        coords = [[round(x, 5), round(y, 5)] for x, y in row.geometry.coords]
        ride_feats.append({
            "type": "Feature",
            "properties": {"name": clean_name(row.name), "highway": row.highway,
                           "trail": is_trail(row.highway)},
            "geometry": {"type": "LineString", "coordinates": coords},
        })
    for name, feats in [("missing-streets", miss_feats), ("ridden-streets", ride_feats)]:
        fc = {"type": "FeatureCollection", "features": feats}
        fp = data_dir / f"{name}.geojson"
        fp.write_text(json.dumps(fc))
        log(f"Wrote {fp.name}: {len(feats):,} features ({fp.stat().st_size / 1e6:.1f} MB)")
    if routes_wgs is not None and len(routes_wgs):
        rw = routes_wgs.to_crs("EPSG:4326")
        r_feats = []
        for row in rw.itertuples():
            coords = [[round(x, 5), round(y, 5)] for x, y in row.geometry.coords]
            r_feats.append({
                "type": "Feature",
                "properties": {"name": clean_name(row.name), "chunk_km": row.chunk_km},
                "geometry": {"type": "LineString", "coordinates": coords},
            })
        fp = data_dir / "planned-routes.geojson"
        fp.write_text(json.dumps({"type": "FeatureCollection", "features": r_feats}))
        log(f"Wrote {fp.name}: {len(r_feats)} features")


def export_rides_thumb(edges_m, bn_dir, simplify_m=120.0):
    """Compact silhouette of the ridden network for the #everystreet header
    icon. Merges every ridden edge into one MultiLineString (simplified to
    ~metre tolerance) so the build-time thumbnail stays small.
    """
    data_dir = bn_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    ridden = edges_m[edges_m.ridden]
    if ridden.empty:
        return
    log("Merging ridden edges into a thumbnail overlay...")
    crs = edges_m.crs
    merged = union_all([g.simplify(simplify_m) for g in ridden.geometry])
    merged_wgs = (gpd.GeoDataFrame(
        geometry=list(getattr(merged, "geoms", [merged])), crs=crs)
        .to_crs("EPSG:4326"))
    lines = [[[round(x, 5), round(y, 5)] for x, y in g.coords]
             for g in merged_wgs.geometry]
    fp = data_dir / "rides-thumb.geojson"
    fp.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "MultiLineString", "coordinates": lines},
        }],
    }, separators=(",", ":")))
    log(f"Wrote {fp.name}: {len(lines):,} line parts "
        f"({fp.stat().st_size/1e6:.1f} MB)")


def export_gpkg(edges_m, rides_wgs, fp):
    fp.parent.mkdir(parents=True, exist_ok=True)
    if fp.exists():
        fp.unlink()
    cols = ["u", "v", "key", "name", "highway", "length",
            "covered_frac", "ridden", "geometry"]
    edges_m[[c for c in cols if c in edges_m.columns]].to_file(fp, layer="edges")
    rides_wgs.to_file(fp, layer="rides")
    log(f"GeoPackage saved: {fp}")


def split_chunk(G, edge_ids, pos, max_km, depth=0):
    def km_of(eids):
        return sum(G.edges[e]["length"] for e in eids) / 1000

    if km_of(edge_ids) <= max_km or len(edge_ids) < 16 or depth > 14:
        return [edge_ids]
    mids = np.array([
        ((pos[e[0]][0] + pos[e[1]][0]) / 2, (pos[e[0]][1] + pos[e[1]][1]) / 2)
        for e in edge_ids
    ])
    spread = mids.max(axis=0) - mids.min(axis=0)
    axis = 0 if spread[0] >= spread[1] else 1
    med = np.median(mids[:, axis])
    left = [e for e, mv in zip(edge_ids, mids[:, axis]) if mv <= med]
    right = [e for e, mv in zip(edge_ids, mids[:, axis]) if mv > med]
    if not left or not right:
        return [edge_ids]
    return (split_chunk(G, left, pos, max_km, depth + 1) +
            split_chunk(G, right, pos, max_km, depth + 1))


def _geom_between(a, b, geom_idx, pos_proj, cache={}):
    key = (a, b)
    if key in cache:
        return cache[key]
    pa, pb = pos_proj[a], pos_proj[b]
    cands = geom_idx.get((a, b), []) + geom_idx.get((b, a), [])
    pts = None
    for g in cands:
        c = list(g.coords)
        if abs(c[0][0] - pa[0]) < 1.0 and abs(c[0][1] - pa[1]) < 1.0:
            pts = c
            break
        if abs(c[-1][0] - pa[0]) < 1.0 and abs(c[-1][1] - pa[1]) < 1.0:
            pts = c[::-1]
            break
    if pts is None or len(pts) < 2:
        pts = [pa, pb]
    cache[key] = pts
    return pts


def _edge_between(G_dir, a, b):
    for u, v in ((a, b), (b, a)):
        if G_dir.has_edge(u, v):
            best = min(G_dir[u][v],
                       key=lambda kk: G_dir[u][v][kk].get("length",
                                                          float("inf")))
            return u, v, G_dir[u][v][best].get("length", 0)
    return None


def _add_path_edges(G_dir, H, seq):
    for n1, n2 in zip(seq[:-1], seq[1:]):
        e = _edge_between(G_dir, n1, n2)
        if e:
            H.add_edge(e[0], e[1], length=e[2])


def _connect_components(G_dir, H, comps):
    first = comps[0]
    Gu = None
    for comp in comps[1:]:
        cset = set(comp)
        src = next(iter(first))
        d, p = nx.single_source_dijkstra(G_dir, src, weight="length")
        reach = [n for n in comp if n in d]
        if not reach:
            if Gu is None:
                Gu = ox.convert.to_undirected(G_dir)
            d, p = nx.single_source_dijkstra(Gu, src, weight="length")
            reach = [n for n in comp if n in d]
        if not reach:
            continue
        tgt = min(reach, key=lambda n: d[n])
        seq = p[tgt]
        _add_path_edges(G_dir, H, seq)
        first = first | cset | set(seq)


def chinese_postman(G_dir, edge_ids, geom_idx, pos_proj):
    H = nx.MultiGraph()
    for u, v, k in edge_ids:
        if G_dir.has_edge(u, v):
            H.add_edge(u, v, length=G_dir.edges[(u, v, k)]["length"])
    if not H.number_of_edges():
        return []
    comps = list(nx.connected_components(H))
    if len(comps) > 1:
        _connect_components(G_dir, H, sorted(comps, key=len, reverse=True))

    odd = sorted(n for n, d in H.degree() if d % 2 == 1)
    if odd:
        Gu = ox.convert.to_undirected(G_dir)
        dist, path = {}, {}
        for src in odd:
            d, p = nx.single_source_dijkstra(Gu, src, weight="length")
            for dst in odd:
                if dst != src:
                    dist[(src, dst)] = d.get(dst)
                    path[(src, dst)] = p.get(dst)
        mg = nx.Graph()
        for i, a in enumerate(odd):
            for b in odd[i + 1:]:
                da, db = dist[(a, b)], dist[(b, a)]
                if da is not None and db is not None:
                    mg.add_edge(a, b, weight=-((da + db) / 2))
        for a, b in nx.max_weight_matching(mg, maxcardinality=True):
            seq = path.get((a, b)) or path.get((b, a))
            if not seq:
                continue
            _add_path_edges(G_dir, H, seq)
    if not H.number_of_edges():
        return []
    try:
        start = max(H.nodes, key=lambda n: H.degree(n))
        node_seq = [u for u, _ in nx.eulerian_circuit(H, source=start)]
    except nx.NetworkXError:
        return []
    coords = []
    for a, b in zip(node_seq[:-1], node_seq[1:]):
        seg = _geom_between(a, b, geom_idx, pos_proj)
        if coords and coords[-1] == seg[0]:
            coords.extend(seg[1:])
        else:
            coords.extend(seg)
    return coords


def plan_routes(G, edges_m, max_chunk_km):
    miss = edges_m[~edges_m.ridden]
    if miss.empty:
        log("Nothing missing!")
        return gpd.GeoDataFrame(columns=["name", "chunk_km", "geometry"],
                                crs=edges_m.crs)
    pos = {n: (d["x"], d["y"]) for n, d in G.nodes(data=True)}
    geom_idx = {}
    for row in miss.itertuples():
        if row.geometry is not None:
            geom_idx.setdefault((row.u, row.v), []).append(row.geometry)
    pos_proj = {row.u: (row.geometry.coords[0][0], row.geometry.coords[0][1])
                for row in miss.itertuples() if row.geometry is not None}
    for row in edges_m.itertuples():
        if row.geometry is not None:
            a = (row.geometry.coords[0][0], row.geometry.coords[0][1])
            b = (row.geometry.coords[-1][0], row.geometry.coords[-1][1])
            pos_proj.setdefault(row.u, a)
            pos_proj.setdefault(row.v, b)
    H = nx.MultiGraph()
    for row in miss.itertuples():
        if row.u != row.v and G.has_edge(row.u, row.v, row.key):
            H.add_edge(row.u, row.v, eid=(row.u, row.v, row.key))

    comps = []
    for comp in sorted(nx.connected_components(H), key=len, reverse=True):
        eids = [d["eid"] for _, _, d in H.subgraph(comp).edges(data=True)]
        km = sum(G.edges[e]["length"] for e in eids) / 1000
        cx = np.mean([pos[e[0]][0] + pos[e[1]][0] for e in eids]) / 2
        cy = np.mean([pos[e[0]][1] + pos[e[1]][1] for e in eids]) / 2
        comps.append({"eids": eids, "km": km, "c": (cx, cy)})
    log(f"{len(comps)} disconnected missing-street clusters")

    buckets, cur, cur_km, cur_c = [], [], 0.0, None
    for comp in comps:
        if cur and cur_km + comp["km"] > max_chunk_km:
            buckets.append(cur)
            cur, cur_km, cur_c = [], 0.0, None
        cur.extend(comp["eids"])
        cur_km += comp["km"]
        cur_c = comp["c"] if cur_c is None else cur_c
    if cur:
        buckets.append(cur)
    chunks = []
    for bucket in buckets:
        km = sum(G.edges[e]["length"] for e in bucket) / 1000
        if km > max_chunk_km:
            chunks += split_chunk(G, bucket, pos, max_chunk_km)
        else:
            chunks.append(bucket)

    routes = []
    for ci, chunk in enumerate(chunks, 1):
        km = sum(G.edges[e]["length"] for e in chunk) / 1000
        log(f"Solving route {ci}/{len(chunks)} ({len(chunk)} segs, {km:.1f} km)")
        coords = chinese_postman(G, chunk, geom_idx, pos_proj)
        if len(coords) >= 2:
            routes.append({"name": f"Everystreet {ci:02d}",
                           "chunk_km": round(km, 1),
                           "geometry": LineString(coords)})
    return gpd.GeoDataFrame(routes, crs=edges_m.crs)


def write_route_gpx(name, km, geom_wgs, fp):
    gpx = gpxpy.gpx.GPX()
    gpx.creator = "everystreet-planner"
    rte = gpxpy.gpx.GPXRoute(name=f"{name} ({km} km)")
    gpx.routes.append(rte)
    for lon, lat in geom_wgs.coords:
        rte.points.append(gpxpy.gpx.GPXRoutePoint(latitude=lat, longitude=lon))
    fp.write_text(gpx.to_xml())


def write_blog_page(stats, blog_dir):
    blog_dir.mkdir(parents=True, exist_ok=True)
    md = f"""---
title: everystreet
---
### #everystreet — Regina, SK

| | |
|---|---|
| Streets ridden | **{stats['ridden_km']:,.0f} km** |
| Total streets | {stats['total_km']:,.0f} km |
| Remaining | {stats['missing_km']:,.0f} km |
| Complete | **{stats['pct_complete']}%** |
| Rides logged | {stats['rides']} |

Updated {stats['generated']}.

<iframe src="map.html" style="width:100%;height:640px;border:1px solid #333;
border-radius:8px" loading="lazy"></iframe>

Red = not yet ridden, green = done, blue dashes = planned route.
"""
    (blog_dir / "index.md").write_text(md)
    log(f"Blog note written: {blog_dir/'index.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--place", default=DEFAULT_PLACE)
    ap.add_argument("--gpx-dir", type=Path, default=DEFAULT_GPX_DIR)
    ap.add_argument("--network-type", default="bike",
                    choices=["drive", "bike", "walk", "drive_service",
                             "all_public"])
    ap.add_argument("--include-trails", action="store_true", default=True,
                    help="include footways, tracks, and bridleways (default: on)")
    ap.add_argument("--no-trails", action="store_true",
                    help="exclude trails, only use --network-type")
    ap.add_argument("--buffer", type=float, default=15,
                    help="GPS buffer radius in metres (default 15)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="fraction of edge covered to count as ridden")
    ap.add_argument("--max-chunk-km", type=float, default=60,
                    help="target max length per planned route")
    ap.add_argument("--no-blog", action="store_true")
    ap.add_argument("--blog-dir", type=Path, default=HOME / "11blog/eleventy-garden",
                    help="path to eleventy-garden repo root")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output dir (default: ~/everystreet/output)")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="OSMnx graph cache dir (default: ~/everystreet/cache)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("coverage")
    sub.add_parser("plan")
    args = ap.parse_args()

    out_dir = args.out_dir or WORKDIR / "output"
    cache = args.cache_dir or WORKDIR / "cache"
    bn_dir = args.blog_dir / "everystreet"
    out_dir.mkdir(parents=True, exist_ok=True)

    rides = load_rides(args.gpx_dir)

    custom_filter = None
    if not args.no_trails:
        custom_filter = (
            '["highway"~"motorway|trunk|primary|secondary|tertiary|'
            'unclassified|residential|living_street|service|cycleway|'
            'path|footway|track|bridleway"]'
        )
    G = get_street_graph(args.place, args.network_type, cache, custom_filter)

    edges_m = compute_coverage(G, rides, args.buffer, args.threshold)
    stats = stats_dict(edges_m, rides, args.place)
    log(json.dumps(stats, indent=2))
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    routes_wgs = None
    if args.cmd == "plan":
        routes_m = plan_routes(G, edges_m, args.max_chunk_km)
        if len(routes_m):
            routes_wgs = routes_m.to_crs("EPSG:4326")
            routes_wgs.to_file(out_dir / "planned_routes.geojson", driver="GeoJSON")
            rdir = out_dir / "routes_gpx"
            rdir.mkdir(exist_ok=True)
            for old in rdir.glob("*.gpx"):
                old.unlink()
            for r in routes_wgs.itertuples():
                fp = rdir / f"{r.name.replace(' ', '_').lower()}.gpx"
                write_route_gpx(r.name, r.chunk_km, r.geometry, fp)
            log(f"Wrote {len(routes_wgs)} route GPX files -> {rdir}")

    export_geojson(edges_m, routes_wgs, bn_dir)
    export_rides_thumb(edges_m, bn_dir)
    (bn_dir / "data" / "stats.json").write_text(json.dumps(stats, indent=2))

    make_map(edges_m, rides, stats, routes_wgs, out_fp=bn_dir / "map.html")
    export_gpkg(edges_m, rides, out_dir / "everystreet.gpkg")
    if not args.no_blog:
        write_blog_page(stats, bn_dir)


if __name__ == "__main__":
    main()
