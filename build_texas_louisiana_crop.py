"""Build a connected Texas–Louisiana ANUGA crop from the full ADCIRC fort.14.

The crop preserves original elements. It does not retriangulate or create KNN
edges. Missing static Manning values are filled conservatively: 0.02 for water
nodes and nearest land-node value for land nodes.
"""

from __future__ import annotations

from array import array
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from convert_harvey_attachments import read_node_dat, spherical_laea


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def extract_crop(fort14: Path, bounds: tuple[float, float, float, float]):
    lon_min, lon_max, lat_min, lat_max = bounds
    with fort14.open("r", encoding="utf-8", errors="replace") as f:
        title = f.readline().strip()
        n_elements, n_nodes = map(int, f.readline().split()[:2])
        global_to_crop = np.full(n_nodes + 1, -1, dtype=np.int32)
        ids = array("i")
        lon = array("d")
        lat = array("d")
        depth = array("d")
        for _ in range(n_nodes):
            parts = f.readline().split()
            node_id = int(parts[0])
            node_lon, node_lat, node_depth = map(float, parts[1:4])
            if lon_min <= node_lon <= lon_max and lat_min <= node_lat <= lat_max:
                global_to_crop[node_id] = len(ids)
                ids.append(node_id); lon.append(node_lon); lat.append(node_lat); depth.append(node_depth)

        triangles_flat = array("i")
        for _ in range(n_elements):
            parts = f.readline().split()
            g = tuple(map(int, parts[-3:]))
            local = tuple(int(global_to_crop[node]) for node in g)
            if min(local) >= 0:
                triangles_flat.extend(local)

        nope = int(f.readline().split()[0])
        neta = int(f.readline().split()[0])
        inherited_open_global: set[tuple[int, int]] = set()
        for _ in range(nope):
            count = int(f.readline().split()[0])
            boundary_ids = [int(f.readline().split()[0]) for _ in range(count)]
            for a, b in zip(boundary_ids[:-1], boundary_ids[1:]):
                if global_to_crop[a] >= 0 and global_to_crop[b] >= 0:
                    inherited_open_global.add(_edge_key(a, b))

        nbou = int(f.readline().split()[0])
        nvel = int(f.readline().split()[0])
        inherited_land_global: set[tuple[int, int]] = set()
        for _ in range(nbou):
            count = int(f.readline().split()[0])
            boundary_ids = [int(f.readline().split()[0]) for _ in range(count)]
            for a, b in zip(boundary_ids[:-1], boundary_ids[1:]):
                if global_to_crop[a] >= 0 and global_to_crop[b] >= 0:
                    inherited_land_global.add(_edge_key(a, b))

    return {
        "title": title,
        "global_counts": {
            "nodes": n_nodes, "elements": n_elements,
            "open_segments": nope, "open_nodes": neta,
            "land_segments": nbou, "land_nodes": nvel,
        },
        "node_id": np.frombuffer(ids, dtype=np.int32).copy().astype(np.int64),
        "lon": np.frombuffer(lon, dtype=np.float64).copy(),
        "lat": np.frombuffer(lat, dtype=np.float64).copy(),
        "adcirc_depth_m": np.frombuffer(depth, dtype=np.float64).copy(),
        "triangles": np.frombuffer(triangles_flat, dtype=np.int32).copy().reshape(-1, 3).astype(np.int64),
        "inherited_open_global": inherited_open_global,
        "inherited_land_global": inherited_land_global,
    }


def retain_largest_connected_component(crop: dict) -> dict:
    tri = crop["triangles"]
    n = len(crop["node_id"])
    used = np.unique(tri)
    old_to_used = np.full(n, -1, dtype=np.int64)
    old_to_used[used] = np.arange(len(used))
    tri_used = old_to_used[tri]
    edges = np.concatenate((tri_used[:, [0, 1]], tri_used[:, [1, 2]], tri_used[:, [2, 0]]))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    adjacency = coo_matrix(
        (np.ones(2 * len(edges), dtype=np.uint8),
         (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
        shape=(len(used), len(used)),
    ).tocsr()
    n_components, labels = connected_components(adjacency, directed=False)
    sizes = np.bincount(labels)
    keep_component = int(np.argmax(sizes))
    keep_used = np.flatnonzero(labels == keep_component)
    used_to_final = np.full(len(used), -1, dtype=np.int64)
    used_to_final[keep_used] = np.arange(len(keep_used))
    triangle_keep = np.all(labels[tri_used] == keep_component, axis=1)
    final_tri = used_to_final[tri_used[triangle_keep]]
    final_old = used[keep_used]
    dropped = {
        "rectangular_nodes": n,
        "unused_after_triangle_clip": int(n - len(used)),
        "components_before_largest_filter": int(n_components),
        "largest_component_nodes": int(len(final_old)),
        "dropped_connected_nodes": int(len(used) - len(final_old)),
        "dropped_triangles": int(len(tri) - len(final_tri)),
    }
    for key in ("node_id", "lon", "lat", "adcirc_depth_m"):
        crop[key] = crop[key][final_old]
    crop["triangles"] = final_tri
    crop["component_audit"] = dropped
    return crop


def assign_manning(crop: dict, source: dict, xy_m: np.ndarray) -> tuple[np.ndarray, dict]:
    source_lookup = {int(node): i for i, node in enumerate(source["node_id"])}
    exact = np.asarray([source_lookup.get(int(node), -1) for node in crop["node_id"]], dtype=np.int64)
    manning = np.empty(len(exact), dtype=np.float64)
    exact_mask = exact >= 0
    manning[exact_mask] = source["manning_n"][exact[exact_mask]]

    missing = ~exact_mask
    water_missing = missing & (crop["adcirc_depth_m"] > 0.0)
    manning[water_missing] = 0.02
    land_missing = missing & ~water_missing

    source_land = source["adcirc_depth_m"] <= 0.0
    lon0 = 0.5 * (source["lon"].min() + source["lon"].max())
    lat0 = 0.5 * (source["lat"].min() + source["lat"].max())
    source_land_xy = spherical_laea(source["lon"][source_land], source["lat"][source_land], lon0, lat0)
    distances, indices = cKDTree(source_land_xy).query(xy_m[land_missing])
    manning[land_missing] = source["manning_n"][source_land][indices]
    report = {
        "exact_source_nodes": int(exact_mask.sum()),
        "water_default_0p02_nodes": int(water_missing.sum()),
        "land_nearest_nodes": int(land_missing.sum()),
        "land_nearest_distance_km": {
            "median": float(np.median(distances) / 1000.0) if len(distances) else 0.0,
            "p95": float(np.percentile(distances, 95) / 1000.0) if len(distances) else 0.0,
            "maximum": float(np.max(distances) / 1000.0) if len(distances) else 0.0,
        },
    }
    return manning, report


def classify_boundaries(crop: dict, bounds, wet_threshold_m: float = 0.5):
    tri = crop["triangles"]
    all_edges = np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]))
    all_edges = np.sort(all_edges, axis=1)
    unique_edges, counts = np.unique(all_edges, axis=0, return_counts=True)
    exterior = unique_edges[counts == 1]
    ids = crop["node_id"]
    global_pairs = np.sort(ids[exterior], axis=1)
    inherited_land = np.asarray([
        tuple(map(int, pair)) in crop["inherited_land_global"] for pair in global_pairs
    ])
    inherited_open = np.asarray([
        tuple(map(int, pair)) in crop["inherited_open_global"] for pair in global_pairs
    ])
    depth = crop["adcirc_depth_m"]
    wet = (depth[exterior[:, 0]] > wet_threshold_m) & (depth[exterior[:, 1]] > wet_threshold_m)

    midpoint_lon = crop["lon"][exterior].mean(axis=1)
    midpoint_lat = crop["lat"][exterior].mean(axis=1)
    lon_min, lon_max, lat_min, lat_max = bounds
    side_distance = np.column_stack((
        (midpoint_lat - lat_min) / max(lat_max - lat_min, 1e-12),
        (midpoint_lon - lon_min) / max(lon_max - lon_min, 1e-12),
        (lon_max - midpoint_lon) / max(lon_max - lon_min, 1e-12),
        (lat_max - midpoint_lat) / max(lat_max - lat_min, 1e-12),
    ))
    # 0 south/offshore, 1 west, 2 east, 3 north/river.
    side = np.argmin(side_distance, axis=1).astype(np.int8)
    artificial_wet = wet & ~inherited_land
    ocean = (artificial_wet & (side != 3)) | inherited_open
    river = artificial_wet & (side == 3) & ~inherited_open
    reflective = ~ocean & ~river
    return {
        "exterior_edges": exterior,
        "open_boundary_edges": exterior[ocean],
        "river_boundary_edges": exterior[river],
        "reflective_boundary_edges": exterior[reflective],
        "exterior_side": side,
        "summary": {
            "exterior_edges": int(len(exterior)),
            "ocean_open_edges": int(ocean.sum()),
            "river_open_edges": int(river.sum()),
            "reflective_edges": int(reflective.sum()),
            "inherited_global_land_edges": int(inherited_land.sum()),
            "inherited_global_open_edges": int(inherited_open.sum()),
            "wet_threshold_m": wet_threshold_m,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fort14", required=True)
    p.add_argument("--source-nodes", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--lon-min", type=float, default=-98.0021427904)
    p.add_argument("--lon-max", type=float, default=-91.960018)
    p.add_argument("--lat-min", type=float, default=25.2402864261)
    p.add_argument("--lat-max", type=float, default=30.799923)
    p.add_argument("--wet-boundary-threshold-m", type=float, default=0.5)
    args = p.parse_args()
    bounds = (args.lon_min, args.lon_max, args.lat_min, args.lat_max)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)

    source = read_node_dat(args.source_nodes)
    crop = retain_largest_connected_component(extract_crop(Path(args.fort14), bounds))
    lon0 = 0.5 * (args.lon_min + args.lon_max)
    lat0 = 0.5 * (args.lat_min + args.lat_max)
    xy = spherical_laea(crop["lon"], crop["lat"], lon0, lat0)
    manning, manning_report = assign_manning(crop, source, xy)
    boundaries = classify_boundaries(crop, bounds, args.wet_boundary_threshold_m)

    mesh_path = output / "texas_louisiana_mesh.npz"
    np.savez_compressed(
        mesh_path,
        node_id=crop["node_id"], lon=crop["lon"], lat=crop["lat"], xy_m=xy,
        triangles=crop["triangles"], adcirc_depth_m=crop["adcirc_depth_m"],
        bed_elevation_m=-crop["adcirc_depth_m"], manning_n=manning,
        open_boundary_edges=boundaries["open_boundary_edges"],
        river_boundary_edges=boundaries["river_boundary_edges"],
        reflective_boundary_edges=boundaries["reflective_boundary_edges"],
        exterior_edges=boundaries["exterior_edges"],
        exterior_side=boundaries["exterior_side"],
        projection=np.asarray("spherical_laea"), projection_lon0_deg=lon0,
        projection_lat0_deg=lat0, source_title=np.asarray(crop["title"]),
    )
    report = {
        "solver_ready_for_ocean_tide_only": True,
        "important_limitation": "river_boundary_edges are stored separately and require hydrographs or a deliberate reflective fallback",
        "bounds_deg": {"lon_min": args.lon_min, "lon_max": args.lon_max,
                       "lat_min": args.lat_min, "lat_max": args.lat_max},
        "nodes": int(len(crop["node_id"])),
        "triangles": int(len(crop["triangles"])),
        "adcirc_depth_range_m": [float(crop["adcirc_depth_m"].min()), float(crop["adcirc_depth_m"].max())],
        "manning_range": [float(manning.min()), float(manning.max())],
        "component_audit": crop["component_audit"],
        "boundary_audit": boundaries["summary"],
        "manning_audit": manning_report,
        "global_fort14": crop["global_counts"],
        "next_required": [
            "ocean tide time series referenced to the bathymetry datum",
            "river hydrographs for river_boundary_edges, or explicit decision to use reflective boundaries",
            "initial state or tidal spin-up",
        ],
    }
    report_path = output / "texas_louisiana_mesh_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"mesh": str(mesh_path), "report": report}, indent=2))


if __name__ == "__main__":
    main()
