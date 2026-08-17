"""Convert HARVEY.xlsx and input1.dat to the ANUGA/HOPNet input conventions.

The DAT attachment contains static ADCIRC-node attributes but no elements or
boundary topology.  With ``--fort14`` this script extracts the missing triangle
and open-boundary records and writes a complete ``mesh.npz``.  Without it, the
script writes a clearly named partial static-node file and never invents KNN or
Delaunay connectivity.
"""

from __future__ import annotations

import argparse
from datetime import datetime, time
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


KNOT_TO_MS = 0.514444
NMI_TO_KM = 1.852
EARTH_RADIUS_M = 6_371_008.8
RHO_AIR = 1.225


def spherical_laea(lon_deg: np.ndarray, lat_deg: np.ndarray,
                   lon0_deg: float, lat0_deg: float) -> np.ndarray:
    """Spherical Lambert azimuthal equal-area coordinates in metres."""
    lon = np.deg2rad(np.asarray(lon_deg, float))
    lat = np.deg2rad(np.asarray(lat_deg, float))
    lon0, lat0 = np.deg2rad([lon0_deg, lat0_deg])
    dlon = lon - lon0
    denom = 1.0 + np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(dlon)
    k = np.sqrt(2.0 / np.maximum(denom, 1e-12))
    x = EARTH_RADIUS_M * k * np.cos(lat) * np.sin(dlon)
    y = EARTH_RADIUS_M * k * (
        np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(dlon)
    )
    return np.column_stack((x, y))


def read_node_dat(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        declared = int(handle.readline().strip())
    values = np.loadtxt(path, skiprows=1)
    if values.shape != (declared, 5):
        raise ValueError(f"{path}: declared {declared} nodes but parsed {values.shape}")
    node_id = values[:, 0].astype(np.int64)
    if len(np.unique(node_id)) != len(node_id):
        raise ValueError("Duplicate node IDs in input1.dat")
    return {
        "node_id": node_id,
        "lon": values[:, 1],
        "lat": values[:, 2],
        # ADCIRC DP is positive downward; ANUGA elevation is positive upward.
        "adcirc_depth_m": values[:, 3],
        "bed_elevation_m": -values[:, 3],
        "manning_n": values[:, 4],
    }


def reconstruct_excel_datetimes(values: Iterable[object]) -> pd.DatetimeIndex:
    """Rebuild dates where Excel retained only a time-of-day value."""
    result: list[datetime] = []
    current_date = None
    for value in values:
        if isinstance(value, pd.Timestamp):
            value = value.to_pydatetime()
        if isinstance(value, datetime):
            current_date = value.date()
            result.append(value)
        elif isinstance(value, time):
            if current_date is None:
                raise ValueError("Time-only track record precedes the first full date")
            result.append(datetime.combine(current_date, value))
        else:
            raise ValueError(f"Unexpected ISO_TIME value: {value!r}")
    index = pd.DatetimeIndex(result)
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("Reconstructed hurricane times are not strictly increasing")
    return index


def parse_quadrant_radii(values: Iterable[object]) -> np.ndarray:
    """Parse IBTrACS-style NE,SE,SW,NW radius strings, preserving missingness."""
    rows = []
    for value in values:
        if pd.isna(value):
            rows.append([np.nan] * 4)
            continue
        parts = [part.strip() for part in str(value).split(",")]
        parts = (parts + [""] * 4)[:4]
        rows.append([float(part) if part else np.nan for part in parts])
    return np.asarray(rows, dtype=float) * NMI_TO_KM


def convert_track(xlsx_path: str | Path, node_data: dict[str, np.ndarray],
                  lon0: float, lat0: float, buffer_deg: float) -> tuple[dict, pd.DataFrame]:
    df = pd.read_excel(xlsx_path, sheet_name="Sheet1", header=0, skiprows=[1])
    dt = reconstruct_excel_datetimes(df["ISO_TIME_________"])
    df = df.assign(iso_time=dt)

    bounds = {
        "lon_min": float(node_data["lon"].min()), "lon_max": float(node_data["lon"].max()),
        "lat_min": float(node_data["lat"].min()), "lat_max": float(node_data["lat"].max()),
    }
    keep = (
        df["USA LON"].between(bounds["lon_min"] - buffer_deg, bounds["lon_max"] + buffer_deg)
        & df["USA LAT"].between(bounds["lat_min"] - buffer_deg, bounds["lat_max"] + buffer_deg)
    )
    local = df.loc[keep].copy().reset_index(drop=True)
    if len(local) < 2:
        raise ValueError("Track/domain filter left fewer than two storm records")

    xy = spherical_laea(local["USA LON"], local["USA LAT"], lon0, lat0)
    time_s = (local["iso_time"] - local["iso_time"].iloc[0]).dt.total_seconds().to_numpy(float)
    vmax_ms = local["USA WIND"].to_numpy(float) * KNOT_TO_MS
    pc = local["USA PRES"].to_numpy(float)
    po = local["USA POCI"].to_numpy(float)
    delta_p_pa = np.maximum(po - pc, 1.0) * 100.0
    # Infer Holland B from the observed 10-m maximum wind. The 0.9 factor is a
    # documented modelling assumption, not an observed workbook field.
    gradient_wind = vmax_ms / 0.9
    holland_b = np.clip(RHO_AIR * np.e * gradient_wind**2 / delta_p_pa, 0.8, 2.5)

    r34 = parse_quadrant_radii(local["USA R34"])
    r50 = parse_quadrant_radii(local["USA R50"])
    r64 = parse_quadrant_radii(local["USA R64"])
    track = {
        "time_s": time_s,
        "iso_time": local["iso_time"].dt.strftime("%Y-%m-%dT%H:%M:%S").to_numpy(),
        "center_x_m": xy[:, 0], "center_y_m": xy[:, 1],
        "center_lon_deg": local["USA LON"].to_numpy(float),
        "center_lat_deg": local["USA LAT"].to_numpy(float),
        "central_pressure_hpa": pc,
        "outer_pressure_hpa": po,
        "rmw_km": local["USA RMW"].to_numpy(float) * NMI_TO_KM,
        "translation_speed_ms": local["STORM SPEED"].to_numpy(float) * KNOT_TO_MS,
        "heading_deg": local["STORM DIR"].to_numpy(float),
        "holland_b": holland_b,
        "max_wind_ms": vmax_ms,
        "roci_km": local["USA ROCI"].to_numpy(float) * NMI_TO_KM,
        "r34_quadrant_km": r34,
        "r50_quadrant_km": r50,
        "r64_quadrant_km": r64,
    }
    scalar_keys = [key for key, value in track.items() if np.asarray(value).ndim == 1]
    table = pd.DataFrame({key: track[key] for key in scalar_keys})
    for radius_name, radius in (("r34", r34), ("r50", r50), ("r64", r64)):
        for q, quadrant in enumerate(("ne", "se", "sw", "nw")):
            table[f"{radius_name}_{quadrant}_km"] = radius[:, q]
    return track, table


def read_fort14_topology(path: str | Path, selected_node_ids: np.ndarray):
    """Read ADCIRC elements and open boundaries, retaining selected nodes."""
    selected = set(map(int, selected_node_ids))
    local_index = {int(node): i for i, node in enumerate(selected_node_ids)}
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        title = f.readline().strip()
        ne, np_total = map(int, f.readline().split()[:2])
        for _ in range(np_total):
            f.readline()
        elements = []
        for _ in range(ne):
            parts = f.readline().split()
            nodes = list(map(int, parts[-3:]))
            if all(n in selected for n in nodes):
                elements.append([local_index[n] for n in nodes])
        nope = int(f.readline().split()[0])
        neta = int(f.readline().split()[0])
        open_edges: list[tuple[int, int]] = []
        selected_open_nodes = 0
        for _ in range(nope):
            n_nodes = int(f.readline().split()[0])
            ids = [int(f.readline().split()[0]) for _ in range(n_nodes)]
            selected_open_nodes += sum(node in selected for node in ids)
            for a, b in zip(ids[:-1], ids[1:]):
                if a in selected and b in selected:
                    open_edges.append((local_index[a], local_index[b]))
        nbou = int(f.readline().split()[0])
        nvel = int(f.readline().split()[0])
        selected_land_nodes = 0
        for _ in range(nbou):
            n_nodes = int(f.readline().split()[0])
            for _ in range(n_nodes):
                selected_land_nodes += int(int(f.readline().split()[0]) in selected)
    if not elements:
        raise ValueError("No fort.14 triangles were fully contained in the selected DAT nodes")
    metadata = {
        "global_nodes": np_total, "global_elements": ne,
        "global_open_segments": nope, "global_open_nodes": neta,
        "selected_open_boundary_nodes": selected_open_nodes,
        "global_land_segments": nbou, "global_land_nodes": nvel,
        "selected_land_boundary_nodes": selected_land_nodes,
    }
    return (title, np.asarray(elements, np.int64),
            np.asarray(open_edges, np.int64).reshape(-1, 2), metadata)


def audit_subset_topology(triangles: np.ndarray, node_count: int) -> dict:
    """Characterize clipping damage without pretending it is solver-ready."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]],
                            triangles[:, [2, 0]]), axis=0)
    edges = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    exterior_edges = unique_edges[counts == 1]
    used_nodes = np.unique(triangles)
    adjacency = coo_matrix(
        (np.ones(2 * len(unique_edges), dtype=np.uint8),
         (np.r_[unique_edges[:, 0], unique_edges[:, 1]],
          np.r_[unique_edges[:, 1], unique_edges[:, 0]])),
        shape=(node_count, node_count),
    ).tocsr()
    _, labels = connected_components(adjacency, directed=False)
    sizes = np.bincount(labels)
    nontrivial = np.sort(sizes[sizes > 1])[::-1]
    return {
        "used_nodes": used_nodes,
        "exterior_edges": exterior_edges,
        "component_labels": labels,
        "component_sizes": sizes,
        "summary": {
            "used_nodes": int(len(used_nodes)),
            "unused_nodes": int(node_count - len(used_nodes)),
            "unique_edges": int(len(unique_edges)),
            "exterior_edges": int(len(exterior_edges)),
            "nontrivial_components": int(len(nontrivial)),
            "largest_component_nodes": int(nontrivial[0]) if len(nontrivial) else 0,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--xlsx", required=True)
    p.add_argument("--nodes", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--fort14", help="Original ADCIRC fort.14; required for complete mesh.npz")
    p.add_argument("--track-buffer-deg", type=float, default=8.0)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    nodes = read_node_dat(args.nodes)
    lon0 = 0.5 * (float(nodes["lon"].min()) + float(nodes["lon"].max()))
    lat0 = 0.5 * (float(nodes["lat"].min()) + float(nodes["lat"].max()))
    xy = spherical_laea(nodes["lon"], nodes["lat"], lon0, lat0)
    track, track_table = convert_track(args.xlsx, nodes, lon0, lat0, args.track_buffer_deg)

    static_path = out / "harvey_nodes_static.npz"
    np.savez_compressed(
        static_path, node_id=nodes["node_id"], lon=nodes["lon"], lat=nodes["lat"],
        xy_m=xy, adcirc_depth_m=nodes["adcirc_depth_m"],
        bed_elevation_m=nodes["bed_elevation_m"], manning_n=nodes["manning_n"],
        projection=np.asarray("spherical_laea"), projection_lon0_deg=lon0,
        projection_lat0_deg=lat0,
    )
    track_path = out / "harvey_track_gulf.npz"
    np.savez_compressed(track_path, **track)
    track_table.to_csv(out / "harvey_track_gulf.csv", index=False)

    complete_mesh = False
    mesh_path = None
    topology_path = None
    topology = {"triangles": 0, "open_boundary_edges": 0}
    fort14_metadata = None
    missing_required = [
        "triangle connectivity (prefer the original ADCIRC fort.14)",
        "ADCIRC open-boundary node sequences/edges from fort.14",
    ]
    if args.fort14:
        title, triangles, open_edges, fort14_metadata = read_fort14_topology(
            args.fort14, nodes["node_id"])
        audit = audit_subset_topology(triangles, len(nodes["node_id"]))
        topology_path = out / "mesh_topology_subset.npz"
        np.savez_compressed(
            topology_path, node_id=nodes["node_id"], lon=nodes["lon"], lat=nodes["lat"],
            xy_m=xy, triangles=triangles, open_boundary_edges=open_edges,
            exterior_edges=audit["exterior_edges"],
            component_labels=audit["component_labels"],
            component_sizes=audit["component_sizes"],
            adcirc_depth_m=nodes["adcirc_depth_m"],
            bed_elevation_m=nodes["bed_elevation_m"], manning_n=nodes["manning_n"],
            projection=np.asarray("spherical_laea"), projection_lon0_deg=lon0,
            projection_lat0_deg=lat0, source_title=np.asarray(title),
        )
        solver_ready = (
            audit["summary"]["nontrivial_components"] == 1
            and audit["summary"]["unused_nodes"] == 0
            and len(open_edges) > 0
        )
        topology = {
            "triangles": int(len(triangles)),
            "open_boundary_edges": int(len(open_edges)),
            **audit["summary"],
        }
        complete_mesh = solver_ready
        if solver_ready:
            mesh_path = out / "mesh.npz"
            # Identical bytes, but this name is reserved for solver-ready input.
            mesh_path.write_bytes(topology_path.read_bytes())
            missing_required = []
        else:
            missing_required = [
                "a connected regional mesh or per-component simulation plan",
                "boundary time series for the artificial exterior edges created by clipping",
                "an ocean boundary: selected nodes do not intersect the global fort.14 open boundary",
            ]

    report = {
        "node_count": int(len(nodes["node_id"])),
        "node_id_min": int(nodes["node_id"].min()),
        "node_id_max": int(nodes["node_id"].max()),
        "longitude_range_deg": [float(nodes["lon"].min()), float(nodes["lon"].max())],
        "latitude_range_deg": [float(nodes["lat"].min()), float(nodes["lat"].max())],
        "adcirc_depth_range_m": [float(nodes["adcirc_depth_m"].min()), float(nodes["adcirc_depth_m"].max())],
        "manning_range": [float(nodes["manning_n"].min()), float(nodes["manning_n"].max())],
        "track_records_total": 141,
        "track_records_selected": int(len(track["time_s"])),
        "track_start": str(track["iso_time"][0]), "track_end": str(track["iso_time"][-1]),
        "projection": {"name": "spherical_laea", "lon0_deg": lon0, "lat0_deg": lat0},
        "complete_mesh": complete_mesh,
        "topology": topology,
        "fort14": fort14_metadata,
        "topology_subset_path": str(topology_path) if topology_path else None,
        "missing_required_data": missing_required,
        "additional_recommended_data": [
            "tidal boundary elevations or harmonic constituents with vertical datum",
            "initial water-surface elevation and velocity",
            "optional SWAN radiation stresses Sxx/Sxy/Syy for wave coupling",
            "a small paired ADCIRC output set for multi-fidelity calibration",
        ],
    }
    (out / "conversion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "static_nodes": str(static_path), "track": str(track_path),
        "mesh": str(mesh_path) if mesh_path else None,
        "topology_subset": str(topology_path) if topology_path else None,
        "report": report,
    }, indent=2))


if __name__ == "__main__":
    main()
