"""Convert downloaded Harvey rainfall and streamflow to ANUGA mesh forcing."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree


def parse_utc(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def track_start_utc(path: str | Path) -> datetime:
    """Read the first track timestamp, including trusted legacy object NPZs.

    New track files should store ``iso_time`` as fixed-width Unicode. Older
    files produced by pandas may contain a NumPy object array, which requires
    pickle support. Only that legacy fallback enables pickle and the extracted
    value is immediately restricted to a scalar string/bytes/datetime value.
    """
    try:
        with np.load(path, allow_pickle=False) as track:
            raw = np.asarray(track["iso_time"]).reshape(-1)
    except ValueError as exc:
        if "Object arrays cannot be loaded" not in str(exc):
            raise
        with np.load(path, allow_pickle=True) as track:
            raw = np.asarray(track["iso_time"]).reshape(-1)
    if not len(raw):
        raise ValueError("track iso_time is empty")
    value = raw[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    elif isinstance(value, np.datetime64):
        value = np.datetime_as_string(value, unit="s")
    elif not isinstance(value, (str, datetime)):
        raise TypeError(
            f"unsupported legacy iso_time scalar type: {type(value).__name__}")
    return parse_utc(str(value))


def spherical_xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lon_r, lat_r = np.deg2rad(lon), np.deg2rad(lat)
    c = np.cos(lat_r)
    return np.column_stack((c * np.cos(lon_r), c * np.sin(lon_r), np.sin(lat_r)))


def stageiv_files(root: Path) -> list[Path]:
    """Find likely GRIB inputs without assuming a particular archive name."""
    candidates = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name.startswith("."):
            continue
        lower = path.name.lower()
        likely_name = (
            any(token in lower for token in ("st4", "stage4", "stageiv", "precip"))
            or path.suffix.lower() in {".grb", ".grib", ".grb2", ".gz", ".bin"}
        )
        if likely_name:
            candidates.append(path)
    return sorted(candidates)


def open_grib_messages(path: Path):
    """Open ordinary or gzip-compressed GRIB data with pygrib."""
    import pygrib

    temporary_name = None
    try:
        if path.suffix.lower() == ".gz":
            with gzip.open(path, "rb") as source:
                with tempfile.NamedTemporaryFile(suffix=".grib", delete=False) as target:
                    shutil.copyfileobj(source, target)
                    temporary_name = target.name
            grib_path = temporary_name
        else:
            grib_path = str(path)
        with pygrib.open(grib_path) as grbs:
            return list(grbs)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def accumulation_seconds(message) -> float:
    """Return the accumulation duration represented by one GRIB message."""
    start_step = getattr(message, "startStep", None)
    end_step = getattr(message, "endStep", None)
    if start_step is not None and end_step is not None:
        try:
            hours = float(end_step) - float(start_step)
            if hours > 0.0:
                return hours * 3600.0
        except (TypeError, ValueError):
            pass
    step_range = str(getattr(message, "stepRange", ""))
    match = re.fullmatch(r"\s*([0-9.]+)-([0-9.]+)\s*", step_range)
    if match:
        hours = float(match.group(2)) - float(match.group(1))
        if hours > 0.0:
            return hours * 3600.0
    length = getattr(message, "lengthOfTimeRange", None)
    unit = getattr(message, "indicatorOfUnitForTimeRange", 1)
    unit_seconds = {0: 60.0, 1: 3600.0, 2: 86400.0, 10: 10800.0,
                    11: 21600.0, 12: 43200.0, 13: 1.0}
    if length is not None and int(unit) in unit_seconds and float(length) > 0.0:
        return float(length) * unit_seconds[int(unit)]
    # Historical ST4 filenames normally represent one-hour accumulations.
    if re.search(r"(?:01h|\.01\b)", str(getattr(message, "name", "")), re.I):
        return 3600.0
    return 3600.0


def read_stageiv_on_nodes(root: Path, node_lon: np.ndarray,
                          node_lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        import pygrib
    except ImportError as exc:
        raise RuntimeError(
            "Hourly Stage IV conversion requires pygrib and ecCodes: "
            "conda install -c conda-forge pygrib eccodes") from exc

    node_xyz = spherical_xyz(node_lon, node_lat)
    values, times, durations = [], [], []
    cached_shape = None
    neighbour_index = neighbour_weight = None
    discovered = stageiv_files(root)
    rejected = []
    for path in discovered:
        try:
            messages = open_grib_messages(path)
        except Exception as exc:
            rejected.append(f"{path.name}: {type(exc).__name__}")
            continue
        message = next((m for m in messages if
                        "precip" in str(getattr(m, "name", "")).lower()), None)
        if message is None:
            continue
        grid = np.asarray(message.values, dtype=np.float64)
        lat, lon = message.latlons()
        valid = np.isfinite(grid) & (grid >= 0.0)
        if not np.any(valid):
            continue
        if cached_shape != grid.shape:
            source_xyz = spherical_xyz(lon[valid].ravel(), lat[valid].ravel())
            distance, neighbour_index = cKDTree(source_xyz).query(
                node_xyz, k=min(4, len(source_xyz)))
            distance = np.atleast_2d(distance)
            if distance.shape[0] != len(node_xyz):
                distance = distance.T
                neighbour_index = np.asarray(neighbour_index).T
            neighbour_weight = 1.0 / np.maximum(distance, 1.0e-12)
            neighbour_weight /= neighbour_weight.sum(axis=1, keepdims=True)
            cached_shape = grid.shape
        source_value = grid[valid].ravel()
        node_mm = np.sum(source_value[neighbour_index] * neighbour_weight, axis=1)
        # kg m^-2 is numerically equivalent to mm of liquid precipitation.
        duration_s = accumulation_seconds(message)
        node_rate = np.maximum(node_mm, 0.0) / 1000.0 / duration_s
        values.append(node_rate.astype(np.float32))
        durations.append(duration_s)
        valid_time = getattr(message, "validDate", None)
        if valid_time is None:
            match = re.search(r"(20\d{8})", path.name)
            if not match:
                continue
            valid_time = datetime.strptime(match.group(1), "%Y%m%d%H")
        if valid_time.tzinfo is None:
            valid_time = valid_time.replace(tzinfo=timezone.utc)
        times.append(valid_time.astimezone(timezone.utc).timestamp())
    if not values:
        examples = "; ".join(rejected[:8]) if rejected else "none discovered"
        raise ValueError(
            f"No readable Stage IV GRIB precipitation below {root}. "
            f"Candidate files={len(discovered)}. Examples: {examples}")
    order = np.argsort(times)
    return (np.asarray(times, dtype=np.float64)[order], np.stack(values)[order],
            np.asarray(durations, dtype=np.float64)[order])


def parse_usgs(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = []
    for series in payload.get("value", {}).get("timeSeries", []):
        variable = series.get("variable", {}).get("variableCode", [{}])[0].get("value")
        if variable != "00060":
            continue
        source = series.get("sourceInfo", {})
        site_id = source.get("siteCode", [{}])[0].get("value", "")
        geo = source.get("geoLocation", {}).get("geogLocation", {})
        raw_values = series.get("values", [{}])[0].get("value", [])
        epoch, discharge = [], []
        for item in raw_values:
            try:
                epoch.append(parse_utc(item["dateTime"]).timestamp())
                discharge.append(float(item["value"]) * 0.028316846592)
            except (KeyError, TypeError, ValueError):
                continue
        if len(epoch) >= 2:
            order = np.argsort(epoch)
            result.append({
                "site_id": site_id,
                "name": source.get("siteName", site_id),
                "lon": float(geo.get("longitude")),
                "lat": float(geo.get("latitude")),
                "epoch": np.asarray(epoch)[order],
                "q_m3s": np.asarray(discharge, dtype=np.float64)[order],
            })
    return result


def interpolate_time(source_epoch: np.ndarray, source_values: np.ndarray,
                     target_epoch: np.ndarray) -> np.ndarray:
    return np.interp(target_epoch, source_epoch, source_values,
                     left=source_values[0], right=source_values[-1])


def edge_to_triangle(triangles: np.ndarray) -> dict[tuple[int, int], int]:
    mapping = {}
    for triangle_index, nodes in enumerate(np.asarray(triangles, dtype=np.int64)):
        for a, b in ((nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0])):
            key = (int(min(a, b)), int(max(a, b)))
            mapping.setdefault(key, triangle_index)
    return mapping


def map_rivers(mesh, gauges: list[dict], target_epoch: np.ndarray,
               approved_sites: set[str], inlet_count: int):
    gauges = [g for g in gauges if not approved_sites or g["site_id"] in approved_sites]
    if not gauges:
        return (np.empty((len(target_epoch), 0), np.float32),
                np.empty((0, inlet_count), np.int64), np.empty(0, dtype="U1"), [])
    if "river_boundary_edges" not in mesh.files:
        raise ValueError("mesh has no river_boundary_edges; rebuild the regional crop")
    edges = np.asarray(mesh["river_boundary_edges"], dtype=np.int64)
    if not len(edges):
        raise ValueError("mesh river_boundary_edges is empty")
    lon, lat = np.asarray(mesh["lon"]), np.asarray(mesh["lat"])
    edge_xyz = spherical_xyz(lon[edges].mean(axis=1), lat[edges].mean(axis=1))
    edge_tree = cKDTree(edge_xyz)
    xy = np.asarray(mesh["xy_m"], dtype=float)
    triangles = np.asarray(mesh["triangles"], dtype=np.int64)
    triangle_centroids = xy[triangles].mean(axis=1)
    triangle_tree = cKDTree(triangle_centroids)
    boundary_lookup = edge_to_triangle(triangles)
    q_columns, inlets, site_ids, audit = [], [], [], []
    for gauge in gauges:
        _, edge_pos = edge_tree.query(spherical_xyz(
            np.asarray([gauge["lon"]]), np.asarray([gauge["lat"]]))[0])
        edge = edges[int(edge_pos)]
        edge_midpoint = xy[edge].mean(axis=0)
        _, near_tri = triangle_tree.query(edge_midpoint, k=min(inlet_count, len(triangles)))
        near_tri = np.atleast_1d(near_tri).astype(np.int64)
        boundary_tri = boundary_lookup.get(tuple(sorted(map(int, edge))))
        if boundary_tri is not None:
            near_tri = np.unique(np.r_[boundary_tri, near_tri])[:inlet_count]
        padded = np.full(inlet_count, -1, dtype=np.int64)
        padded[:len(near_tri)] = near_tri
        inlets.append(padded)
        q_columns.append(interpolate_time(gauge["epoch"], gauge["q_m3s"], target_epoch))
        site_ids.append(gauge["site_id"])
        audit.append({
            "site_id": gauge["site_id"], "name": gauge["name"],
            "gage_lon": gauge["lon"], "gage_lat": gauge["lat"],
            "mapped_edge_nodes": list(map(int, edge)),
            "maximum_discharge_m3s": float(np.max(gauge["q_m3s"])),
        })
    return (np.stack(q_columns, axis=1).astype(np.float32), np.stack(inlets),
            np.asarray(site_ids), audit)


def load_node_parameter(path: str | None, key: str, count: int,
                        default: float) -> np.ndarray:
    if not path:
        return np.full(count, default, dtype=np.float32)
    with np.load(path) as data:
        values = np.asarray(data[key], dtype=np.float32)
    if values.shape != (count,):
        raise ValueError(f"{path}:{key} must have shape [{count}]")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--track", required=True)
    parser.add_argument("--download-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration-hours", type=float, default=282.0)
    parser.add_argument("--interval-minutes", type=float, default=30.0)
    parser.add_argument(
        "--start-iso",
        help="Optional UTC start override, e.g. 2017-08-21T18:00:00Z")
    parser.add_argument("--approved-usgs-sites", default="")
    parser.add_argument("--inlet-triangles", type=int, default=16)
    parser.add_argument("--soil-npz", help="Optional NPZ with ksat_mps [N]")
    parser.add_argument("--landcover-npz", help="Optional NPZ with impervious_fraction [N]")
    parser.add_argument("--default-ksat-mps", type=float, default=2.0e-6)
    parser.add_argument("--default-impervious", type=float, default=0.15)
    parser.add_argument("--allow-missing-rain", action="store_true")
    args = parser.parse_args()

    start = parse_utc(args.start_iso) if args.start_iso else track_start_utc(args.track)
    with np.load(args.mesh) as mesh:
        node_count = len(mesh["xy_m"])
        dt = args.interval_minutes * 60.0
        time_s = np.arange(0.0, args.duration_hours * 3600.0 + 0.5 * dt, dt)
        dense_gib = len(time_s) * node_count * 4 / 1024**3
        if dense_gib >= 4.0:
            print(
                "WARNING: each dense float32 [T,N] field will require "
                f"approximately {dense_gib:.2f} GiB; rainfall plus effective "
                f"rainfall requires at least {2.0 * dense_gib:.2f} GiB.",
                flush=True,
            )
        target_epoch = start.timestamp() + time_s
        time_iso = np.asarray([
            datetime.fromtimestamp(t, tz=timezone.utc).isoformat().replace("+00:00", "Z")
            for t in target_epoch])

        rain_root = Path(args.download_dir) / "stageiv_hourly"
        try:
            rain_epoch, rain_nodes, rain_duration_s = read_stageiv_on_nodes(
                rain_root, np.asarray(mesh["lon"]), np.asarray(mesh["lat"]))
            rain_rate = np.empty((len(time_s), node_count), dtype=np.float32)
            # Avoid a Python loop over ~741k mesh nodes and bound temporary RAM.
            for first in range(0, node_count, 8192):
                last = min(first + 8192, node_count)
                interpolator = interp1d(
                    rain_epoch, rain_nodes[:, first:last], axis=0,
                    bounds_error=False,
                    fill_value=(rain_nodes[0, first:last], rain_nodes[-1, first:last]),
                    assume_sorted=True,
                )
                rain_rate[:, first:last] = interpolator(target_epoch).astype(np.float32)
        except (ValueError, RuntimeError) as exc:
            if not args.allow_missing_rain:
                raise
            print(f"WARNING: {exc}; using zero rainfall")
            rain_rate = np.zeros((len(time_s), node_count), dtype=np.float32)
            rain_epoch = np.empty(0, dtype=np.float64)
            rain_duration_s = np.empty(0, dtype=np.float64)

        ksat = load_node_parameter(
            args.soil_npz, "ksat_mps", node_count, args.default_ksat_mps)
        impervious = np.clip(load_node_parameter(
            args.landcover_npz, "impervious_fraction", node_count,
            args.default_impervious), 0.0, 1.0)
        rain_effective = (
            impervious[None, :] * rain_rate
            + (1.0 - impervious[None, :])
            * np.maximum(rain_rate - ksat[None, :], 0.0)
        ).astype(np.float32)

        usgs_path = Path(args.download_dir) / "usgs_harvey_iv.json"
        gauges = parse_usgs(usgs_path) if usgs_path.exists() else []
        approved = {x.strip() for x in args.approved_usgs_sites.split(",") if x.strip()}
        q, inlet_triangles, site_ids, river_audit = map_rivers(
            mesh, gauges, target_epoch, approved, args.inlet_triangles)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, time_s=time_s, time_iso=time_iso,
        rain_rate_mps=rain_rate, rain_effective_mps=rain_effective,
        ksat_mps=ksat, impervious_fraction=impervious,
        river_discharge_m3s=q, river_inlet_triangle_index=inlet_triangles,
        river_site_id=site_ids,
    )
    audit_path = output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps({
        "start_utc": start.isoformat(), "duration_hours": args.duration_hours,
        "rainfall": {
            "maximum_rate_mm_per_hour": float(np.max(rain_rate) * 3.6e6),
            "maximum_effective_rate_mm_per_hour": float(np.max(rain_effective) * 3.6e6),
            "source_records": int(len(rain_epoch)),
            "source_accumulation_hours_min": (
                float(np.min(rain_duration_s) / 3600.0) if len(rain_duration_s) else None),
            "source_accumulation_hours_max": (
                float(np.max(rain_duration_s) / 3600.0) if len(rain_duration_s) else None),
            "default_ksat_mps": args.default_ksat_mps,
            "default_impervious": args.default_impervious,
        },
        "rivers": river_audit,
        "warning": "Review every automatic gage-to-boundary mapping before production use.",
    }, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
