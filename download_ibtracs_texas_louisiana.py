"""Download IBTrACS and extract storms affecting Texas and Louisiana.

The program downloads NOAA's official North Atlantic IBTrACS v04r01 CSV,
identifies storms having at least one track point inside a buffered regional
box, and retains the complete track of every selected storm.  It can also
write one selected storm in the NPZ convention consumed by the accompanying
ANUGA forcing code.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

import numpy as np


IBTRACS_NA_URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.NA.list.v04r01.csv"
)
KNOT_TO_MS = 0.514444
NMI_TO_KM = 1.852
EARTH_RADIUS_M = 6_371_008.8
RHO_AIR = 1.225


def number(value: str | None) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return np.nan


def first_number(row: dict[str, str], *keys: str) -> float:
    for key in keys:
        value = number(row.get(key))
        if np.isfinite(value):
            return value
    return np.nan


def download(url: str, destination: Path, force: bool = False) -> None:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        print(f"Using cached archive: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "ANUGA-IBTrACS/1.0"})
    print(f"Downloading {url}")
    with urlopen(request, timeout=120) as response, partial.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
    partial.replace(destination)


def read_tracks(path: Path) -> dict[str, list[dict[str, object]]]:
    tracks: dict[str, list[dict[str, object]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            sid = str(raw.get("SID", "")).strip()
            iso = str(raw.get("ISO_TIME", "")).strip()
            # The second IBTrACS CSV row contains units rather than a record.
            if not sid or not iso or not re.match(r"^\d{4}-\d{2}-\d{2}", iso):
                continue
            lon = first_number(raw, "USA_LON", "LON")
            lat = first_number(raw, "USA_LAT", "LAT")
            if lon > 180.0:
                lon -= 360.0
            record: dict[str, object] = {
                "sid": sid,
                "season": int(number(raw.get("SEASON"))),
                "name": str(raw.get("NAME", "UNNAMED")).strip() or "UNNAMED",
                "iso_time": iso.replace(" ", "T"),
                "lon_deg": lon,
                "lat_deg": lat,
                "nature": str(raw.get("NATURE", "")).strip(),
                "usa_status": str(raw.get("USA_STATUS", "")).strip(),
                "usa_atcf_id": str(raw.get("USA_ATCF_ID", "")).strip(),
                "max_wind_kt": first_number(raw, "USA_WIND", "WMO_WIND"),
                "central_pressure_hpa": first_number(raw, "USA_PRES", "WMO_PRES"),
                "outer_pressure_hpa": first_number(raw, "USA_POCI"),
                "rmw_nmi": first_number(raw, "USA_RMW"),
                "roci_nmi": first_number(raw, "USA_ROCI"),
                "dist2land_km": first_number(raw, "DIST2LAND"),
            }
            for radius in (34, 50, 64):
                for quadrant in ("NE", "SE", "SW", "NW"):
                    record[f"r{radius}_{quadrant.lower()}_nmi"] = first_number(
                        raw, f"USA_R{radius}_{quadrant}"
                    )
            tracks.setdefault(sid, []).append(record)
    for records in tracks.values():
        records.sort(key=lambda item: str(item["iso_time"]))
    return tracks


def select_tracks(tracks: dict[str, list[dict[str, object]]], bbox: list[float],
                  buffer_deg: float, start_year: int, end_year: int,
                  name: str | None) -> dict[str, list[dict[str, object]]]:
    west, east, south, north = bbox
    west -= buffer_deg
    east += buffer_deg
    south -= buffer_deg
    north += buffer_deg
    wanted_name = name.upper() if name else None
    selected = {}
    for sid, records in tracks.items():
        season = int(records[0]["season"])
        storm_name = str(records[0]["name"]).upper()
        if not start_year <= season <= end_year:
            continue
        if wanted_name and storm_name != wanted_name:
            continue
        intersects = any(
            np.isfinite(float(r["lon_deg"])) and np.isfinite(float(r["lat_deg"]))
            and west <= float(r["lon_deg"]) <= east
            and south <= float(r["lat_deg"]) <= north
            for r in records
        )
        if intersects:
            selected[sid] = records
    return selected


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_") or "UNNAMED"


def write_csvs(selected: dict[str, list[dict[str, object]]], output_dir: Path) -> Path:
    tracks_dir = output_dir / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    catalog_rows = []
    for sid, records in sorted(selected.items()):
        name = str(records[0]["name"])
        path = tracks_dir / f"{sid}_{safe_name(name)}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        winds = np.asarray([r["max_wind_kt"] for r in records], float)
        pressures = np.asarray([r["central_pressure_hpa"] for r in records], float)
        catalog_rows.append({
            "sid": sid, "season": records[0]["season"], "name": name,
            "usa_atcf_id": records[0]["usa_atcf_id"],
            "start_iso": records[0]["iso_time"], "end_iso": records[-1]["iso_time"],
            "points": len(records),
            "max_wind_kt": np.nanmax(winds) if np.isfinite(winds).any() else "",
            "min_pressure_hpa": np.nanmin(pressures) if np.isfinite(pressures).any() else "",
            "track_csv": str(path.relative_to(output_dir)),
        })
    catalog = output_dir / "texas_louisiana_storm_catalog.csv"
    fields = list(catalog_rows[0]) if catalog_rows else ["sid", "season", "name"]
    with catalog.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(catalog_rows)
    return catalog


def fill_series(values: np.ndarray, fallback: float) -> np.ndarray:
    values = np.asarray(values, float)
    good = np.isfinite(values)
    if not good.any():
        return np.full(values.shape, fallback, dtype=float)
    index = np.arange(len(values))
    return np.interp(index, index[good], values[good])


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=timezone.utc)


def crop_time_window(records: list[dict[str, object]], start_iso: str | None,
                     end_iso: str | None) -> list[dict[str, object]]:
    start = parse_time(start_iso) if start_iso else None
    end = parse_time(end_iso) if end_iso else None
    if start and end and end <= start:
        raise ValueError("--track-end-iso must be later than --track-start-iso")
    cropped = []
    for record in records:
        instant = parse_time(str(record["iso_time"]))
        if start is not None and instant < start:
            continue
        if end is not None and instant > end:
            continue
        cropped.append(record)
    if len(cropped) < 2:
        raise ValueError("The requested track time window contains fewer than two IBTrACS records")
    return cropped


def spherical_laea(lon_deg: np.ndarray, lat_deg: np.ndarray,
                   lon0_deg: float, lat0_deg: float) -> np.ndarray:
    lon = np.deg2rad(np.asarray(lon_deg, float))
    lat = np.deg2rad(np.asarray(lat_deg, float))
    lon0, lat0 = np.deg2rad([lon0_deg, lat0_deg])
    dlon = lon - lon0
    denominator = 1 + np.sin(lat0) * np.sin(lat) + np.cos(lat0) * np.cos(lat) * np.cos(dlon)
    scale = np.sqrt(2 / np.maximum(denominator, 1.0e-12))
    x = EARTH_RADIUS_M * scale * np.cos(lat) * np.sin(dlon)
    y = EARTH_RADIUS_M * scale * (
        np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(dlon)
    )
    return np.column_stack((x, y))


def projection_origin(mesh_path: Path | None, bbox: list[float]) -> tuple[float, float]:
    if mesh_path is None:
        return (bbox[0] + bbox[1]) / 2, (bbox[2] + bbox[3]) / 2
    with np.load(mesh_path, allow_pickle=False) as mesh:
        if "projection_lon0_deg" in mesh and "projection_lat0_deg" in mesh:
            return float(mesh["projection_lon0_deg"]), float(mesh["projection_lat0_deg"])
        if "lon" in mesh and "lat" in mesh:
            return float(np.mean(mesh["lon"])), float(np.mean(mesh["lat"]))
        if "pos" in mesh:
            pos = np.asarray(mesh["pos"], float)
            return float(pos[:, 0].mean()), float(pos[:, 1].mean())
    raise ValueError(f"Cannot determine geographic projection origin from {mesh_path}")


def write_anuga_npz(records: list[dict[str, object]], path: Path,
                    mesh_path: Path | None, bbox: list[float], default_rmw_km: float) -> None:
    iso = np.asarray([str(r["iso_time"]) for r in records], dtype="U32")
    datetimes = [parse_time(value) for value in iso]
    epoch = np.asarray([item.timestamp() for item in datetimes], float)
    if np.any(np.diff(epoch) <= 0):
        raise ValueError("Selected track times are not strictly increasing")
    time_s = epoch - epoch[0]
    lon = np.asarray([r["lon_deg"] for r in records], float)
    lat = np.asarray([r["lat_deg"] for r in records], float)
    lon0, lat0 = projection_origin(mesh_path, bbox)
    xy = spherical_laea(lon, lat, lon0, lat0)
    velocity_x = np.gradient(xy[:, 0], time_s, edge_order=1)
    velocity_y = np.gradient(xy[:, 1], time_s, edge_order=1)
    translation_speed = np.hypot(velocity_x, velocity_y)
    heading = np.mod(np.rad2deg(np.arctan2(velocity_x, velocity_y)), 360.0)

    vmax_kt = fill_series([r["max_wind_kt"] for r in records], 0.0)
    vmax_ms = vmax_kt * KNOT_TO_MS
    pressure = fill_series([r["central_pressure_hpa"] for r in records], 1010.0)
    outer = fill_series([r["outer_pressure_hpa"] for r in records], 1013.25)
    outer = np.maximum(outer, pressure + 1.0)
    rmw_km = fill_series([number(str(r["rmw_nmi"])) * NMI_TO_KM for r in records], default_rmw_km)
    roci_km = fill_series([number(str(r["roci_nmi"])) * NMI_TO_KM for r in records], 300.0)
    delta_pressure_pa = np.maximum(outer - pressure, 1.0) * 100.0
    gradient_wind = vmax_ms / 0.9
    holland_b = np.clip(RHO_AIR * np.e * gradient_wind**2 / delta_pressure_pa, 0.8, 2.5)

    radii = {}
    for threshold in (34, 50, 64):
        radii[f"r{threshold}_quadrant_km"] = np.column_stack([
            np.asarray([r[f"r{threshold}_{quadrant}_nmi"] for r in records], float) * NMI_TO_KM
            for quadrant in ("ne", "se", "sw", "nw")
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, time_s=time_s, iso_time=iso,
        center_x_m=xy[:, 0], center_y_m=xy[:, 1],
        center_lon_deg=lon, center_lat_deg=lat,
        central_pressure_hpa=pressure, outer_pressure_hpa=outer,
        rmw_km=rmw_km, roci_km=roci_km,
        translation_speed_ms=translation_speed, heading_deg=heading,
        holland_b=holland_b, max_wind_ms=vmax_ms,
        projection=np.asarray("spherical_laea"),
        projection_lon0_deg=np.asarray(lon0), projection_lat0_deg=np.asarray(lat0),
        ibtracs_sid=np.asarray(str(records[0]["sid"])),
        storm_name=np.asarray(str(records[0]["name"])), **radii,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--url", default=IBTRACS_NA_URL)
    parser.add_argument("--archive-cache", type=Path)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--bbox", nargs=4, type=float,
                        default=[-98.1, -91.9, 25.2, 30.8],
                        metavar=("WEST", "EAST", "SOUTH", "NORTH"))
    parser.add_argument("--buffer-deg", type=float, default=2.5)
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--end-year", type=int, default=datetime.now().year)
    parser.add_argument("--name", help="Optional exact IBTrACS storm name, e.g. HARVEY")
    parser.add_argument("--season", type=int, help="Optional exact season")
    parser.add_argument("--anuga-npz", type=Path,
                        help="Write the single selected storm in ANUGA track convention")
    parser.add_argument("--mesh", type=Path,
                        help="Mesh NPZ used to match the ANUGA map projection")
    parser.add_argument("--track-start-iso",
                        help="Optional inclusive start of the ANUGA NPZ time window")
    parser.add_argument("--track-end-iso",
                        help="Optional inclusive end of the ANUGA NPZ time window")
    parser.add_argument("--default-rmw-km", type=float, default=45.0)
    args = parser.parse_args()

    if args.season is not None:
        args.start_year = args.end_year = args.season
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.archive_cache or args.output_dir / "ibtracs.NA.list.v04r01.csv"
    download(args.url, archive, args.force_download)
    selected = select_tracks(
        read_tracks(archive), args.bbox, args.buffer_deg,
        args.start_year, args.end_year, args.name,
    )
    catalog = write_csvs(selected, args.output_dir)
    selection = {
        "archive_url": args.url, "bbox": args.bbox, "buffer_deg": args.buffer_deg,
        "start_year": args.start_year, "end_year": args.end_year,
        "name": args.name, "selected_storm_count": len(selected),
        "selected_sids": sorted(selected),
    }
    (args.output_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(f"Selected {len(selected)} storms; catalog: {catalog}")

    if args.anuga_npz:
        if len(selected) != 1:
            raise ValueError(
                f"--anuga-npz requires exactly one storm; current filters selected {len(selected)}. "
                "Add --name and --season."
            )
        records = next(iter(selected.values()))
        records = crop_time_window(records, args.track_start_iso, args.track_end_iso)
        write_anuga_npz(records, args.anuga_npz, args.mesh, args.bbox, args.default_rmw_km)
        print(f"ANUGA track: {args.anuga_npz}")


if __name__ == "__main__":
    main()
