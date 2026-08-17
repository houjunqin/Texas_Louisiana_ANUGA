"""Generate FES2022b ocean-tide boundary elevations for an ANUGA mesh.

The recommended backend is the official CNES ``pyfes`` package.  Predictions
are evaluated only at nodes referenced by ``open_boundary_edges`` and stored in
a sparse boundary format; generating a dense [time, all_nodes] array is neither
necessary for ANUGA nor memory efficient for a regional ADCIRC mesh.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np


FES2022_CONSTITUENTS = (
    "2N2", "Eps2", "J1", "K1", "K2", "L2", "Lambda2", "M2", "M3",
    "M4", "M6", "M8", "Mf", "MKS2", "Mm", "MN4", "MS4", "MSf",
    "MSqm", "Mtm", "Mu2", "N2", "N4", "Nu2", "O1", "P1", "Q1",
    "R2", "S1", "S2", "S4", "Sa", "Ssa", "T2",
)


def boundary_nodes(open_boundary_edges: np.ndarray) -> np.ndarray:
    edges = np.asarray(open_boundary_edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2 or len(edges) == 0:
        raise ValueError("open_boundary_edges must have shape [B,2] and be non-empty")
    return np.unique(edges)


def discover_cartesian_tide_files(root: str | Path) -> dict[str, Path]:
    """Find FES NetCDF files below *root* and map them to constituents.

    FES archives are distributed with several naming schemes, including
    ``m2.nc``, ``M2_tide.nc`` and ``m2_ocean_tide_extrapolated.nc``.  Match a
    constituent only as a complete filename token so, for example, ``M2`` does
    not accidentally match another name.
    """

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"FES directory does not exist: {root}")
    canonical = {name.lower(): name for name in FES2022_CONSTITUENTS}
    found: dict[str, Path] = {}
    candidates = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".nc", ".nc4"}
    ]
    for path in candidates:
        lower = path.name.lower()
        if "radial" in lower or "load" in lower:
            continue
        tokens = set(filter(None, re.split(r"[^a-z0-9]+", path.stem.lower())))
        matches = [canonical[token] for token in tokens if token in canonical]
        if len(matches) == 1:
            name = matches[0]
            previous = found.get(name)
            if previous is not None and previous != path.resolve():
                raise RuntimeError(
                    f"Multiple ocean files match constituent {name}: "
                    f"{previous} and {path.resolve()}")
            found[name] = path.resolve()
    if not found:
        archives = [
            path for path in root.rglob("*")
            if path.is_file() and any(
                path.name.lower().endswith(ext)
                for ext in (
                    ".zip", ".tar", ".tar.gz", ".tgz", ".gz",
                    ".tar.xz", ".xz", ".tar.bz2", ".bz2",
                ))
        ]
        samples = ", ".join(path.name for path in candidates[:8]) or "none"
        archive_note = (
            f" Found {len(archives)} archive(s); extract them first."
            if archives else ""
        )
        raise FileNotFoundError(
            f"No recognizable FES constituent NetCDF files were found below {root}. "
            f"NetCDF samples: {samples}.{archive_note} Expected a filename token "
            "such as m2 in m2.nc or m2_ocean_tide_extrapolated.nc."
        )
    print(
        f"Discovered {len(found)} FES ocean constituents below {root}: "
        + ", ".join(sorted(found))
    )
    return found


def write_pyfes_config(path: str | Path, tide_files: dict[str, Path]) -> Path:
    """Write a current PyFES Darwin/cartesian configuration."""

    path = Path(path)
    lines = ["engine: darwin", "tide:", "  cartesian:", "    paths:"]
    order = {name: i for i, name in enumerate(FES2022_CONSTITUENTS)}
    for name, source in sorted(tide_files.items(), key=lambda item: order[item[0]]):
        # JSON-style quoting is valid YAML and safely handles spaces in paths.
        escaped = str(source).replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'      {name}: "{escaped}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def predict_boundary_tide(
    config_path: str | Path,
    longitude_deg: np.ndarray,
    latitude_deg: np.ndarray,
    dates_utc: np.ndarray,
    atlas_units: str = "cm",
    datum_offset_m: float = 0.0,
    batch_size: int = 200_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate official PyFES tide plus long-period equilibrium tide.

    Returns ``(stage_m, flags)`` with shape ``[T,Nb]``.  ``datum_offset_m``
    converts FES zero-mean elevations to the vertical datum used by the mesh.
    """

    try:
        import pyfes
    except ImportError as exc:
        raise RuntimeError(
            "Install the official predictor first: conda install -c conda-forge pyfes"
        ) from exc

    lon = np.asarray(longitude_deg, dtype=np.float64)
    lat = np.asarray(latitude_deg, dtype=np.float64)
    dates = np.asarray(dates_utc, dtype="datetime64[s]")
    if lon.shape != lat.shape or lon.ndim != 1 or dates.ndim != 1:
        raise ValueError("longitude/latitude must be [Nb] and dates_utc must be [T]")
    if atlas_units not in {"cm", "m"}:
        raise ValueError("atlas_units must be 'cm' or 'm'")

    config = pyfes.config.load(str(config_path))
    count = len(dates) * len(lon)
    flat_tide = np.empty(count, dtype=np.float64)
    flat_flags = np.empty(count, dtype=np.int16)
    # Flatten in time-major order: each consecutive block contains all nodes.
    flat_dates = np.repeat(dates, len(lon))
    flat_lon = np.tile(lon, len(dates))
    flat_lat = np.tile(lat, len(dates))
    for first in range(0, count, batch_size):
        last = min(first + batch_size, count)
        tide, long_period, flags = pyfes.evaluate_tide(
            config.models["tide"],
            flat_dates[first:last],
            flat_lon[first:last],
            flat_lat[first:last],
            settings=config.settings,
        )
        flat_tide[first:last] = np.asarray(tide) + np.asarray(long_period)
        flat_flags[first:last] = np.asarray(flags, dtype=np.int16)

    scale = 0.01 if atlas_units == "cm" else 1.0
    stage_m = flat_tide.reshape(len(dates), len(lon)) * scale + datum_offset_m
    if not np.all(np.isfinite(stage_m)):
        bad = int(np.size(stage_m) - np.isfinite(stage_m).sum())
        raise ValueError(
            f"PyFES returned {bad} invalid values. Use the extrapolated ocean-tide "
            "grid or move invalid boundary points offshore; do not replace them silently."
        )
    return stage_m.astype(np.float32), flat_flags.reshape(len(dates), len(lon))


def _start_from_track(track_path: str | Path) -> np.datetime64:
    with np.load(track_path, allow_pickle=True) as track:
        if "iso_time" not in track.files or len(track["iso_time"]) == 0:
            raise ValueError("track NPZ has no iso_time array; pass --start-utc explicitly")
        return np.datetime64(str(track["iso_time"][0]), "s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create sparse FES2022b ocean-boundary stage for ANUGA")
    parser.add_argument("--mesh", required=True, help="Mesh NPZ with lon/lat and open edges")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="Existing PyFES FES2022b YAML configuration")
    source.add_argument("--fes-dir", help="Directory containing *_tide.nc ocean files")
    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument("--start-utc", help="First time, for example 2017-08-21T18:00:00")
    start.add_argument("--track", help="Track NPZ whose first iso_time supplies start UTC")
    parser.add_argument("--duration-hours", type=float, required=True)
    parser.add_argument("--interval-minutes", type=float, default=30.0)
    parser.add_argument("--atlas-units", choices=("cm", "m"), default="cm")
    parser.add_argument("--datum-offset-m", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with np.load(args.mesh) as mesh:
        required = {"lon", "lat", "open_boundary_edges"}
        missing = required - set(mesh.files)
        if missing:
            raise ValueError(f"mesh is missing keys: {sorted(missing)}")
        nodes = boundary_nodes(mesh["open_boundary_edges"])
        lon = np.asarray(mesh["lon"])[nodes]
        lat = np.asarray(mesh["lat"])[nodes]

    start_utc = (np.datetime64(args.start_utc, "s") if args.start_utc
                 else _start_from_track(args.track))
    step_s = int(round(args.interval_minutes * 60.0))
    if step_s <= 0 or args.duration_hours <= 0:
        raise ValueError("duration and interval must be positive")
    time_s = np.arange(0.0, args.duration_hours * 3600.0 + 0.5 * step_s, step_s)
    dates = start_utc + time_s.astype("timedelta64[s]")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = write_pyfes_config(
            output.with_name("fes2022b_ocean_tide.yml"),
            discover_cartesian_tide_files(args.fes_dir),
        )
    tide_m, flags = predict_boundary_tide(
        config_path, lon, lat, dates, args.atlas_units, args.datum_offset_m)
    np.savez_compressed(
        output,
        time_s=time_s.astype(np.float64),
        datetime_utc=dates,
        boundary_node_index=nodes,
        boundary_lon=lon,
        boundary_lat=lat,
        tide_boundary_m=tide_m,
        pyfes_flags=flags,
        datum_offset_m=np.asarray(args.datum_offset_m),
        source=np.asarray("FES2022b ocean tide + PyFES long-period equilibrium tide"),
    )
    print(output)
    print(f"boundary nodes={len(nodes)}, times={len(time_s)}, "
          f"range_m=[{tide_m.min():.3f}, {tide_m.max():.3f}]")


if __name__ == "__main__":
    main()
