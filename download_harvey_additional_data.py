"""Download authoritative Hurricane Harvey compound-flood source data.

Downloads:
  * NOAA/NWS Stage IV hourly QPE source archives for 2017-08-21--2017-09-03
  * USGS instantaneous discharge (00060) and stage (00065)
  * USACE National Levee Database GIS layers within the mesh bounding box

The script only downloads and records source data. Use
``prepare_harvey_forcing.py`` to interpolate it to the ANUGA mesh.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tarfile
import urllib.error
import urllib.parse
import urllib.request

import numpy as np


NOAA_STAGE_IV = (
    "https://water.noaa.gov/resources/downloads/precip/stageIV/"
    "{year:04d}/{month:02d}/{day:02d}/"
    "ncep_stage_iv_source_files_{year:04d}{month:02d}{day:02d}.tar"
)
USGS_IV = "https://waterservices.usgs.gov/nwis/iv/"
NLD_FEATURE_SERVER = (
    "https://geospatial.sec.usace.army.mil/dls/rest/services/"
    "NLD/Public/FeatureServer"
)
NLD_LAYERS = {
    "crossings": 1,
    "pipes": 8,
    "closure_structures": 10,
    "embankments": 11,
    "floodwalls": 12,
}


def parse_utc(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def download(url: str, target: Path, timeout: int = 180) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return target
    request = urllib.request.Request(
        url, headers={"User-Agent": "Harvey-ANUGA-research/1.0"})
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with temporary.open("wb") as output:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def safe_extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive) as tar:
        safe = []
        for member in tar.getmembers():
            resolved = (destination / member.name).resolve()
            if root != resolved and root not in resolved.parents:
                raise ValueError(f"Unsafe path in {archive}: {member.name}")
            if member.isfile():
                safe.append(member)
        # ``filter=`` was added after Python 3.10. Paths were already checked
        # above, so the compatibility fallback remains traversal-safe.
        try:
            tar.extractall(destination, members=safe, filter="data")
        except TypeError:
            tar.extractall(destination, members=safe)


def download_stage_iv(start: datetime, end: datetime, output: Path) -> list[str]:
    archives = output / "stageiv_archives"
    extracted = output / "stageiv_hourly"
    records = []
    day = start.date()
    while day <= end.date():
        url = NOAA_STAGE_IV.format(year=day.year, month=day.month, day=day.day)
        target = archives / Path(urllib.parse.urlparse(url).path).name
        try:
            download(url, target)
            day_dir = extracted / day.isoformat()
            marker = day_dir / ".extracted"
            if not marker.exists():
                safe_extract_tar(target, day_dir)
                marker.touch()
            records.append(str(target))
        except urllib.error.HTTPError as exc:
            print(f"WARNING: Stage IV unavailable for {day}: HTTP {exc.code}")
        day += timedelta(days=1)
    return records


def download_usgs(start: datetime, end: datetime, bbox: tuple[float, ...],
                  output: Path, sites: list[str]) -> Path:
    params = {
        "format": "json",
        "startDT": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endDT": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "parameterCd": "00060,00065",
        "siteStatus": "all",
    }
    if sites:
        params["sites"] = ",".join(sites)
    else:
        params["bBox"] = ",".join(f"{x:.6f}" for x in bbox)
        params["siteType"] = "ST"
    url = USGS_IV + "?" + urllib.parse.urlencode(params, safe=",")
    target = output / "usgs_harvey_iv.json"
    download(url, target)
    return target


def download_nld(bbox: tuple[float, ...], output: Path) -> list[str]:
    paths = []
    envelope = ",".join(f"{x:.6f}" for x in bbox)
    for name, layer in NLD_LAYERS.items():
        params = {
            "where": "1=1",
            "geometry": envelope,
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
        }
        url = f"{NLD_FEATURE_SERVER}/{layer}/query?" + urllib.parse.urlencode(
            params, safe=",=")
        target = output / "nld" / f"{name}.geojson"
        try:
            download(url, target)
            paths.append(str(target))
        except urllib.error.HTTPError as exc:
            print(f"WARNING: NLD layer {name} failed: HTTP {exc.code}")
    return paths


def mesh_bbox(mesh_path: str) -> tuple[float, float, float, float]:
    with np.load(mesh_path) as mesh:
        if "lon" not in mesh.files or "lat" not in mesh.files:
            raise ValueError("mesh NPZ must contain lon and lat")
        lon = np.asarray(mesh["lon"], float)
        lat = np.asarray(mesh["lat"], float)
    return float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", default="2017-08-21T18:00:00Z")
    parser.add_argument("--end", default="2017-09-03T00:00:00Z")
    parser.add_argument(
        "--usgs-sites", default="",
        help="Comma-separated approved gage IDs. Empty downloads all stream sites in the bbox.")
    parser.add_argument("--skip-rain", action="store_true")
    parser.add_argument("--skip-usgs", action="store_true")
    parser.add_argument("--skip-nld", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    start, end = parse_utc(args.start), parse_utc(args.end)
    if end <= start:
        raise ValueError("--end must be after --start")
    bbox = mesh_bbox(args.mesh)
    sites = [x.strip() for x in args.usgs_sites.split(",") if x.strip()]
    manifest = {
        "start_utc": start.isoformat(), "end_utc": end.isoformat(),
        "bbox_wsen": bbox,
        "stageiv_archives": [] if args.skip_rain else download_stage_iv(start, end, output),
        "usgs": None if args.skip_usgs else str(
            download_usgs(start, end, bbox, output, sites)),
        "nld_layers": [] if args.skip_nld else download_nld(bbox, output),
        "notes": [
            "USGS bbox mode may contain multiple gauges on the same river; approve a subset before forcing.",
            "NLD geometries are downloaded for mesh rebuilding and are not burned into an existing mesh automatically.",
            "Stage IV hourly GRIB extraction requires pygrib/eccodes during preparation.",
        ],
    }
    manifest_path = output / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
