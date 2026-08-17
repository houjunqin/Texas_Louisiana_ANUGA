"""Command-line orchestration for one ANUGA/HOPNet scenario.

Example:
  python build_anuga_dataset.py --mesh mesh.npz --track track.npz \
      --output dataset/storm_001.npz --duration-hours 48 --output-minutes 15
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import numpy as np

from anuga_adcirc_dataset import MeshData, read_sww_state, run_anuga_scenario, save_hopnet_scenario
from storm_forcing import StormTrack, generate_holland_forcing


RUNNER_VERSION = "2026-08-14-harvey-compound-v1"


def _load_mesh(path: str) -> MeshData:
    z = np.load(path)
    return MeshData(
        xy_m=z["xy_m"], triangles=z["triangles"],
        bed_elevation_m=z["bed_elevation_m"], manning_n=z["manning_n"],
        open_boundary_edges=z["open_boundary_edges"],
    )


def _load_track(path: str) -> StormTrack:
    z = np.load(path)
    optional_b = z["holland_b"] if "holland_b" in z.files else None
    return StormTrack(
        time_s=z["time_s"], center_x_m=z["center_x_m"], center_y_m=z["center_y_m"],
        central_pressure_hpa=z["central_pressure_hpa"],
        outer_pressure_hpa=z["outer_pressure_hpa"], rmw_km=z["rmw_km"],
        translation_speed_ms=z["translation_speed_ms"], heading_deg=z["heading_deg"],
        holland_b=optional_b,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--version", action="version", version=f"%(prog)s {RUNNER_VERSION}")
    p.add_argument("--mesh", required=True)
    p.add_argument("--track", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--scenario-id", default="storm_000")
    p.add_argument("--duration-hours", type=float, default=48.0)
    p.add_argument("--output-minutes", type=float, default=15.0)
    p.add_argument("--open-boundary-segments", type=int, default=16)
    p.add_argument(
        "--omp-threads", type=int,
        help="ANUGA OpenMP thread count; otherwise use OMP_NUM_THREADS/default",
    )
    p.add_argument(
        "--uncompressed-output", action="store_true",
        help="Write a larger NPZ much faster by disabling ZIP compression",
    )
    p.add_argument(
        "--forcing-chunk-times", type=int, default=8,
        help="Holland forcing generation time slices per memory-bounded chunk",
    )
    p.add_argument(
        "--tide-npz",
        help=("Optional NPZ containing dense tide_m [T,N], or sparse "
              "tide_boundary_m [T,Nb] and boundary_node_index [Nb]"),
    )
    p.add_argument(
        "--additional-forcing-npz",
        help=("Output of prepare_harvey_forcing.py containing rainfall, "
              "losses, river hydrographs, and optional wave force"),
    )
    args = p.parse_args()

    mesh = _load_mesh(args.mesh)
    track = _load_track(args.track)
    dt = args.output_minutes * 60.0
    times = np.arange(0.0, args.duration_hours * 3600.0 + 0.5 * dt, dt)
    tide_file = np.load(args.tide_npz) if args.tide_npz else None
    dense_tide = tide_file["tide_m"] if tide_file is not None and "tide_m" in tide_file.files else None
    forcing = generate_holland_forcing(
        mesh.xy_m, times, track, tide_m=dense_tide, minimal=True,
        time_chunk_size=args.forcing_chunk_times)
    if tide_file is not None and "tide_boundary_m" in tide_file.files:
        tide_time = np.asarray(tide_file["time_s"], dtype=float)
        if tide_time.shape != times.shape or not np.allclose(tide_time, times):
            raise ValueError("tide NPZ time_s must match the requested ANUGA output times")
        forcing.pop("tide_m", None)
        forcing["tide_boundary_m"] = tide_file["tide_boundary_m"]
        forcing["tide_boundary_node_index"] = tide_file["boundary_node_index"]
    forcing["time_s"] = times

    if args.additional_forcing_npz:
        with np.load(args.additional_forcing_npz, allow_pickle=False) as extra:
            if "time_s" not in extra.files:
                raise ValueError("additional forcing NPZ is missing time_s")
            extra_time = np.asarray(extra["time_s"], dtype=float)
            if extra_time.shape != times.shape or not np.allclose(extra_time, times):
                raise ValueError(
                    "additional forcing time_s must match duration/output-minutes")
            for key in extra.files:
                if key != "time_s":
                    forcing[key] = extra[key]

    # Temporary SWW is discarded after conversion; only the compact training file remains.
    with tempfile.TemporaryDirectory(prefix="anuga_sww_") as work:
        sww = run_anuga_scenario(
            mesh, times, forcing, work, args.scenario_id,
            open_boundary_segments=args.open_boundary_segments,
            omp_num_threads=args.omp_threads,
        )
        solution = read_sww_state(sww, mesh.xy_m)
        saved = save_hopnet_scenario(
            args.output, mesh, solution, forcing,
            compress=not args.uncompressed_output,
        )
    print(saved)


if __name__ == "__main__":
    main()
