"""ANUGA-to-HOPNet dataset bridge for ADCIRC-like coastal experiments.

ANUGA is imported lazily so graph construction and dataset packaging can be
tested on machines where the solver is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class MeshData:
    """Triangular mesh in a projected CRS with metre units."""

    xy_m: np.ndarray                 # [N,2]
    triangles: np.ndarray            # [M,3]
    bed_elevation_m: np.ndarray      # [N], positive upward (ocean usually < 0)
    manning_n: np.ndarray            # [N]
    open_boundary_edges: np.ndarray  # [B,2] undirected node pairs
    lon_deg: Optional[np.ndarray] = None  # [N], east-positive longitude
    lat_deg: Optional[np.ndarray] = None  # [N], north-positive latitude

    def validate(self) -> None:
        n = len(self.xy_m)
        if np.asarray(self.xy_m).shape != (n, 2):
            raise ValueError("xy_m must have shape [N,2]")
        if np.asarray(self.triangles).ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError("triangles must have shape [M,3]")
        if np.asarray(self.bed_elevation_m).shape != (n,):
            raise ValueError("bed_elevation_m must have shape [N]")
        if np.asarray(self.manning_n).shape != (n,):
            raise ValueError("manning_n must have shape [N]")
        if self.lon_deg is not None and np.asarray(self.lon_deg).shape != (n,):
            raise ValueError("lon_deg must have shape [N]")
        if self.lat_deg is not None and np.asarray(self.lat_deg).shape != (n,):
            raise ValueError("lat_deg must have shape [N]")
        if np.min(self.triangles) < 0 or np.max(self.triangles) >= n:
            raise ValueError("triangle node indices are outside [0,N)")


def triangle_boundary_tags(
    triangles: np.ndarray, open_boundary_edges: np.ndarray
) -> dict[tuple[int, int], str]:
    """Convert node-pair boundaries to ANUGA ``(triangle, local_edge)`` tags.

    ANUGA local edge ``k`` is opposite triangle vertex ``k``.
    """

    return triangle_boundary_topology(triangles, open_boundary_edges)[0]


def _undirected_edge_keys(edges: np.ndarray, node_count: int) -> np.ndarray:
    edges = np.asarray(edges, dtype=np.int64)
    lo = np.minimum(edges[:, 0], edges[:, 1])
    hi = np.maximum(edges[:, 0], edges[:, 1])
    return lo * np.int64(node_count) + hi


def triangle_boundary_topology(
    triangles: np.ndarray, open_boundary_edges: np.ndarray
) -> tuple[dict[tuple[int, int], str],
           dict[tuple[int, int], tuple[int, int]]]:
    """Vectorized exterior-edge extraction and ANUGA boundary tagging."""

    tri = np.asarray(triangles, dtype=np.int64)
    node_count = int(tri.max()) + 1
    # Block order matches ANUGA: local edge k is opposite triangle vertex k.
    all_edges = np.concatenate((tri[:, [1, 2]], tri[:, [2, 0]], tri[:, [0, 1]]))
    keys = _undirected_edge_keys(all_edges, node_count)
    _, first, counts = np.unique(keys, return_index=True, return_counts=True)
    exterior_index = first[counts == 1]
    exterior_edges = np.sort(all_edges[exterior_index], axis=1)
    exterior_keys = keys[exterior_index]
    triangle_index = exterior_index % len(tri)
    local_edge = exterior_index // len(tri)

    requested = np.sort(
        np.asarray(open_boundary_edges, dtype=np.int64).reshape(-1, 2), axis=1)
    requested_keys = _undirected_edge_keys(requested, node_count)
    is_open = np.isin(exterior_keys, requested_keys)
    invalid = np.setdiff1d(requested_keys, exterior_keys, assume_unique=False)
    if len(invalid):
        raise ValueError(f"{len(invalid)} open edges are not exterior triangle edges")

    tags: dict[tuple[int, int], str] = {}
    ref_to_open_pair: dict[tuple[int, int], tuple[int, int]] = {}
    for ti, le, pair, open_flag in zip(
            triangle_index, local_edge, exterior_edges, is_open):
        ref = (int(ti), int(le))
        tags[ref] = "open" if open_flag else "land"
        if open_flag:
            ref_to_open_pair[ref] = (int(pair[0]), int(pair[1]))
    return tags, ref_to_open_pair


def mesh_graph(mesh: MeshData) -> dict[str, np.ndarray]:
    """Build the physical mesh graph (not a KNN approximation)."""

    mesh.validate()
    tri = np.asarray(mesh.triangles, dtype=np.int64)
    undirected = np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]))
    keys = np.unique(_undirected_edge_keys(undirected, len(mesh.xy_m)))
    lo = keys // np.int64(len(mesh.xy_m))
    hi = keys % np.int64(len(mesh.xy_m))
    edges = np.stack((np.concatenate((lo, hi)), np.concatenate((hi, lo))))
    src, dst = edges
    delta = mesh.xy_m[dst] - mesh.xy_m[src]
    distance = np.linalg.norm(delta, axis=1, keepdims=True)
    return {
        "edge_index": edges,
        "edge_attr": np.concatenate((delta, distance), axis=1).astype(np.float32),
    }


class TimeNodeField:
    """Time interpolation plus cached nearest-node spatial sampling for ANUGA."""

    def __init__(self, time_s: np.ndarray, node_xy: np.ndarray, values: np.ndarray):
        from scipy.spatial import cKDTree

        self.time_s = np.asarray(time_s, float)
        self.values = np.asarray(values, float)
        if self.values.shape != (len(self.time_s), len(node_xy)):
            raise ValueError("field values must have shape [T,N]")
        self.tree = cKDTree(np.asarray(node_xy, float))
        self._cache: dict[tuple[int, float, float], np.ndarray] = {}

    def __call__(self, t: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x, y = np.asarray(x), np.asarray(y)
        key = (x.size, float(x.flat[0]), float(y.flat[0]))
        indices = self._cache.get(key)
        if indices is None:
            indices = self.tree.query(np.column_stack((x.ravel(), y.ravel())))[1]
            self._cache[key] = indices
        hi = int(np.searchsorted(self.time_s, t, side="right"))
        hi = min(max(hi, 1), len(self.time_s) - 1)
        lo = hi - 1
        alpha = np.clip((t - self.time_s[lo]) / (self.time_s[hi] - self.time_s[lo]), 0.0, 1.0)
        return ((1.0 - alpha) * self.values[lo, indices] + alpha * self.values[hi, indices]).reshape(x.shape)


class TimeArrayField:
    """Linear temporal interpolation of full node arrays without spatial search."""

    def __init__(self, time_s: np.ndarray, values: np.ndarray, node_count: int):
        self.time_s = np.asarray(time_s, dtype=np.float64)
        self.values = np.asarray(values, dtype=np.float32)
        if self.values.shape != (len(self.time_s), node_count):
            raise ValueError("time-varying node field must have shape [T,N]")

    def at(self, t: float) -> np.ndarray:
        if t <= self.time_s[0]:
            return self.values[0]
        if t >= self.time_s[-1]:
            return self.values[-1]
        hi = int(np.searchsorted(self.time_s, t, side="right"))
        lo = hi - 1
        alpha = np.float32(
            (t - self.time_s[lo]) / (self.time_s[hi] - self.time_s[lo]))
        return (np.float32(1.0) - alpha) * self.values[lo] + alpha * self.values[hi]


class VectorizedWindStress:
    """ANUGA-compatible wind forcing without its per-triangle Python loop."""

    def __init__(self, time_s: np.ndarray, triangles: np.ndarray,
                 wind_u10: np.ndarray, wind_v10: np.ndarray,
                 stress_coefficient: Optional[float] = None):
        self.triangles = np.asarray(triangles, dtype=np.int64)
        node_count = np.asarray(wind_u10).shape[1]
        if int(self.triangles.max()) >= node_count:
            raise ValueError("wind fields do not cover every triangle node")
        self.wind_u = TimeArrayField(time_s, wind_u10, node_count)
        self.wind_v = TimeArrayField(time_s, wind_v10, node_count)
        if stress_coefficient is None:
            from anuga.config import eta_w, rho_a, rho_w
            stress_coefficient = eta_w * rho_a / rho_w
        self.const = np.float32(stress_coefficient)

    def __call__(self, domain) -> None:
        t = domain.get_time()
        u = self.wind_u.at(t)[self.triangles].mean(axis=1)
        v = self.wind_v.at(t)[self.triangles].mean(axis=1)
        speed = np.hypot(u, v)
        domain.quantities["xmomentum"].explicit_update += self.const * speed * u
        domain.quantities["ymomentum"].explicit_update += self.const * speed * v


class VectorizedBarometricPressure:
    """Atmospheric-pressure source using pressure in Pascals.

    For depth-integrated momentum, the barometric source is
    ``-(h/rho_w) * grad(p_a)``.  The negative sign accelerates water toward
    lower atmospheric pressure.  Supplying hPa here would make the force 100
    times too small.
    """

    def __init__(self, time_s: np.ndarray, node_xy: np.ndarray,
                 triangles: np.ndarray, pressure_pa: np.ndarray,
                 water_density: Optional[float] = None):
        self.triangles = np.asarray(triangles, dtype=np.int64)
        node_count = len(node_xy)
        self.pressure = TimeArrayField(time_s, pressure_pa, node_count)
        if water_density is None:
            from anuga.config import rho_w
            water_density = rho_w
        self.inv_rho_w = np.float32(1.0 / water_density)

        q = np.asarray(node_xy, dtype=np.float64)[self.triangles]
        x0, y0 = q[:, 0, 0], q[:, 0, 1]
        x1, y1 = q[:, 1, 0], q[:, 1, 1]
        x2, y2 = q[:, 2, 0], q[:, 2, 1]
        det = (y2 - y0) * (x1 - x0) - (y1 - y0) * (x2 - x0)
        if np.any(np.abs(det) < 1e-15):
            raise ValueError("pressure forcing encountered degenerate triangles")
        self.ax1 = ((y2 - y0) / det).astype(np.float32)
        self.ax2 = (-(y1 - y0) / det).astype(np.float32)
        self.ay1 = (-(x2 - x0) / det).astype(np.float32)
        self.ay2 = ((x1 - x0) / det).astype(np.float32)

    def __call__(self, domain) -> None:
        p = self.pressure.at(domain.get_time())
        p_tri = p[self.triangles]
        dp1 = p_tri[:, 1] - p_tri[:, 0]
        dp2 = p_tri[:, 2] - p_tri[:, 0]
        grad_x = self.ax1 * dp1 + self.ax2 * dp2
        grad_y = self.ay1 * dp1 + self.ay2 * dp2
        height = (domain.quantities["stage"].centroid_values
                  - domain.quantities["elevation"].centroid_values)
        domain.quantities["xmomentum"].explicit_update -= (
            height * grad_x * self.inv_rho_w)
        domain.quantities["ymomentum"].explicit_update -= (
            height * grad_y * self.inv_rho_w)


def run_anuga_scenario(
    mesh: MeshData,
    output_time_s: np.ndarray,
    forcing: Mapping[str, np.ndarray],
    output_dir: str | Path,
    scenario_id: str,
    initial_stage_m: Optional[np.ndarray] = None,
    minimum_storable_depth_m: float = 0.01,
    open_boundary_segments: int = 16,
    omp_num_threads: Optional[int] = None,
) -> Path:
    """Run one ANUGA scenario and return its SWW file.

    Expected atmospheric keys are ``wind_u10``, ``wind_v10``, and
    ``pressure_pa``. Tide may be dense ``tide_m [T,N]`` or sparse as
    ``tide_boundary_m [T,Nb]`` plus ``tide_boundary_node_index [Nb]``.
    Atmospheric forcing objects are constructed with ANUGA's named arguments
    and explicitly appended to ``domain.forcing_terms``.
    """

    try:
        import anuga
    except ImportError as exc:
        raise RuntimeError("Install ANUGA and netCDF4 in a dedicated environment first") from exc

    mesh.validate()
    times = np.asarray(output_time_s, float)
    if times[0] != 0.0 or np.any(np.diff(times) <= 0):
        raise ValueError("output_time_s must start at zero and increase strictly")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    boundary, ref_to_pair = triangle_boundary_topology(
        mesh.triangles, mesh.open_boundary_edges)
    # Split a long ocean boundary into a small number of spatial tide segments.
    # This is much closer to ADCIRC than prescribing one domain-wide mean tide.
    open_refs = list(ref_to_pair)
    if open_refs:
        mids = np.asarray([mesh.xy_m[list(ref_to_pair[r])].mean(axis=0) for r in open_refs])
        centered = mids - mids.mean(axis=0, keepdims=True)
        axis = np.linalg.svd(centered, full_matrices=False)[2][0] if len(mids) > 1 else np.array([1., 0.])
        rank = np.argsort(np.argsort(centered @ axis))
        nseg = min(max(1, int(open_boundary_segments)), len(open_refs))
        segment_id = np.minimum(rank * nseg // len(open_refs), nseg - 1)
        for ref, sid in zip(open_refs, segment_id):
            boundary[ref] = f"open_{int(sid):03d}"
    domain = anuga.Domain(mesh.xy_m, mesh.triangles, boundary)
    if omp_num_threads is not None:
        if int(omp_num_threads) < 1:
            raise ValueError("omp_num_threads must be at least 1")
        domain.set_multiprocessor_mode(1)
        domain.set_omp_num_threads(int(omp_num_threads))
    print(
        f"ANUGA domain: triangles={len(mesh.triangles):,}, "
        f"omp_threads={domain.omp_num_threads}, "
        f"multiprocessor_mode={domain.get_multiprocessor_mode()}",
        flush=True,
    )
    domain.set_name(scenario_id)
    domain.set_datadir(str(output_dir))
    domain.set_store_vertices_smoothly(True)
    domain.set_minimum_storable_height(minimum_storable_depth_m)
    domain.set_quantities_to_be_stored({
        "elevation": 1, "stage": 2, "xmomentum": 2, "ymomentum": 2,
    })

    # The mesh node order is identical to ANUGA's input node order. Assign
    # unique-vertex arrays directly instead of building/querying three KDTrees.
    domain.set_quantity(
        "elevation", numeric=np.asarray(mesh.bed_elevation_m, float),
        location="unique vertices")
    domain.set_quantity(
        "friction", numeric=np.asarray(mesh.manning_n, float),
        location="unique vertices")
    if initial_stage_m is None:
        initial_stage_m = np.maximum(mesh.bed_elevation_m, 0.0)
    initial_stage_m = np.asarray(initial_stage_m, float)
    domain.set_quantity(
        "stage", numeric=initial_stage_m, location="unique vertices")

    boundary_objects = {"land": anuga.Reflective_boundary(domain)}
    open_names = sorted({name for name in boundary.values() if name.startswith("open_")})
    if "tide_boundary_m" in forcing:
        tide_nodes = np.asarray(forcing["tide_boundary_node_index"], dtype=np.int64)
        tide_values = np.asarray(forcing["tide_boundary_m"], dtype=float)
        if tide_values.shape != (len(times), len(tide_nodes)):
            raise ValueError("tide_boundary_m must be [T,Nb]")
        tide_column = np.full(len(mesh.xy_m), -1, dtype=np.int64)
        tide_column[tide_nodes] = np.arange(len(tide_nodes))
    elif "tide_m" not in forcing:
        # A zero ocean stage is useful for a fast meteorological smoke test.
        tide_values = None
    for name in open_names:
        refs = [ref for ref, tag in boundary.items() if tag == name]
        nodes = np.unique([node for ref in refs for node in ref_to_pair[ref]])
        if "tide_boundary_m" in forcing:
            columns = tide_column[nodes]
            if np.any(columns < 0):
                raise ValueError(f"sparse tide is missing nodes in boundary segment {name}")
            segment_tide = np.mean(tide_values[:, columns], axis=1)
        elif "tide_m" in forcing:
            segment_tide = np.mean(np.asarray(forcing["tide_m"])[:, nodes], axis=1)
        else:
            segment_tide = np.zeros(len(times), dtype=np.float32)

        def stage_at_time(t, series=segment_tide):
            return float(np.interp(t, times, series))

        # Weakly reflecting open boundary when supported by installed ANUGA;
        # otherwise use a time-varying zero-momentum boundary.
        if hasattr(anuga, "Flather_external_stage_zero_velocity_boundary"):
            boundary_objects[name] = anuga.Flather_external_stage_zero_velocity_boundary(
                domain, function=stage_at_time)
        else:
            boundary_objects[name] = anuga.Time_boundary(
                domain=domain, function=lambda t, f=stage_at_time: [f(t), 0.0, 0.0])
    domain.set_boundary(boundary_objects)

    domain.forcing_terms.append(
        VectorizedWindStress(
            times, mesh.triangles, forcing["wind_u10"], forcing["wind_v10"]))
    domain.forcing_terms.append(
        VectorizedBarometricPressure(
            times, mesh.xy_m, mesh.triangles, forcing["pressure_pa"]))

    # yieldstep controls storage frequency, not the internal CFL timestep.
    yieldstep = float(np.min(np.diff(times)))
    for _ in domain.evolve(yieldstep=yieldstep, finaltime=float(times[-1])):
        pass
    return output_dir / f"{scenario_id}.sww"


def read_sww_state(sww_path: str | Path, target_xy_m: np.ndarray,
                   dry_tolerance_m: float = 0.02) -> dict[str, np.ndarray]:
    """Read SWW and return wet-masked ``[eta,u,v]`` in target-node order.

    ANUGA sets stage equal to bed elevation at dry land nodes.  Exporting that
    value as water level makes dry topography appear as several metres of
    water.  The model-safe state therefore uses zero at dry nodes and retains
    ``wet_mask`` plus ``eta_reference_m`` for interpretation.
    """

    try:
        from netCDF4 import Dataset
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("read_sww_state requires netCDF4 and scipy") from exc

    with Dataset(str(sww_path), "r") as ds:
        xy_local = np.column_stack((np.asarray(ds.variables["x"][:]), np.asarray(ds.variables["y"][:])))
        origin = np.array([float(getattr(ds, "xllcorner", 0.0)),
                           float(getattr(ds, "yllcorner", 0.0))])
        target = np.asarray(target_xy_m, float)
        # Depending on how Domain was created, SWW may use local coordinates.
        candidates = (xy_local, xy_local + origin)
        xy = min(candidates, key=lambda q: np.mean(cKDTree(q).query(target)[0]))
        query_dist, order = cKDTree(xy).query(target)
        scale = max(np.ptp(target_xy_m[:, 0]), np.ptp(target_xy_m[:, 1]), 1.0)
        if np.max(query_dist) > 1e-7 * scale:
            raise ValueError("SWW mesh does not match the requested target mesh")
        triangles = np.asarray(ds.variables["volumes"][:], dtype=np.int64)

        def to_sww_nodes(a: np.ndarray) -> np.ndarray:
            """Accept SWW vertex or centroid fields and return [T,N_sww]."""
            a = np.asarray(a, dtype=np.float32)
            if a.ndim == 1:
                a = a[None, :]
            if a.shape[-1] == len(xy):
                return a
            if a.shape[-1] != len(triangles):
                raise ValueError(f"Unsupported SWW field width {a.shape[-1]}")
            out = np.zeros((len(a), len(xy)), dtype=np.float32)
            counts = np.bincount(triangles.ravel(), minlength=len(xy)).astype(np.float32)
            for local_vertex in range(3):
                np.add.at(out, (slice(None), triangles[:, local_vertex]), a)
            return out / np.maximum(counts[None, :], 1.0)

        stage = to_sww_nodes(ds.variables["stage"][:])[:, order]
        bed_node = to_sww_nodes(ds.variables["elevation"][:])
        bed = bed_node[-1, order]
        xmom = to_sww_nodes(ds.variables["xmomentum"][:])[:, order]
        ymom = to_sww_nodes(ds.variables["ymomentum"][:])[:, order]
        time_s = np.asarray(ds.variables["time"][:], dtype=np.float64)
    depth = np.maximum(stage - bed[None, :], 0.0)
    wet = depth > dry_tolerance_m
    safe_depth = np.maximum(depth, dry_tolerance_m)
    u = np.where(wet, xmom / safe_depth, 0.0)
    v = np.where(wet, ymom / safe_depth, 0.0)
    eta = np.where(wet, stage, 0.0)
    return {
        "time_s": time_s,
        "state": np.stack((eta, u, v), axis=-1).astype(np.float32),
        "wet_mask": wet,
        "eta_reference_m": stage[0].astype(np.float32),
    }


def save_hopnet_scenario(
    output_path: str | Path,
    mesh: MeshData,
    solution: Mapping[str, np.ndarray],
    forcing: Mapping[str, np.ndarray],
    compress: bool = True,
) -> Path:
    """Save a compact, model-ready scenario without pre-building windows."""

    graph = mesh_graph(mesh)
    solution_time = np.asarray(solution["time_s"], float)
    forcing_time = np.asarray(forcing.get("time_s", solution_time), float)

    def align(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if len(values) == len(solution_time) and np.allclose(forcing_time, solution_time):
            return values
        hi = np.searchsorted(forcing_time, solution_time, side="right")
        hi = np.clip(hi, 1, len(forcing_time) - 1)
        lo = hi - 1
        alpha = ((solution_time - forcing_time[lo])
                 / (forcing_time[hi] - forcing_time[lo])).astype(np.float32)
        alpha = np.clip(alpha, 0.0, 1.0).reshape(
            (len(solution_time),) + (1,) * (values.ndim - 1))
        return (np.float32(1.0) - alpha) * values[lo] + alpha * values[hi]

    if "tide_m" in forcing:
        tide_for_model = forcing["tide_m"]
    elif "tide_boundary_m" in forcing:
        # A boundary-condition feature: zero at interior nodes and prescribed
        # stage at open-boundary nodes. The companion graph/static data lets a
        # surrogate distinguish this from an interior zero tide.
        tide_for_model = np.zeros(
            (len(forcing_time), len(mesh.xy_m)), dtype=np.float32)
        tide_nodes = np.asarray(forcing["tide_boundary_node_index"], dtype=np.int64)
        tide_for_model[:, tide_nodes] = np.asarray(forcing["tide_boundary_m"], dtype=np.float32)
    else:
        tide_for_model = np.zeros(
            (len(forcing_time), len(mesh.xy_m)), dtype=np.float32)
    pressure_head = forcing.get("pressure_head_m")
    if pressure_head is None:
        pressure_head = np.asarray(forcing["pressure_pa"], dtype=np.float32) / np.float32(
            1025.0 * 9.81)
    forcing_tensor = np.stack((
        align(forcing["wind_u10"]),
        align(forcing["wind_v10"]),
        align(pressure_head),
        align(tide_for_model),
    ), axis=-1).astype(np.float32)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_npz = np.savez_compressed if compress else np.savez
    save_npz(
        path,
        time_s=np.asarray(solution["time_s"]),
        datetime_utc=np.asarray(
            solution.get("datetime_utc", np.empty(0, dtype="datetime64[s]"))),
        iso_time=np.asarray(
            solution.get("iso_time", np.empty(0, dtype="U1"))),
        pos=mesh.xy_m.astype(np.float32),
        lon=np.asarray(
            mesh.lon_deg if mesh.lon_deg is not None else np.full(len(mesh.xy_m), np.nan),
            dtype=np.float64),
        lat=np.asarray(
            mesh.lat_deg if mesh.lat_deg is not None else np.full(len(mesh.xy_m), np.nan),
            dtype=np.float64),
        triangles=mesh.triangles.astype(np.int64),
        edge_index=graph["edge_index"],
        edge_attr=graph["edge_attr"],
        static_features=np.stack((-mesh.bed_elevation_m, mesh.manning_n), axis=-1).astype(np.float32),
        state=np.asarray(solution["state"], dtype=np.float32),
        state_names=np.asarray(["water_level_m", "velocity_u_ms", "velocity_v_ms"]),
        eta_reference_m=np.asarray(solution["eta_reference_m"], dtype=np.float32),
        dry_node_state_fill=np.asarray(0.0, dtype=np.float32),
        pressure_units=np.asarray("Pa"),
        pressure_source_formula=np.asarray("-(water_depth/rho_water)*gradient(pressure_pa)"),
        forcing=forcing_tensor,
        forcing_names=np.asarray(["wind_u10", "wind_v10", "pressure_head_m", "tide_m"]),
        tide_boundary_node_index=np.asarray(
            forcing.get("tide_boundary_node_index", np.empty(0)), dtype=np.int64),
        wet_mask=np.asarray(solution["wet_mask"], dtype=bool),
    )
    return path


def make_windows(state: np.ndarray, forcing: np.ndarray, history: int,
                 horizon: int, stride: int = 1) -> dict[str, np.ndarray]:
    """Construct leak-free direct-forecast windows used by the current trainer."""

    state = np.asarray(state)
    forcing = np.asarray(forcing)
    if state.shape[:2] != forcing.shape[:2]:
        raise ValueError("state and forcing must share [T,N]")
    starts = np.arange(0, len(state) - history - horizon + 1, stride)
    return {
        "history_state": np.stack([state[s:s + history] for s in starts]),
        "future_forcing": np.stack([forcing[s + history:s + history + horizon] for s in starts]),
        "target_state": np.stack([state[s + history:s + history + horizon] for s in starts]),
        "window_start": starts,
    }
