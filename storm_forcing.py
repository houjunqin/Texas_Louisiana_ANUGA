"""Fast parametric tropical-cyclone forcing on an unstructured coastal mesh.

The functions in this module generate *inputs* to a hydrodynamic solver.  They
do not attempt to replace ADCIRC or SWAN.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


RHO_AIR = 1.225
RHO_WATER = 1025.0
GRAVITY = 9.81


@dataclass(frozen=True)
class StormTrack:
    """Storm track sampled in the same projected CRS as ``node_xy``.

    All arrays are one-dimensional and ordered by increasing ``time_s``.
    Pressure is in hPa, radius of maximum wind in km, translation velocity in
    m/s, and heading is clockwise from geographic/projected north.
    """

    time_s: np.ndarray
    center_x_m: np.ndarray
    center_y_m: np.ndarray
    central_pressure_hpa: np.ndarray
    outer_pressure_hpa: np.ndarray
    rmw_km: np.ndarray
    translation_speed_ms: np.ndarray
    heading_deg: np.ndarray
    holland_b: Optional[np.ndarray] = None

    def validate(self) -> None:
        arrays = [
            self.time_s, self.center_x_m, self.center_y_m,
            self.central_pressure_hpa, self.outer_pressure_hpa, self.rmw_km,
            self.translation_speed_ms, self.heading_deg,
        ]
        n = len(self.time_s)
        if n < 2 or any(np.asarray(a).shape != (n,) for a in arrays):
            raise ValueError("All StormTrack fields must be 1-D arrays of equal length >= 2")
        if not np.all(np.diff(self.time_s) > 0):
            raise ValueError("StormTrack.time_s must be strictly increasing")
        if self.holland_b is not None and np.asarray(self.holland_b).shape != (n,):
            raise ValueError("holland_b must have the same shape as time_s")


def _interp(track: StormTrack, name: str, output_time_s: np.ndarray) -> np.ndarray:
    return np.interp(output_time_s, track.time_s, np.asarray(getattr(track, name), float))


def generate_holland_forcing(
    node_xy_m: np.ndarray,
    output_time_s: np.ndarray,
    track: StormTrack,
    tide_m: Optional[np.ndarray] = None,
    inflow_angle_deg: float = 20.0,
    translation_fraction: float = 0.55,
    minimum_radius_km: float = 1.0,
    minimal: bool = False,
    time_chunk_size: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """Generate asymmetric Holland-type wind and pressure fields.

    Returns node fields with shape ``[T, N]``.  Wind components are east/x and
    north/y in m/s. Pressure is returned in Pa and as an equivalent water head.
    ``tide_m`` may be ``[T, N]``, ``[T, 1]``, or omitted.
    """

    track.validate()
    xy = np.asarray(node_xy_m, dtype=np.float64)
    times = np.asarray(output_time_s, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or times.ndim != 1:
        raise ValueError("node_xy_m must be [N,2] and output_time_s must be [T]")
    if time_chunk_size is not None and len(times) > int(time_chunk_size):
        chunk_size = max(1, int(time_chunk_size))
        assembled: dict[str, np.ndarray] = {}
        for first in range(0, len(times), chunk_size):
            last = min(first + chunk_size, len(times))
            tide_chunk = None if tide_m is None else np.asarray(tide_m)[first:last]
            part = generate_holland_forcing(
                xy, times[first:last], track, tide_m=tide_chunk,
                inflow_angle_deg=inflow_angle_deg,
                translation_fraction=translation_fraction,
                minimum_radius_km=minimum_radius_km,
                minimal=minimal,
                time_chunk_size=None,
            )
            if not assembled:
                assembled = {
                    key: np.empty((len(times),) + value.shape[1:], dtype=value.dtype)
                    for key, value in part.items()
                }
            for key, value in part.items():
                assembled[key][first:last] = value
        return assembled

    cx = _interp(track, "center_x_m", times)[:, None]
    cy = _interp(track, "center_y_m", times)[:, None]
    pc = _interp(track, "central_pressure_hpa", times)[:, None]
    po = _interp(track, "outer_pressure_hpa", times)[:, None]
    rmw = _interp(track, "rmw_km", times)[:, None]
    vt = _interp(track, "translation_speed_ms", times)[:, None]
    heading = np.deg2rad(_interp(track, "heading_deg", times))[:, None]
    if track.holland_b is None:
        # Conservative bounded default; calibration should replace this.
        b = np.full_like(pc, 1.5)
    else:
        b = np.interp(times, track.time_s, track.holland_b)[:, None]
    b = np.clip(b, 0.8, 2.5)

    dx = xy[None, :, 0] - cx
    dy = xy[None, :, 1] - cy
    radius_km = np.maximum(np.hypot(dx, dy) / 1000.0, minimum_radius_km)
    ratio_b = np.power(np.maximum(rmw, 0.1) / radius_km, b)
    exp_term = np.exp(-ratio_b)
    delta_p_pa = np.maximum(po - pc, 0.0) * 100.0

    pressure_pa = (pc + (po - pc) * exp_term) * 100.0
    gradient_speed = np.sqrt(np.maximum(b * delta_p_pa / RHO_AIR * ratio_b * exp_term, 0.0))

    radial_angle = np.arctan2(dy, dx)
    wind_angle = radial_angle + math.pi / 2.0 - math.radians(inflow_angle_deg)
    u_rot = gradient_speed * np.cos(wind_angle)
    v_rot = gradient_speed * np.sin(wind_angle)
    # A heading clockwise from north has components (sin heading, cos heading).
    u_trans = translation_fraction * vt * np.sin(heading)
    v_trans = translation_fraction * vt * np.cos(heading)
    wind_u10 = u_rot + u_trans
    wind_v10 = v_rot + v_trans

    result = {
        "wind_u10": wind_u10.astype(np.float32),
        "wind_v10": wind_v10.astype(np.float32),
        "pressure_pa": pressure_pa.astype(np.float32),
    }
    if tide_m is not None:
        result["tide_m"] = np.broadcast_to(
            np.asarray(tide_m, dtype=np.float32), wind_u10.shape).copy()
    if not minimal:
        speed = np.hypot(wind_u10, wind_v10)
        drag = np.minimum(0.001 * (0.75 + 0.067 * speed), 0.003)
        tau_x = drag * RHO_AIR * speed * wind_u10
        tau_y = drag * RHO_AIR * speed * wind_v10
        result.update({
            "wind_speed": speed.astype(np.float32),
            # Ordinary mathematical degrees: 0=east/x, 90=north/y.
            "wind_direction_deg": np.mod(
                np.rad2deg(np.arctan2(wind_v10, wind_u10)), 360.0).astype(np.float32),
            "tau_x_pa": tau_x.astype(np.float32),
            "tau_y_pa": tau_y.astype(np.float32),
            "pressure_head_m": (
                pressure_pa / (RHO_WATER * GRAVITY)).astype(np.float32),
        })
        if tide_m is None:
            result["tide_m"] = np.zeros_like(result["wind_u10"])
    return result
