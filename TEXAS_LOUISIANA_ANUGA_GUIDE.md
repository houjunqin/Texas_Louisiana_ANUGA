# Texas–Louisiana regional ANUGA mesh

## Delivered mesh

The mesh was cropped from the original `LA_ERDC_v05j_chk.grd` `fort.14` without
retriangulation. Only the largest connected component was retained.

```text
Bounds:       -98.0021428° to -91.9600180° longitude
               25.2402864° to  30.7999230° latitude
Nodes:          741,254
Triangles:    1,467,685
Components:           1
```

All 1,467,685 triangles have positive orientation after projection and none is
degenerate.

## Boundary classification

```text
Ocean/lateral open edges:    441
Northern river edges:        116
Reflective edges:         14,324
Total exterior edges:     14,881
```

`open_boundary_edges` contains wet clipped edges nearest the southern, western,
or eastern regional limits. These can receive tide/surge boundary conditions.

`river_boundary_edges` contains wet clipped edges nearest the northern limit.
They are intentionally separate. Supply discharge/stage hydrographs for
compound flooding. If the current ANUGA runner is used without a river
extension, these edges are treated as reflective.

## Static values

Bathymetry comes directly from `fort.14` and is stored in both conventions:

```text
adcirc_depth_m:  positive downward
bed_elevation_m: positive upward = -adcirc_depth_m
```

Manning values were assigned as follows:

```text
Exact input1.dat matches:       13,423 nodes
Missing water nodes:           318,342 nodes → n = 0.02
Missing land nodes:            409,489 nodes → nearest input1.dat land value
```

For land interpolation, the median source distance is 0.209 km, the 95th
percentile is 0.553 km, and the maximum is 7.953 km. This is appropriate for a
baseline generator but should be replaced by the original spatial roughness
field if available.

## First ANUGA run

The Harvey track and mesh use the same local spherical-LAEA projection.

```bash
python build_anuga_dataset.py \
  --mesh texas_louisiana_mesh.npz \
  --track harvey_track_gulf.npz \
  --output harvey_anuga_dataset.npz \
  --scenario-id harvey_tx_la \
  --duration-hours 168 \
  --output-minutes 30 \
  --open-boundary-segments 64
```

The 64 segments preserve more along-boundary FES variation than the default
16, while avoiding one separate ANUGA boundary object for every edge.

For the first stability test, use 6–12 hours rather than the full event. The
output interval controls storage frequency, but ANUGA still uses its own
CFL-limited internal timesteps.

## FES2022b ocean-tide boundary

Use the FES2022b **ocean-tide elevation** files for the hydrodynamic boundary.
Do not add `load_tide`/`radial` values: those describe elastic crustal loading,
not the water level to prescribe against a fixed ANUGA bathymetry.

Install the official predictor in the ANUGA environment:

```bash
conda install -c conda-forge pyfes
```

Generate tides only at the unique nodes in `open_boundary_edges`. The Harvey
track begins at `2017-08-21T18:00:00Z`, and all prediction times are UTC:

```bash
python fes2022_boundary.py \
  --mesh outputs/texas_louisiana_crop/texas_louisiana_mesh.npz \
  --fes-dir /path/to/fes2022b/ocean_tide_extrapolated \
  --track outputs/texas_louisiana_crop/harvey_track_gulf.npz \
  --duration-hours 168 \
  --interval-minutes 30 \
  --atlas-units cm \
  --datum-offset-m 0.0 \
  --output outputs/texas_louisiana_crop/harvey_fes2022b_boundary.npz
```

The script discovers standard `*_tide.nc` files and writes a PyFES YAML file.
Alternatively, pass an existing official configuration with `--config`.
Use the extrapolated ocean grid when a boundary node lies in a coastal mask.

Then pass the sparse tide file to the existing runner:

```bash
python build_anuga_dataset.py \
  --mesh outputs/texas_louisiana_crop/texas_louisiana_mesh.npz \
  --track outputs/texas_louisiana_crop/harvey_track_gulf.npz \
  --tide-npz outputs/texas_louisiana_crop/harvey_fes2022b_boundary.npz \
  --output outputs/harvey_anuga_dataset.npz \
  --scenario-id harvey_tx_la \
  --duration-hours 168 \
  --output-minutes 30
```

`datum_offset_m` is not a tuning parameter. It must transform the FES
zero-mean water level to the vertical datum used by `fort.14`. Leave it at zero
only if the ADCIRC depths/stages are referenced consistently with mean sea
level. Determine a nonzero offset from the model datum metadata or colocated
gauge/ADCIRC harmonic boundary data.

For production, run a tide-only spin-up (typically several days) and restart
the storm simulation from the spun-up state. Starting the mesh at a spatially
uniform stage while immediately imposing a spatially varying tide creates an
artificial gravity-wave transient.

## Important fidelity requirements

Before interpreting total water levels, add:

1. ocean tidal boundary elevations using the same vertical datum as `fort.14`;
2. tidal or hydrodynamic spin-up;
3. river hydrographs for the 116 northern wet boundary edges;
4. spatial rainfall and infiltration/loss assumptions for Harvey compound
   flooding;
5. paired ADCIRC or gauge results for calibration.

The current mesh is full regional resolution. It is suitable for a high-fidelity
regional campaign but is not a lightweight screening mesh. Use MPI/OpenMP or
make a separately validated coarsened mesh before generating a large ensemble.
