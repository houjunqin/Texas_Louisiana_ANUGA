# Hindcasting Hurricane Strom Surges in Texas–Louisiana region via ANUGA

## Background

The project employs ANUGA to hindcast hurricane storm surges in the Texas–Louisiana region, generating high-quality datasets for graph-based spatiotemporal deep learning models. The core methodology integrates multiple structural and environmental data inputs, including the mesh and nodefile ([ADCIRC](https://adcirc.org/) fort.14 fort.13), hurricane tracks([IBTrACS](https://ncics.org/ibtracs/index.php?name=Grid-42745)), and tidal data ([AVISO FES2022b](https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/global-tide-fes/release-fes22.html)). To ensure physical accuracy, the model incorporates time-varying MRMS/NEXRAD rainfall, river discharge hydrographs, watershed inflow locations, and infiltration-loss parameters, alongside critical hydraulic infrastructure such as levees, barriers, roadways, channels, and culverts. The primary objective is to generate an extensive repository of hurricane storm surge datasets within a fixed mesh domain. By maintaining a constant geographical boundary and an identical node count across all simulations, thereby providing robust datasets for our [deep learning models](https://github.com/houjunqin/DeepStormSurge) training.

<picture>
<img src="./assets/images/Harvey_ANUGA.png" alt="Harvey" width=33% height=33%>
<picture>
<img src="./assets/images/hurricanes.png" alt="hurricanes" width=45% height=33%>
</picture>
</picture>

## 1. Create mesh file: texas_louisiana_mesh.npz
### Delivered mesh

The mesh was cropped from the original `fort.14` without
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

### Boundary classification

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

### Static values

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

For a production compound-flood simulation, use spatially variable Manning’s n from `fort.13`, NLCD land cover, or another roughness dataset; using one value everywhere may underestimate resistance over wetlands and developed land. 
```bash
python build_texas_louisiana_crop.py \
  --fort14 input/fort.14 \
  --fort13 input/fort.13 \
  --output-dir outputs
```
Optional: specify the crop bounds explicitly

The default bounds are:
```text
longitude: -98.0021427904 to -91.960018
latitude:   25.2402864261 to 30.799923
```
Equivalent explicit command:
```bash
python build_texas_louisiana_crop.py \
  --fort14 input/fort.14 \
  --fort13 input/fort.13 \
  --output-dir outputs \
  --lon-min -98.0021427904 \
  --lon-max -91.960018 \
  --lat-min 25.2402864261 \
  --lat-max 30.799923 \
  --wet-boundary-threshold-m 0.5
```


## 2. Create a Harvey track file: harvey_track_gulf.npz
### Download the official IBTrACS storm track

The included downloader uses NOAA's official North Atlantic IBTrACS v04r01
CSV. It selects storms with at least one point inside a buffered
Texas-Louisiana box, but preserves each selected storm's complete lifecycle
track. This is important for estimating storm translation and beginning the
meteorological forcing before landfall.

Create a catalog of regional storms since 1980:

```bash
python download_ibtracs_texas_louisiana.py \
  --output-dir outputs/ibtracs_texas_louisiana \
  --start-year 1980 \
  --buffer-deg 2.5
```
Outputs:
```bash
outputs/ibtracs_texas_louisiana/
├── ibtracs.NA.list.v04r01.csv
├── texas_louisiana_storm_catalog.csv
├── selection.json
└── tracks/
    ├── SID_STORM.csv
    └── ...
```
The default regional bounding box is:
```bash
longitude: -98.1 to -91.9
latitude:   25.2 to 30.8
buffer:      2.5 degrees
```
A storm is selected if any center position enters the buffered region, but its complete lifecycle track is retained.
### Create a Harvey track file for ANUGA
Download and convert the 2017 Harvey track directly to the ANUGA convention:

```bash
python download_ibtracs_texas_louisiana.py \
  --output-dir outputs/ibtracs_texas_louisiana \
  --name HARVEY \
  --season 2017 \
  --mesh outputs/texas_louisiana_mesh.npz \
  --anuga-npz outputs/harvey_track_gulf.npz \
  --event-buffer-deg 2.5 \
  --pre-impact-hours 48 \
  --post-impact-hours 72 \
  --output-interval-minutes 30
```
The validated Harvey identification is:
```bash
IBTrACS SID: 2017228N14314
ATCF ID:     AL092017
Name:        HARVEY
```
The resulting ANUGA file contains:
```bash
time_s
iso_time
center_lon_deg
center_lat_deg
center_x_m
center_y_m
central_pressure_hpa
outer_pressure_hpa
max_wind_ms
rmw_km
roci_km
translation_speed_ms
heading_deg
holland_b
r34_quadrant_km
r50_quadrant_km
r64_quadrant_km
```
The automatic timing logic is:

- Find the first and last IBTrACS records inside the buffered region:
 ```bash
longitude: −98.1 to −91.9
latitude:   25.2 to 30.8
buffer:      2.5 degrees
```
- Start 48 hours before the first nearby record.
- End 72 hours after the last nearby record.
- Clip these times to the available IBTrACS track.
- Snap start and end to actual IBTrACS records.
- Calculate duration_hours automatically.

You can change the approach padding:
```bash
--pre-impact-hours 72 \
--post-impact-hours 96
```
- The generated NPZ contains fixed-width Unicode `iso_time` values and therefore
- loads safely with `allow_pickle=False`. Missing IBTrACS outer pressure, RMW, or
- ROCI values are temporally interpolated where possible; otherwise the explicit
- fallbacks are 1013.25 hPa, 45 km, and 300 km. Review these assumptions before
- production surge simulations.

Generated parameter files

The command produces:
```bash
outputs/harvey_track_gulf.npz
outputs/harvey_track_gulf_event_parameters.json
outputs/harvey_track_gulf_event_parameters.sh
```
The JSON contains:
```bash
{
  "event": {
    "ibtracs_sid": "2017228N14314",
    "atcf_id": "AL092017",
    "name": "HARVEY",
    "season": 2017
  },
  "timing": {
    "track_start_iso": "...",
    "track_end_iso": "...",
    "duration_hours": 0,
    "output_interval_minutes": 30,
    "output_time_count": 0
  },
  "approved_usgs_sites": [
    "08211500",
    "08162500",
    "08116650",
    "08072000",
    "08041000",
    "08030500"
  ]
}
```
## 3. Load the parameters into the terminal

Run:
```bash
source outputs/harvey_track_gulf_event_parameters.sh
```
Inspect them:
```bash
echo "$TRACK_START_ISO"
echo "$TRACK_END_ISO"
echo "$DURATION_HOURS"
echo "$OUTPUT_INTERVAL_MINUTES"
echo "$APPROVED_USGS_SITES"
echo "$TRACK_NPZ"
```
Available variables include:
```bash
IBTRACS_SID
ATCF_ID
STORM_NAME
SEASON
TRACK_START_ISO
TRACK_END_ISO
DURATION_HOURS
OUTPUT_INTERVAL_MINUTES
APPROVED_USGS_SITES
TRACK_NPZ
```
## 4. Create a Harvey compound forcing file: harvey_compound_forcing.npz
### Download compound-flood observations

- NOAA/NWS Stage IV hourly multisensor precipitation;
- USGS river discharge hydrographs;
- a transparent constant-capacity rainfall-loss model;
- USACE National Levee Database downloads for subsequent mesh rebuilding;
- optional ADCIRC/SWAN-style wave-force fields;
- corrected atmospheric-pressure units and pressure-gradient sign.

NOAA's historical `ncep_stage_iv_source_files_YYYYMMDD.tar` products are published as daily source archives. The reader now discovers GRIB/GRIB2 files by content/name, supports `.gz`, and reads each message's accumulation period.
Check `source_accumulation_hours_min/max` in the audit file. If both equal 24, the forcing is daily mean rainfall and true hourly Stage IV/MRMS data should be obtained before a final Harvey calibration.

It does **not** automatically burn levees, roads, bridges, or culverts into an already-created mesh. Those structures alter topology/elevation and require a mesh rebuild and engineering review.

Download additional data using derived dates:

```bash

python download_harvey_additional_data.py \
  --mesh outputs/texas_louisiana_mesh.npz \
  --output-dir outputs/harvey_additional_raw \
  --start "$TRACK_START_ISO" \
  --end "$TRACK_END_ISO" \
  --usgs-sites "$APPROVED_USGS_SITES"
```

The site list is only a starting candidate list. Verify every station and avoid using two stations that represent the same river inflow. If `--usgs-sites` is omitted, the downloader retrieves all stream gauges in the mesh bounding box; the preparation step can then select a reviewed subset.


### Prepare mesh-aligned forcing

Install pygrib:
```bash
conda install -c conda-forge pygrib eccodes
```
```bash
Prepare compound forcing
python prepare_harvey_forcing.py \
  --mesh outputs/texas_louisiana_mesh.npz \
  --track "$TRACK_NPZ" \
  --download-dir outputs/harvey_additional_raw \
  --output outputs/harvey_compound_forcing.npz \
  --start-iso "$TRACK_START_ISO" \
  --duration-hours "$DURATION_HOURS" \
  --interval-minutes "$OUTPUT_INTERVAL_MINUTES" \
  --approved-usgs-sites "$APPROVED_USGS_SITES" \
  --inlet-triangles 16
```

- The preparation program writes an adjacent `*.audit.json`. Inspect its gage-to-boundary mapping before a production simulation. Automatic nearest-edge mapping is a reproducible initial guess, not a hydrologic guarantee.

- `--start-iso` is optional. It is useful for legacy `harvey_track_gulf.npz` files whose `iso_time` array was saved with object dtype. The revised loader can read those trusted legacy files automatically, while new files should use fixed-width Unicode timestamps.

- At 741,254 nodes, 282 hours at a 5-minute interval produces 3,385 time levels.
- Each dense float32 `[T,N]` field is about 9.35 GiB, so the preparation program prints an explicit memory warning. 
- A 30-minute interval is strongly preferred
for initial testing and surrogate training.

Optional spatial soil and land-cover files use these conventions:

```text
soil_parameters.npz:       ksat_mps [N]
landcover_parameters.npz:  impervious_fraction [N]
```

Pass them with:

```bash
  --soil-npz soil_parameters.npz \
  --landcover-npz landcover_parameters.npz
```

Without them, the script clearly records the fallback assumptions:

```text
ksat_mps = 2.0e-6
impervious_fraction = 0.15
```

The effective rainfall model is

\[
R_{eff}=f_{imp}R+(1-f_{imp})\max(R-K_s,0).
\]

This is appropriate for a first compound-flood experiment, but it is not a
calibrated Green-Ampt or watershed model.

## 5. Create a tide-only forcing file: harvey_fes2022b_boundary.npz
### FES2022b ocean-tide boundary

Use the FES2022b **ocean-tide elevation** files for the hydrodynamic boundary.
Do not add `load_tide`/`radial` values: those describe elastic crustal loading,
not the water level to prescribe against a fixed ANUGA bathymetry.

Install the official predictor in the ANUGA environment:

```bash
conda install -c conda-forge pyfes
```

Generate tides only at the unique nodes in `open_boundary_edges`. The Harvey track begins at `2017-08-21T18:00:00Z`, and all prediction times are UTC:

```bash
python fes2022_boundary.py \
  --mesh outputs/texas_louisiana_mesh.npz \
  --track "$TRACK_NPZ" \
  --fes-dir /path/to/FES2022b/ocean_tide \
  --duration-hours "$DURATION_HOURS" \
  --interval-minutes "$OUTPUT_INTERVAL_MINUTES" \
  --output outputs/harvey_fes2022b_boundary.npz
```

## 6. Create a Harvey ANUGA dataset: harvey_anuga_dataset.npz

### Environment

Use the same environment that contains ANUGA and install the readers:

```bash
conda create -n anuga-harvey \
  -c conda-forge \
  --strict-channel-priority \
  python=3.11 \
  numpy=1.26.4 \
  scipy=1.13.1 \
  netcdf4 \
  pandas \
  matplotlib \
  pyproj \
  shapely \
  meshpy \
  cython \
  pybind11 \
  openmpi \
  mpi4py \
  pip

conda activate anuga-harvey

python -m pip install --no-cache-dir anuga
```

## First ANUGA run

The Harvey track and mesh use the same local spherical-LAEA projection.

```bash
python build_anuga_dataset.py \
  --mesh outputs/texas_louisiana_mesh.npz \
  --track outputs/harvey_track_gulf.npz \
  --output outputs/harvey_anuga_dataset.npz \
  --scenario-id harvey_tx_la \
  --duration-hours 282 \
  --output-minutes 30 \
  --open-boundary-segments 64
```

The 64 segments preserve more along-boundary FES variation than the default
16, while avoiding one separate ANUGA boundary object for every edge.

For the first stability test, use 6–12 hours rather than the full event. The
output interval controls storage frequency, but ANUGA still uses its own
CFL-limited internal timesteps.

The script discovers standard `*_tide.nc` files and writes a PyFES YAML file.
Alternatively, pass an existing official configuration with `--config`.
Use the extrapolated ocean grid when a boundary node lies in a coastal mask.

Then pass the sparse tide file to the existing runner:

```bash
python build_anuga_dataset.py \
  --mesh outputs/texas_louisiana_mesh.npz \
  --track outputs/harvey_track_gulf.npz \
  --tide-npz outputs/harvey_fes2022b_boundary.npz \
  --output outputs/harvey_anuga_dataset.npz \
  --scenario-id harvey_tx_la \
  --duration-hours 282 \
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

### Generate the ANUGA/HOPNet dataset
Run ANUGA with matching times
```bash
python build_anuga_dataset.py \
  --mesh outputs/texas_louisiana_mesh.npz \
  --track "$TRACK_NPZ" \
  --tide-npz outputs/harvey_fes2022b_boundary.npz \
  --additional-forcing-npz outputs/harvey_compound_forcing.npz \
  --output outputs/harvey_compound_dataset.npz \
  --scenario-id harvey_compound \
  --duration-hours "$DURATION_HOURS" \
  --output-minutes "$OUTPUT_INTERVAL_MINUTES" \
  --open-boundary-segments 64 \
  --omp-threads 8 \
  --forcing-chunk-times 8 \
  --uncompressed-output
```

For a smoke test, create a separate 1-hour additional-forcing file with
`--duration-hours 1`, then run the builder with the same duration. All forcing files must use exactly the same `time_s` grid.

## Output conventions

The dataset contains:

```text
state[T,N,3] = [surface elevation, velocity_u, velocity_v]
forcing[T,N,F]
forcing_names = [wind_u10, wind_v10, pressure_head_m, tide_m,
                 rain_rate_mps, rain_effective_mps, ...]
river_discharge_m3s[T,R]
river_inlet_triangle_index[R,K]
river_site_id[R]
ksat_mps[N]
impervious_fraction[N]
time_iso[T]
```

## Wave forcing

Nearshore `Sxx`, `Sxy`, and `Syy` cannot be downloaded uniquely from buoy `Hs/Tp/direction`. Run SWAN and compute/store the ADCIRC-style force:

\[
RSX=-\rho_w^{-1}(\partial_xS_{xx}+\partial_yS_{xy}),\qquad
RSY=-\rho_w^{-1}(\partial_xS_{xy}+\partial_yS_{yy}).
\]

If the additional NPZ contains `wave_force_x_m2s2 [T,N]` and `wave_force_y_m2s2 [T,N]`, the patched ANUGA bridge applies them to the two momentum equations.

## Important limitations

- The regional mesh currently identifies artificial northern wet boundaries as  possible river boundaries. Review them against NHDPlus flowlines.
- A gauge may be upstream of storage, branching, or tidal backwater. Its measured discharge is not automatically the correct coastal boundary flux.
- NLD GeoJSON is downloaded but not automatically inserted into the mesh.
- The simple loss model does not represent soil-moisture evolution.
- A full Harvey inland-flood simulation is much more expensive than a  wind-and-tide-only run. Validate first on a smaller spatial crop or shorter interval.
