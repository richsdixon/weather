import os
# Silence tqdm progress bars and URL streaming globally
os.environ["TQDM_DISABLE"] = "1"

import shutil
import logging
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import xarray as xr

# Disable internal ECMWF / multiurl verbosity
logging.getLogger("ecmwf.opendata").setLevel(logging.ERROR)
logging.getLogger("multiurl").setLevel(logging.ERROR)

from ecmwf.opendata import Client

# Force non-interactive Matplotlib backend for GitHub Actions runner
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.collections import LineCollection

# Map feature extraction dependencies
import cartopy.feature as cfeature
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon

# --- CONFIGURATION & PATHS ---
CROP_LAT_NORTH = 62.5
CROP_LAT_SOUTH = 37.5
CROP_LON_WEST = -12.5
CROP_LON_EAST = 20.0
GUST_THRESHOLD = 20  # m/s

PLOT_LAT_NORTH = CROP_LAT_NORTH
PLOT_LAT_SOUTH = CROP_LAT_SOUTH
PLOT_LON_WEST = CROP_LON_WEST
PLOT_LON_EAST = CROP_LON_EAST
font_name = "sans-serif"

SOURCE_FILE = Path("pg.grib2")
LSM_FILE = Path("lsm.grib2")
OUTPUT_FILE = Path("gust_summary_2x2.png")

# --- DETERMINE INITIAL FORECAST RUN TIME ---
now_utc = datetime.now(timezone.utc)
current_hour = now_utc.hour
current_minute = now_utc.minute
total_minutes = current_hour * 60 + current_minute

# 08:45 UTC (525 min) to 21:45 UTC (1305 min) -> Try 00Z today
if 525 <= total_minutes < 1305:
    target_time = 0   # 00Z ensemble run
    target_date = 0   # Today
    run_label = "00Z"
# 21:45 UTC onwards -> Try 12Z today
elif total_minutes >= 1305:
    target_time = 12  # 12Z ensemble run
    target_date = 0   # Today
    run_label = "12Z"
# 00:00 UTC to 08:44 UTC -> Try 12Z from yesterday
else:
    target_time = 12  # 12Z ensemble run from previous day
    target_date = -1  # Yesterday
    run_label = "12Z (Yesterday)"

print(f"Current UTC time: {now_utc.strftime('%Y-%m-%d %H:%M UTC')}", flush=True)

# --- MAP VECTOR LAYERS ---
def extract_line_coords(feature_name, category="cultural", scale="50m"):
    """Extract line coordinate arrays for Matplotlib LineCollection."""
    feature = cfeature.NaturalEarthFeature(category, feature_name, scale)
    lines = []
    
    for geom in feature.geometries():
        if isinstance(geom, LineString):
            lines.append(np.array(geom.coords))
        elif isinstance(geom, MultiLineString):
            for line in geom.geoms:
                lines.append(np.array(line.coords))
        elif isinstance(geom, Polygon):
            lines.append(np.array(geom.exterior.coords))
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                lines.append(np.array(poly.exterior.coords))
                
    return lines

coastlines = extract_line_coords("coastline", category="physical", scale="50m")
borders = extract_line_coords("admin_0_boundary_lines_land", category="cultural", scale="50m")

# --- HELPER FUNCTIONS ---
def crop_region(dataset):
    dataset = dataset.assign_coords(
        longitude=((dataset.longitude + 180) % 360) - 180
    ).sortby("longitude").sortby("latitude", ascending=False)
    
    cropped = dataset.sel(
        latitude=slice(CROP_LAT_NORTH, CROP_LAT_SOUTH),
        longitude=slice(CROP_LON_WEST, CROP_LON_EAST)
    )
    if cropped.sizes["latitude"] == 0 or cropped.sizes["longitude"] == 0:
        raise ValueError("The selected crop contains no grid points.")
    return cropped

def make_scale(low, high, step, colour_map):
    bounds = np.append(np.arange(low, high, step), high)
    cmap = plt.get_cmap(colour_map, len(bounds) - 1).copy()
    cmap.set_bad((0, 0, 0, 0))
    norm = colors.BoundaryNorm(bounds, ncolors=cmap.N, clip=True)
    return bounds, cmap, norm

def cleanup_temporary_files():
    """Remove partial downloaded steps and existing grib files."""
    for p in Path(".").glob("step_*.tmp"):
        try:
            p.unlink()
        except Exception:
            pass
    if SOURCE_FILE.exists():
        SOURCE_FILE.unlink()
    if LSM_FILE.exists():
        LSM_FILE.unlink()

# --- 1. DATA RETRIEVAL WITH FALLBACK ---
client = Client(source="azure")

def fetch_ensemble_run(date, time, label):
    """Attempt to download all forecast steps for a specific date and time."""
    print(f"Attempting download for {label} run (time={time}, date={date})...", flush=True)

    def download_single_step(step):
        temp_file = f"step_{step:03d}.tmp"
        client.retrieve(
            date=date,
            time=time,
            stream="enfo",
            type="pf",
            param="10fg",
            step=step,
            target=temp_file
        )
        print(f"  [{label}] Downloaded step t+{step:02d}h from Azure", flush=True)
        return temp_file

    steps = list(range(3, 93, 3))

    with ThreadPoolExecutor(max_workers=10) as executor:
        temp_files = list(executor.map(download_single_step, steps))

    print(f"Concatenating step files into {SOURCE_FILE}...", flush=True)
    with open(SOURCE_FILE, "wb") as outfile:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                with open(temp_file, "rb") as infile:
                    outfile.write(infile.read())
                os.remove(temp_file)

    print(f"Downloading land-sea mask for {label} run...", flush=True)
    client.retrieve(
        date=date,
        time=time,
        stream="enfo",
        type="pf",
        param="lsm",
        step=[0],
        target=str(LSM_FILE)
    )
    print("Land-sea mask downloaded successfully.", flush=True)

# Attempt primary run with fallback to 00Z if 12Z is incomplete/missing on Azure
try:
    fetch_ensemble_run(target_date, target_time, run_label)
except Exception as e:
    if target_time == 12:
        print(f"\n[WARNING] Failed to download {run_label} ensemble run: {e}", flush=True)
        print("12Z data is likely not fully uploaded to Azure yet. Falling back to 00Z run...", flush=True)
        cleanup_temporary_files()
        
        target_time = 0
        target_date = 0
        run_label = "00Z (Fallback)"
        fetch_ensemble_run(target_date, target_time, run_label)
    else:
        cleanup_temporary_files()
        raise e

# --- 2. LAND-SEA MASK PROCESSING ---
with xr.open_dataset(
    LSM_FILE, engine="cfgrib", backend_kwargs={"indexpath": ""}
) as dataset:
    lsm = crop_region(dataset).load()

lsm["lsm"] = (lsm["lsm"] >= 0.5).astype("int8")
lsm["lsm"].attrs = {"long_name": "Binary land mask", "units": "1"}
land_mask = lsm["lsm"].isel({
    dim: 0 for dim in lsm["lsm"].dims if dim not in ("latitude", "longitude")
}, drop=True)
print("Land-sea mask ready.", flush=True)

# --- 3. DIRECT DATA PROCESSING ---
print("Processing GRIB dataset with xarray...", flush=True)
with xr.open_dataset(
    SOURCE_FILE, engine="cfgrib", backend_kwargs={"indexpath": ""}
) as ds:
    cropped = crop_region(ds)
    gust_max = cropped["fg10"].max(dim="step", keep_attrs=True).load()

gust_field, land_mask = xr.align(gust_max, land_mask, join="exact")
landgust = (gust_field * land_mask).rename("landgust")
landgust.attrs = gust_max.attrs.copy()
landgust.attrs["long_name"] = "Maximum gust over land"

member_score = (
    (landgust - GUST_THRESHOLD).where(landgust > GUST_THRESHOLD, 0) ** 3
).sum(dim=["latitude", "longitude"]).rename("member_score")

print(f"Calculations complete for {landgust.sizes['number']} members.", flush=True)

# --- 4. PLOTTING & VISUALIZATION ---
THRESHOLDS_MPH = [50, 70]
PROB_COLOUR_MAP = "jet"
PROB_COLOUR_STEP = 10
MAX_GUST_COLOUR_MAP = "jet"
MAX_GUST_MIN = 50
MAX_GUST_MAX = 130
MAX_GUST_STEP = 10
MEDIAN_GUST_COLOUR_MAP = "jet"
MEDIAN_GUST_MIN = 40
MEDIAN_GUST_MAX = 100
MEDIAN_GUST_STEP = 5

gust_mph = gust_max / 0.44704
thresholds = xr.DataArray(
    THRESHOLDS_MPH,
    dims="threshold_mph",
    coords={"threshold_mph": THRESHOLDS_MPH}
)

valid_members = gust_mph.count(dim="number")
gust_probability = (
    100 * (gust_mph > thresholds).sum(dim="number")
    / valid_members.where(valid_members > 0)
).rename("gust_probability")

absolute_max_gust = gust_mph.max(dim="number", skipna=True)
median_max_gust = gust_mph.median(dim="number", skipna=True)

prob_bounds, prob_cmap, prob_norm = make_scale(0, 100, PROB_COLOUR_STEP, PROB_COLOUR_MAP)
max_bounds, max_cmap, max_norm = make_scale(MAX_GUST_MIN, MAX_GUST_MAX, MAX_GUST_STEP, MAX_GUST_COLOUR_MAP)
median_bounds, median_cmap, median_norm = make_scale(MEDIAN_GUST_MIN, MEDIAN_GUST_MAX, MEDIAN_GUST_STEP, MEDIAN_GUST_COLOUR_MAP)

forecast_time = gust_max["time"].dt.strftime("%d/%m/%y %HZ").item()

panels = [
    (gust_probability.sel(threshold_mph=THRESHOLDS_MPH[0]).where(gust_probability.sel(threshold_mph=THRESHOLDS_MPH[0]) > 0),
     f"Maximum gust > {THRESHOLDS_MPH[0]} mph", prob_cmap, prob_norm),
    (gust_probability.sel(threshold_mph=THRESHOLDS_MPH[1]).where(gust_probability.sel(threshold_mph=THRESHOLDS_MPH[1]) > 0),
     f"Maximum gust > {THRESHOLDS_MPH[1]} mph", prob_cmap, prob_norm),
    (absolute_max_gust.where(absolute_max_gust >= MAX_GUST_MIN),
     "Absolute maximum gust across members", max_cmap, max_norm),
    (median_max_gust.where(median_max_gust >= MEDIAN_GUST_MIN),
     "Median maximum gust across members", median_cmap, median_norm)
]

with plt.rc_context({"font.family": font_name, "font.size": 11}):
    fig_probability, axes_probability = plt.subplots(2, 2, figsize=(15, 12))
    fig_probability.subplots_adjust(left=0.03, right=0.87, bottom=0.11, top=0.92, wspace=0.08, hspace=0.08)

    for ax, (field, title, cmap, norm) in zip(axes_probability.flat, panels):
        field = field.transpose("latitude", "longitude")
        ax.pcolormesh(
            field.longitude.values,
            field.latitude.values,
            np.ma.masked_invalid(field.values),
            shading="auto",
            cmap=cmap,
            norm=norm,
            rasterized=True
        )
        if coastlines:
            ax.add_collection(LineCollection(coastlines, colors="black", linewidths=0.65, zorder=3))
        if borders:
            ax.add_collection(LineCollection(borders, colors="black", linewidths=0.6, zorder=4))
            
        ax.set_xlim(PLOT_LON_WEST, PLOT_LON_EAST)
        ax.set_ylim(PLOT_LAT_SOUTH, PLOT_LAT_NORTH)
        ax.set_aspect("equal")
        ax.set_anchor("N")
        ax.set_title(title, fontsize=13, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])

    fig_probability.suptitle(
        f"Next 90 Hours: ECMWF Ensemble Gusts | Forecast Time: {forecast_time}",
        fontsize=16, y=0.98
    )

    position = axes_probability[0, 1].get_position()
    cax = fig_probability.add_axes([0.90, position.y0, 0.018, position.height])
    cbar = fig_probability.colorbar(
        plt.cm.ScalarMappable(norm=prob_norm, cmap=prob_cmap),
        cax=cax, boundaries=prob_bounds, ticks=prob_bounds,
        spacing="proportional", drawedges=True, format="%g"
    )
    cbar.set_label("Chance (%)", fontsize=12, labelpad=12)

    for column, cmap, norm, bounds, label in [
        (0, max_cmap, max_norm, max_bounds, "Absolute maximum gust (mph)"),
        (1, median_cmap, median_norm, median_bounds, "Median maximum gust (mph)")
    ]:
        position = axes_probability[1, column].get_position()
        cax = fig_probability.add_axes([position.x0, position.y0 - 0.045, position.width, 0.018])
        cbar = fig_probability.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cax, orientation="horizontal", boundaries=bounds, ticks=bounds,
            spacing="proportional", drawedges=True, format="%g"
        )
        cbar.set_label(label, fontsize=11, labelpad=7)
        cbar.ax.tick_params(labelsize=10)

    fig_probability.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig_probability)
    print(f"Summary figure saved to {OUTPUT_FILE}", flush=True)

# --- 5. CLEANUP TEMPORARY FILES ---
print("Cleaning up temporary GRIB files...", flush=True)
cleanup_temporary_files()
print("Processing complete.", flush=True)
