import os
# Silence tqdm progress bars and URL streaming globally
os.environ["TQDM_DISABLE"] = "1"

import logging
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

# Disable internal ECMWF / multiurl verbosity
logging.getLogger("ecmwf.opendata").setLevel(logging.ERROR)
logging.getLogger("multiurl").setLevel(logging.ERROR)

from ecmwf.opendata import Client

# --- CONFIGURATION & PATHS ---
SOURCE_FILE = Path("pg.grib2")
LSM_FILE = Path("lsm.grib2")

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

def cleanup_temporary_files():
    """Remove partial downloaded steps and existing grib files on fallback/failure."""
    for p in Path(".").glob("step_*.tmp"):
        try:
            p.unlink()
        except Exception:
            pass
    if SOURCE_FILE.exists():
        SOURCE_FILE.unlink()
    if LSM_FILE.exists():
        LSM_FILE.unlink()

# --- DATA RETRIEVAL WITH FALLBACK ---
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
        
print("Download script complete.", flush=True)