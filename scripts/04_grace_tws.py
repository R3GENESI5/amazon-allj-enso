"""
GRACE terrestrial water storage anomaly time series.

Computes area-weighted mean TWS anomaly over the Amazon basin
(5N--15S, 75W--45W) from GRACE/GRACE-FO JPL mascon data, with
ENSO period shading and 12-month running mean.

Inputs:  data/oni_classification.csv
         data/grace/GRCTellus.JPL.*.GLO.RL06.*.MSCNv04CRI.nc
Outputs: figures/fig03_grace_tws_anomaly.png

Reference: Section 3.3, Figure 3
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

oni = pd.read_csv(DATA_DIR / "oni_classification.csv")

# Find the GRACE file (filename may vary by version)
grace_dir = DATA_DIR / "grace"
grace_files = list(grace_dir.glob("GRCTellus.JPL.*.nc"))
if not grace_files:
    raise FileNotFoundError(f"No GRACE file found in {grace_dir}")
ds = xr.open_dataset(grace_files[0])

lats = ds["lat"].values
lons = ds["lon"].values

# Handle 0-360 longitude convention
if lons.max() > 180:
    lon_min, lon_max = 285, 315
else:
    lon_min, lon_max = -75, -45
lat_min, lat_max = -15, 5

lat_sel = (ds.lat >= lat_min) & (ds.lat <= lat_max)
lon_sel = (ds.lon >= lon_min) & (ds.lon <= lon_max)

land_mask = ds["land_mask"].where(lat_sel & lon_sel, drop=True)
scale_factor = ds["scale_factor"].where(lat_sel & lon_sel, drop=True)
tws_raw = ds["lwe_thickness"].where(lat_sel & lon_sel, drop=True)
tws = tws_raw * scale_factor
tws = tws.where(land_mask == 1)

weights = np.cos(np.deg2rad(tws.lat))
tws_mean = tws.weighted(weights).mean(dim=["lat", "lon"])

times = pd.DatetimeIndex(ds["time"].values)
tws_values = tws_mean.values

print(f"TWS: {len(tws_values)} months, {times[0]:%Y-%m} to {times[-1]:%Y-%m}")
print(f"Range: {np.nanmin(tws_values):.1f} to {np.nanmax(tws_values):.1f} cm")

# ENSO period shading
elnino_periods, lanina_periods = [], []
for _, row in oni.iterrows():
    yr = int(row["year"])
    phase = row["phase"]
    start = datetime(yr - 1, 7, 1)
    end = datetime(yr, 6, 30)
    if phase == "El Nino":
        elnino_periods.append((start, end))
    elif phase == "La Nina":
        lanina_periods.append((start, end))

# ── Figure ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))

for start, end in elnino_periods:
    ax.axvspan(start, end, alpha=0.15, color="#E63946", zorder=0)
for start, end in lanina_periods:
    ax.axvspan(start, end, alpha=0.15, color="#457B9D", zorder=0)

ax.plot(times, tws_values, color="#2D3142", linewidth=1.2, zorder=3)

tws_smooth = pd.Series(tws_values, index=times).rolling(
    window=12, center=True, min_periods=6).mean()
ax.plot(times, tws_smooth.values, color="#E63946", linewidth=2.0, zorder=4,
        label="12-month running mean")
ax.axhline(y=0, color="black", linewidth=0.5, zorder=2)

legend_elements = [
    Patch(facecolor="#E63946", alpha=0.2, label="El Ni\u00f1o periods"),
    Patch(facecolor="#457B9D", alpha=0.2, label="La Ni\u00f1a periods"),
    plt.Line2D([0], [0], color="#E63946", linewidth=2,
               label="12-month running mean"),
    plt.Line2D([0], [0], color="#2D3142", linewidth=1.2,
               label="Monthly TWS anomaly"),
]
ax.legend(handles=legend_elements, fontsize=9, loc="lower left",
          framealpha=0.9, ncol=2)

ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("TWS anomaly (cm eq. water height)", fontsize=11)
ax.set_title("GRACE Terrestrial Water Storage Anomaly over the Amazon Basin "
             "(5\u00b0N\u201315\u00b0S, 75\u00b0W\u201345\u00b0W)",
             fontsize=13, fontweight="bold", loc="left")
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(labelsize=9)
ax.grid(True, alpha=0.2, linewidth=0.5)
ax.set_xlim(times[0], times[-1])

plt.tight_layout()
outpath = FIG_DIR / "fig03_grace_tws_anomaly.png"
fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {outpath}")
plt.close()
ds.close()
print("Script 04 complete.")
