"""
CHIRPS precipitation coast-to-interior transect by ENSO phase.

Extracts February precipitation along a 0--3S latitude band from
44W (coast) to 70W (interior), composites by ENSO phase, and computes
the coast-to-interior precipitation gradient.

Inputs:  data/oni_classification.csv
         data/chirps/chirps-v2.0.YYYY.monthly.nc
Outputs: figures/fig02_chirps_transect_enso.png

Reference: Section 3.2, Figure 2
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

oni = pd.read_csv(DATA_DIR / "oni_classification.csv")

# Transect: 0-3S, 44W to 70W
lat_min, lat_max = -3.0, 0.0
lon_min, lon_max = -70.0, -44.0
chirps_dir = DATA_DIR / "chirps"

transect_data = {}
lons_transect = None

for yr in range(1981, 2025):
    fname = chirps_dir / f"chirps-v2.0.{yr}.monthly.nc"
    if not fname.exists():
        continue
    ds = xr.open_dataset(fname)
    feb_precip = ds["precip"].isel(time=1)
    lat_sel = (ds.latitude >= lat_min) & (ds.latitude <= lat_max)
    lon_sel = (ds.longitude >= lon_min) & (ds.longitude <= lon_max)
    region = feb_precip.where(lat_sel & lon_sel, drop=True)
    transect = region.mean(dim="latitude").values
    if lons_transect is None:
        lons_transect = region.longitude.values
    transect_data[yr] = transect
    ds.close()

print(f"Loaded {len(transect_data)} years of CHIRPS data")

# Classify by ENSO phase
elnino_t, lanina_t, neutral_t = [], [], []
for yr, transect in transect_data.items():
    row = oni[oni["year"] == yr]
    if len(row) == 0:
        neutral_t.append(transect)
        continue
    phase = row["phase"].values[0]
    if phase == "El Nino":
        elnino_t.append(transect)
    elif phase == "La Nina":
        lanina_t.append(transect)
    else:
        neutral_t.append(transect)

elnino_arr = np.array(elnino_t)
lanina_arr = np.array(lanina_t)
neutral_arr = np.array(neutral_t)

elnino_mean = np.nanmean(elnino_arr, axis=0)
lanina_mean = np.nanmean(lanina_arr, axis=0)
neutral_mean = np.nanmean(neutral_arr, axis=0)
elnino_se = np.nanstd(elnino_arr, axis=0) / np.sqrt(len(elnino_t))
lanina_se = np.nanstd(lanina_arr, axis=0) / np.sqrt(len(lanina_t))
neutral_se = np.nanstd(neutral_arr, axis=0) / np.sqrt(len(neutral_t))

# Coast-to-interior gradient
coast_mask = (lons_transect >= -52) & (lons_transect <= -44)
interior_mask = (lons_transect >= -70) & (lons_transect <= -58)

def gradient_metric(arr):
    coast_mean = np.nanmean(arr[:, coast_mask], axis=1)
    interior_mean = np.nanmean(arr[:, interior_mask], axis=1)
    return interior_mean - coast_mean

grad_elnino = gradient_metric(elnino_arr)
grad_lanina = gradient_metric(lanina_arr)
grad_neutral = gradient_metric(neutral_arr)

# ── Figure ───────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                gridspec_kw={"width_ratios": [2.5, 1]})

# Panel (a): Transect
for mean, se, color, n, label in [
    (elnino_mean, elnino_se, "#E63946", len(elnino_t), "El Ni\u00f1o"),
    (lanina_mean, lanina_se, "#457B9D", len(lanina_t), "La Ni\u00f1a"),
    (neutral_mean, neutral_se, "#888888", len(neutral_t), "Neutral"),
]:
    ax1.fill_between(lons_transect, mean - 1.96 * se, mean + 1.96 * se,
                      alpha=0.15 if color != "#888888" else 0.10, color=color)
    ls = "--" if color == "#888888" else "-"
    ax1.plot(lons_transect, mean, color=color, linewidth=2.2,
             linestyle=ls, label=f"{label} (n={n})", zorder=5)

ax1.axvspan(-52, -44, alpha=0.06, color="cyan", label="Coast zone")
ax1.axvspan(-70, -58, alpha=0.06, color="green", label="Interior zone")
ax1.set_xlabel("Longitude (\u00b0W)", fontsize=11)
ax1.set_ylabel("February precipitation (mm month$^{-1}$)", fontsize=11)
ax1.set_title("(a) Coast-to-interior precipitation transect (0\u20133\u00b0S)",
              fontsize=12, fontweight="bold", loc="left")
ax1.legend(fontsize=9, loc="upper left", framealpha=0.9)
ax1.set_xlim(lon_min, lon_max)
ax1.invert_xaxis()
ax1.grid(True, alpha=0.3, linewidth=0.5)

# Panel (b): Gradient box plot
bp = ax2.boxplot([grad_elnino, grad_neutral, grad_lanina],
                  positions=[1, 2, 3], widths=0.5, patch_artist=True,
                  showmeans=True,
                  meanprops=dict(marker="D", markerfacecolor="white",
                                markeredgecolor="black", markersize=6),
                  medianprops=dict(color="black", linewidth=1.5))
for patch, color in zip(bp["boxes"], ["#E63946", "#888888", "#457B9D"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax2.set_xticklabels(["El Ni\u00f1o", "Neutral", "La Ni\u00f1a"], fontsize=10)
ax2.set_ylabel("Interior \u2212 Coast precip. (mm month$^{-1}$)", fontsize=10)
ax2.set_title("(b) Precipitation gradient", fontsize=12,
              fontweight="bold", loc="left")
ax2.axhline(y=0, color="black", linewidth=0.5)
ax2.grid(True, axis="y", alpha=0.3, linewidth=0.5)

t_stat, p_val = stats.ttest_ind(grad_elnino, grad_lanina, equal_var=False)
sig_text = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"
ax2.annotate(f"EN vs LN: {sig_text}", xy=(0.5, 0.95),
             xycoords="axes fraction", fontsize=8, ha="center",
             style="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                       alpha=0.8))

plt.tight_layout()
outpath = FIG_DIR / "fig02_chirps_transect_enso.png"
fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {outpath}")
plt.close()
print("Script 03 complete.")
