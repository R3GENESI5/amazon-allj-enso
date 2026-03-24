"""
500 hPa omega (vertical velocity) composites by ENSO phase.

Tests whether El Nino produces broad subsidence over Amazonia via
the Walker circulation shift. Computes February-mean omega at 500 hPa,
composites by ENSO phase, and applies Welch's t-test for significance.

Inputs:  data/oni_classification.csv
         data/era5/era5_monthly_means_upper_feb_1979_2023.nc
Outputs: figures/fig04_enso_subsidence_omega500.png

Reference: Section 3.4, Figure 4
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

oni = pd.read_csv(DATA_DIR / "oni_classification.csv")
ds = xr.open_dataset(DATA_DIR / "era5" /
                     "era5_monthly_means_upper_feb_1979_2023.nc")

omega = ds["w"].sel(pressure_level=500.0)
lat = ds["latitude"].values
lon = ds["longitude"].values

times = pd.DatetimeIndex(ds["valid_time"].values)
years_all = times.year
unique_years = sorted(set(years_all))
n_years = len(unique_years)

omega_vals = omega.values
omega_annual = np.zeros((n_years, len(lat), len(lon)))
for i, yr in enumerate(unique_years):
    mask = years_all == yr
    omega_annual[i] = np.nanmean(omega_vals[mask], axis=0)

# Classify
elnino_idx, lanina_idx, neutral_idx = [], [], []
for i, yr in enumerate(unique_years):
    row = oni[oni["year"] == yr]
    if len(row) == 0:
        neutral_idx.append(i)
        continue
    phase = row["phase"].values[0]
    if phase == "El Nino":
        elnino_idx.append(i)
    elif phase == "La Nina":
        lanina_idx.append(i)
    else:
        neutral_idx.append(i)

omega_elnino = np.nanmean(omega_annual[elnino_idx], axis=0)
omega_lanina = np.nanmean(omega_annual[lanina_idx], axis=0)
omega_neutral = np.nanmean(omega_annual[neutral_idx], axis=0)
omega_diff = omega_elnino - omega_lanina

# Scale: display as 10^-2 Pa/s
scale = 100
unit_label = r"$\omega$ ($\times 10^{-2}$ Pa s$^{-1}$)"

# Significance
pval = np.zeros((len(lat), len(lon)))
for j in range(len(lat)):
    for k in range(len(lon)):
        _, pval[j, k] = stats.ttest_ind(
            omega_annual[elnino_idx, j, k],
            omega_annual[lanina_idx, j, k], equal_var=False)
sig_mask = pval < 0.05

# Basin-mean diagnostic
lat_mask = (lat >= -15) & (lat <= -5)
lon_mask = (lon >= -70) & (lon <= -50)
print(f"Amazon basin mean omega (Pa/s):")
print(f"  El Nino:  {np.nanmean(omega_elnino[np.ix_(lat_mask, lon_mask)]):.5f}")
print(f"  La Nina:  {np.nanmean(omega_lanina[np.ix_(lat_mask, lon_mask)]):.5f}")
print(f"  Neutral:  {np.nanmean(omega_neutral[np.ix_(lat_mask, lon_mask)]):.5f}")

# ── Figure ───────────────────────────────────────────────────────────
LON, LAT_g = np.meshgrid(lon, lat)
extent = [-80, -30, -20, 10]

vmax_comp = max(np.nanpercentile(np.abs(omega_elnino * scale), 97),
                np.nanpercentile(np.abs(omega_lanina * scale), 97),
                np.nanpercentile(np.abs(omega_neutral * scale), 97))
vmax_diff = np.nanpercentile(np.abs(omega_diff * scale), 97)

fig, axes = plt.subplots(2, 2, figsize=(12, 9),
                          subplot_kw={"projection": ccrs.PlateCarree()})

panels = [
    (axes[0, 0], omega_elnino * scale,
     f"(a) El Ni\u00f1o (n={len(elnino_idx)})", -vmax_comp, vmax_comp, False),
    (axes[0, 1], omega_lanina * scale,
     f"(b) La Ni\u00f1a (n={len(lanina_idx)})", -vmax_comp, vmax_comp, False),
    (axes[1, 0], omega_neutral * scale,
     f"(c) Neutral (n={len(neutral_idx)})", -vmax_comp, vmax_comp, False),
    (axes[1, 1], omega_diff * scale,
     "(d) El Ni\u00f1o \u2212 La Ni\u00f1a", -vmax_diff, vmax_diff, True),
]

for ax, data, title, vmin, vmax, is_diff in panels:
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color="#333333")
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle="--",
                   color="#888888")
    cf = ax.pcolormesh(LON, LAT_g, data, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                       transform=ccrs.PlateCarree(), shading="auto")
    if is_diff:
        ax.contourf(LON, LAT_g, sig_mask.astype(float), levels=[0.5, 1.5],
                    hatches=["..."], colors="none",
                    transform=ccrs.PlateCarree(), zorder=4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray",
                       alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([-80, -70, -60, -50, -40, -30])
    gl.ylocator = mticker.FixedLocator([-20, -15, -10, -5, 0, 5, 10])
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
    cbar = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.08,
                         shrink=0.85, aspect=25)
    cbar.set_label(f"\u0394{unit_label}" if is_diff else unit_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)

axes[0, 0].text(0.02, 0.02,
    "warm = subsidence (+\u03c9)\ncool = ascent (\u2212\u03c9)",
    transform=axes[0, 0].transAxes, fontsize=6.5, va="bottom", ha="left",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
              edgecolor="#cccccc", alpha=0.85), zorder=10)

fig.suptitle("February 500 hPa Vertical Velocity (\u03c9) by ENSO Phase "
             "(1979\u20132023)", fontsize=13, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])

outpath = FIG_DIR / "fig04_enso_subsidence_omega500.png"
fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {outpath}")
plt.close()
ds.close()
print("Script 05 complete.")
