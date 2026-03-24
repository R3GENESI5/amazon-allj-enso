"""
Nocturnal vs daytime moisture flux partition (spatial map).

Panel (a): Nocturnal fraction of moisture flux magnitude at each grid point
           with arrows for nocturnal mean flux direction
Panel (b): El Nino minus La Nina difference in nocturnal fraction

Data: ERA5 monthly means at 900 hPa, hourly resolution for Feb 1979--2023
Nocturnal = 00--06 UTC (~21--03 local Amazon time)
Daytime   = 12--18 UTC (~09--15 local Amazon time)

Inputs:  data/oni_classification.csv
         data/era5/era5_monthly_means_900hpa_feb_1979_2023.nc
Outputs: figures/fig10_nocturnal_partition.png

Reference: Section 3.10, Figure 10
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ds = xr.open_dataset(DATA_DIR / "era5" /
                     "era5_monthly_means_900hpa_feb_1979_2023.nc")

u = ds["u"].squeeze("pressure_level")
v = ds["v"].squeeze("pressure_level")
q = ds["q"].squeeze("pressure_level")
lat = ds["latitude"].values
lon = ds["longitude"].values
times = pd.DatetimeIndex(ds["valid_time"].values)

oni = pd.read_csv(DATA_DIR / "oni_classification.csv")
unique_years = sorted(times.year.unique())
n_years = len(unique_years)

nocturnal_hours = [0, 1, 2, 3, 4, 5, 6]
daytime_hours = [12, 13, 14, 15, 16, 17, 18]
hours = times.hour

qu_vals = (q * u).values
qv_vals = (q * v).values

noc_mag_annual = np.zeros((n_years, len(lat), len(lon)))
day_mag_annual = np.zeros((n_years, len(lat), len(lon)))
full_mag_annual = np.zeros((n_years, len(lat), len(lon)))
noc_qu_annual = np.zeros((n_years, len(lat), len(lon)))
noc_qv_annual = np.zeros((n_years, len(lat), len(lon)))

for i, yr in enumerate(unique_years):
    yr_mask = times.year == yr

    noc_mask = yr_mask & hours.isin(nocturnal_hours)
    qu_noc = np.nanmean(qu_vals[noc_mask], axis=0)
    qv_noc = np.nanmean(qv_vals[noc_mask], axis=0)
    noc_mag_annual[i] = np.sqrt(qu_noc**2 + qv_noc**2)
    noc_qu_annual[i] = qu_noc
    noc_qv_annual[i] = qv_noc

    day_mask = yr_mask & hours.isin(daytime_hours)
    qu_day = np.nanmean(qu_vals[day_mask], axis=0)
    qv_day = np.nanmean(qv_vals[day_mask], axis=0)
    day_mag_annual[i] = np.sqrt(qu_day**2 + qv_day**2)

    qu_full = np.nanmean(qu_vals[yr_mask], axis=0)
    qv_full = np.nanmean(qv_vals[yr_mask], axis=0)
    full_mag_annual[i] = np.sqrt(qu_full**2 + qv_full**2)

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

# Nocturnal fraction: (noc_mean * 7h) / (full_mean * 24h)
noc_frac_annual = (noc_mag_annual * 7.0) / (full_mag_annual * 24.0 + 1e-20)
noc_frac_clim = np.nanmean(noc_frac_annual, axis=0)
noc_frac_elnino = np.nanmean(noc_frac_annual[elnino_idx], axis=0)
noc_frac_lanina = np.nanmean(noc_frac_annual[lanina_idx], axis=0)
noc_frac_diff = noc_frac_elnino - noc_frac_lanina

noc_qu_clim = np.nanmean(noc_qu_annual, axis=0)
noc_qv_clim = np.nanmean(noc_qv_annual, axis=0)

# Domain average
amazon_mask = ((lat[:, None] >= -15) & (lat[:, None] <= 5) &
               (lon[None, :] >= -75) & (lon[None, :] <= -45))
mean_noc = np.nanmean(noc_frac_clim[amazon_mask])
print(f"Domain-average nocturnal fraction: {mean_noc:.3f} ({mean_noc*100:.1f}%)")

# ── Figure ───────────────────────────────────────────────────────────
skip = 8
extent = [-80, -30, -20, 15]
LON, LAT_grid = np.meshgrid(lon, lat)

fig, (ax_a, ax_b) = plt.subplots(
    1, 2, figsize=(16, 6),
    subplot_kw={"projection": ccrs.PlateCarree()})

# (a) Nocturnal fraction
ax_a.set_extent(extent, crs=ccrs.PlateCarree())
ax_a.add_feature(cfeature.COASTLINE, linewidth=0.8, color="black", zorder=6)
ax_a.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle="--",
                 color="#444444", zorder=6)

norm_frac = mcolors.TwoSlopeNorm(vmin=0.40, vcenter=0.50, vmax=0.65)
cf_a = ax_a.pcolormesh(LON, LAT_grid, noc_frac_clim, cmap="RdYlBu_r",
                       norm=norm_frac, transform=ccrs.PlateCarree(),
                       shading="auto", zorder=2)

# Direction arrows
qu_sub = noc_qu_clim[::skip, ::skip]
qv_sub = noc_qv_clim[::skip, ::skip]
mag_sub = np.sqrt(qu_sub**2 + qv_sub**2)
mag_sub = np.where(mag_sub < 1e-10, 1e-10, mag_sub)
threshold = np.nanpercentile(mag_sub[mag_sub > 0], 10)
qu_norm = np.where(mag_sub > threshold, qu_sub / mag_sub, np.nan)
qv_norm = np.where(mag_sub > threshold, qv_sub / mag_sub, np.nan)
ax_a.quiver(LON[::skip, ::skip], LAT_grid[::skip, ::skip], qu_norm, qv_norm,
            transform=ccrs.PlateCarree(), scale=28, scale_units="width",
            width=0.003, headwidth=4, headlength=3.5, headaxislength=3,
            color="#1a1a1a", alpha=0.55, zorder=4)

gl_a = ax_a.gridlines(draw_labels=True, linewidth=0.3, color="gray",
                       alpha=0.5, linestyle="--")
gl_a.top_labels = False
gl_a.right_labels = False
gl_a.xlocator = mticker.FixedLocator([-75, -60, -45, -30])
gl_a.ylocator = mticker.FixedLocator([-15, -5, 5])
ax_a.set_title("(a) Nocturnal fraction of moisture flux", fontsize=11,
               fontweight="bold", loc="left")
ax_a.text(0.02, 0.02, f"Amazon core mean: {mean_noc*100:.0f}% nocturnal",
          transform=ax_a.transAxes, fontsize=8, va="bottom",
          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
          zorder=10)

cbar_a = plt.colorbar(cf_a, ax=ax_a, orientation="horizontal", pad=0.07,
                       shrink=0.85, aspect=25, extend="both")
cbar_a.set_label("Nocturnal fraction", fontsize=9)
cbar_a.ax.axvline(0.5, color="black", linewidth=1.2, zorder=10)

# (b) El Nino - La Nina difference
ax_b.set_extent(extent, crs=ccrs.PlateCarree())
ax_b.add_feature(cfeature.COASTLINE, linewidth=0.8, color="black", zorder=6)
ax_b.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle="--",
                 color="#444444", zorder=6)

clim_total_mag = np.nanmean(noc_mag_annual + day_mag_annual, axis=0)
flux_threshold = np.nanpercentile(clim_total_mag, 15)
noc_frac_diff_masked = np.where(clim_total_mag > flux_threshold,
                                noc_frac_diff, np.nan)
valid = noc_frac_diff_masked[np.isfinite(noc_frac_diff_masked)]
vmax_diff = max(abs(np.nanpercentile(valid, 5)),
                abs(np.nanpercentile(valid, 95)), 0.01)

cf_b = ax_b.pcolormesh(LON, LAT_grid, noc_frac_diff_masked, cmap="RdBu_r",
                       vmin=-vmax_diff, vmax=vmax_diff,
                       transform=ccrs.PlateCarree(), shading="auto", zorder=2)

gl_b = ax_b.gridlines(draw_labels=True, linewidth=0.3, color="gray",
                       alpha=0.5, linestyle="--")
gl_b.top_labels = False
gl_b.right_labels = False
gl_b.left_labels = False
gl_b.xlocator = mticker.FixedLocator([-75, -60, -45, -30])
gl_b.ylocator = mticker.FixedLocator([-15, -5, 5])
ax_b.set_title(f"(b) El Ni\u00f1o minus La Ni\u00f1a nocturnal fraction",
               fontsize=11, fontweight="bold", loc="left")
ax_b.text(0.02, 0.02, f"n = {len(elnino_idx)} vs {len(lanina_idx)} years",
          transform=ax_b.transAxes, fontsize=8, va="bottom",
          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
          zorder=10)

cbar_b = plt.colorbar(cf_b, ax=ax_b, orientation="horizontal", pad=0.07,
                       shrink=0.85, aspect=25, extend="both")
cbar_b.set_label(r"$\Delta$ Nocturnal fraction (El Ni\u00f1o $-$ La Ni\u00f1a)",
                 fontsize=9)

fig.suptitle("February 900 hPa Nocturnal Moisture Flux Partition "
             "(1979\u20132023)", fontsize=13, fontweight="bold", y=1.02)
plt.subplots_adjust(wspace=0.08)

outpath = FIG_DIR / "fig10_nocturnal_partition.png"
fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {outpath}")
plt.close()
ds.close()
print("Script 11 complete.")
