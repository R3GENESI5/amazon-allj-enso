"""
45-year ENSO composite of ALLJ moisture flux at 900 hPa.

Computes February-mean moisture flux (q*u, q*v) at 900 hPa for each year,
composites by ENSO phase (El Nino / La Nina / Neutral), and plots:
  - 4-panel map: El Nino, La Nina, Neutral composites + difference
  - Split versions: 1x3 composites and standalone difference panel

Inputs:  data/oni_classification.csv
         data/era5/era5_monthly_means_900hpa_feb_1979_2023.nc
Outputs: figures/fig01_enso_moisture_flux_composites.png
         figures/fig01a_enso_composites.png
         figures/fig01b_enso_difference.png

Reference: Section 3.1, Figure 1
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

# ── Paths ────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────
oni = pd.read_csv(DATA_DIR / "oni_classification.csv")
ds = xr.open_dataset(DATA_DIR / "era5" /
                     "era5_monthly_means_900hpa_feb_1979_2023.nc")

u = ds["u"].squeeze("pressure_level")
v = ds["v"].squeeze("pressure_level")
q = ds["q"].squeeze("pressure_level")
lat = ds["latitude"].values
lon = ds["longitude"].values

# Group by year and compute February means
times = pd.DatetimeIndex(ds["valid_time"].values)
years = times.year
unique_years = sorted(set(years))
n_years = len(unique_years)

qu_all = (q * u).values
qv_all = (q * v).values

qu_annual = np.zeros((n_years, len(lat), len(lon)))
qv_annual = np.zeros((n_years, len(lat), len(lon)))

for i, yr in enumerate(unique_years):
    mask = years == yr
    qu_annual[i] = np.nanmean(qu_all[mask], axis=0)
    qv_annual[i] = np.nanmean(qv_all[mask], axis=0)

print(f"Annual means: {qu_annual.shape}, years {unique_years[0]}-{unique_years[-1]}")

# ── Classify years ───────────────────────────────────────────────────
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

print(f"El Nino: {len(elnino_idx)}, La Nina: {len(lanina_idx)}, "
      f"Neutral: {len(neutral_idx)}")

# ── Composites ───────────────────────────────────────────────────────
qu_elnino = np.nanmean(qu_annual[elnino_idx], axis=0)
qv_elnino = np.nanmean(qv_annual[elnino_idx], axis=0)
qu_lanina = np.nanmean(qu_annual[lanina_idx], axis=0)
qv_lanina = np.nanmean(qv_annual[lanina_idx], axis=0)
qu_neutral = np.nanmean(qu_annual[neutral_idx], axis=0)
qv_neutral = np.nanmean(qv_annual[neutral_idx], axis=0)

qu_diff = qu_elnino - qu_lanina
qv_diff = qv_elnino - qv_lanina

qflux_elnino = np.sqrt(qu_elnino**2 + qv_elnino**2)
qflux_lanina = np.sqrt(qu_lanina**2 + qv_lanina**2)
qflux_neutral = np.sqrt(qu_neutral**2 + qv_neutral**2)
qflux_diff = np.sqrt(qu_diff**2 + qv_diff**2)

# ── Significance (Welch's t-test) ───────────────────────────────────
pval_qu = np.zeros((len(lat), len(lon)))
pval_qv = np.zeros((len(lat), len(lon)))
for j in range(len(lat)):
    for k in range(len(lon)):
        _, pval_qu[j, k] = stats.ttest_ind(
            qu_annual[elnino_idx, j, k], qu_annual[lanina_idx, j, k],
            equal_var=False)
        _, pval_qv[j, k] = stats.ttest_ind(
            qv_annual[elnino_idx, j, k], qv_annual[lanina_idx, j, k],
            equal_var=False)

sig_mask = (pval_qu < 0.05) | (pval_qv < 0.05)

# Land-only significance
qflux_clim = np.sqrt(np.nanmean(qu_annual, axis=0)**2 +
                      np.nanmean(qv_annual, axis=0)**2)
land_threshold = np.nanpercentile(qflux_clim[qflux_clim > 0], 25)
land_mask = qflux_clim > land_threshold
n_sig_land = (sig_mask & land_mask).sum()
print(f"Significant land grid points: {n_sig_land} / {land_mask.sum()} "
      f"({100 * n_sig_land / land_mask.sum():.1f}%)")

# ── Scale to g/kg * m/s ─────────────────────────────────────────────
scale = 1000
qu_elnino_s = qu_elnino * scale
qv_elnino_s = qv_elnino * scale
qu_lanina_s = qu_lanina * scale
qv_lanina_s = qv_lanina * scale
qu_neutral_s = qu_neutral * scale
qv_neutral_s = qv_neutral * scale
qu_diff_s = qu_diff * scale
qv_diff_s = qv_diff * scale

# ── 4-panel figure ───────────────────────────────────────────────────
skip = 8
extent = [-80, -30, -20, 10]
vmax_abs = max(np.nanpercentile(qflux_elnino * scale, 98),
               np.nanpercentile(qflux_lanina * scale, 98),
               np.nanpercentile(qflux_neutral * scale, 98))
vmax_diff = np.nanpercentile(np.abs(qflux_diff * scale), 98)
LON, LAT = np.meshgrid(lon, lat)

fig, axes = plt.subplots(2, 2, figsize=(13, 10),
                          subplot_kw={"projection": ccrs.PlateCarree()})

panels = [
    (axes[0, 0], qu_elnino_s, qv_elnino_s, qflux_elnino * scale,
     f"(a) El Ni\u00f1o (n={len(elnino_idx)})", "YlGnBu", 0, vmax_abs, False),
    (axes[0, 1], qu_lanina_s, qv_lanina_s, qflux_lanina * scale,
     f"(b) La Ni\u00f1a (n={len(lanina_idx)})", "YlGnBu", 0, vmax_abs, False),
    (axes[1, 0], qu_neutral_s, qv_neutral_s, qflux_neutral * scale,
     f"(c) Neutral (n={len(neutral_idx)})", "YlGnBu", 0, vmax_abs, False),
    (axes[1, 1], qu_diff_s, qv_diff_s, qflux_diff * scale,
     "(d) El Ni\u00f1o \u2212 La Ni\u00f1a", "RdBu_r", None, vmax_diff, True),
]

for ax, qu_p, qv_p, mag, title, cmap, vmin_p, vmax_p, is_diff in panels:
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=1.4, color="#000000", zorder=6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.7, linestyle="--",
                   color="#444444", zorder=6)
    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", alpha=0.3, zorder=1)

    if is_diff:
        cf = ax.pcolormesh(LON, LAT, qu_diff_s, cmap="RdBu_r",
                           vmin=-vmax_p, vmax=vmax_p,
                           transform=ccrs.PlateCarree(), shading="auto")
    else:
        cf = ax.pcolormesh(LON, LAT, mag, cmap=cmap, vmin=0, vmax=vmax_p,
                           transform=ccrs.PlateCarree(), shading="auto")

    # Normalized direction arrows
    qu_sub = qu_p[::skip, ::skip]
    qv_sub = qv_p[::skip, ::skip]
    mag_sub = np.sqrt(qu_sub**2 + qv_sub**2)
    mag_sub = np.where(mag_sub < 1e-10, 1e-10, mag_sub)
    flux_threshold = np.nanpercentile(mag_sub[mag_sub > 0], 10)
    qu_norm = np.where(mag_sub > flux_threshold, qu_sub / mag_sub, np.nan)
    qv_norm = np.where(mag_sub > flux_threshold, qv_sub / mag_sub, np.nan)
    arrow_color = "#4a0082" if is_diff else "#1a5276"
    ax.quiver(LON[::skip, ::skip], LAT[::skip, ::skip], qu_norm, qv_norm,
              transform=ccrs.PlateCarree(), scale=28, scale_units="width",
              width=0.003, headwidth=4, headlength=3.5, headaxislength=3,
              color=arrow_color, alpha=0.65, zorder=4)

    if is_diff:
        ax.contourf(LON, LAT, sig_mask.astype(float), levels=[0.5, 1.5],
                    hatches=["..."], colors="none",
                    transform=ccrs.PlateCarree(), zorder=4)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray",
                       alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([-80, -70, -60, -50, -40, -30])
    gl.ylocator = mticker.FixedLocator([-20, -15, -10, -5, 0, 5, 10])
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")

    cbar = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.06,
                         shrink=0.80, aspect=25)
    if is_diff:
        cbar.set_label("Zonal moisture flux diff. (g kg$^{-1}$ m s$^{-1}$)",
                        fontsize=8)
    else:
        cbar.set_label("|Moisture flux| (g kg$^{-1}$ m s$^{-1}$)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

axes[1, 1].annotate("Arrows: flow direction\nShading: flux magnitude",
                     xy=(0.98, 0.02), xycoords="axes fraction",
                     fontsize=7, ha="right", va="bottom",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                               alpha=0.85))

fig.suptitle("February 900 hPa Moisture Flux Composites by ENSO Phase "
             "(1979\u20132023)", fontsize=13, fontweight="bold", y=0.98)
plt.subplots_adjust(hspace=0.25, wspace=0.12, bottom=0.04, top=0.93)

outpath = FIG_DIR / "fig01_enso_moisture_flux_composites.png"
fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {outpath}")
plt.close()

ds.close()
print("Script 02 complete.")
