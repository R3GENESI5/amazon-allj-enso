"""
Moisture flux convergence transect -- coast to interior.

Tests the rain-out hypothesis: during El Nino the ALLJ strengthens but
coastal precipitation also increases. Does net moisture reaching the
interior increase or decrease?

Computes q*u at 900 hPa along longitude transects (48W--64W), averaged
over 0--5S, composited by ENSO phase, and compares with CHIRPS
precipitation. Also computes flux retention ratio (64W / 48W).

Inputs:  data/oni_classification.csv
         data/era5/era5_monthly_means_900hpa_feb_1979_2023.nc
         data/chirps/chirps-v2.0.YYYY.monthly.nc
Outputs: figures/fig05_moisture_budget_transect.png

Reference: Section 3.5, Figure 5
"""

import os
import calendar
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

# Colours
C_ELNINO = "#D7263D"
C_LANINA = "#2166AC"
C_NEUTRAL = "#888888"

TRANSECT_LONS = [-48, -52, -56, -60, -64]
LAT_BAND = (0, -5)
CHIRPS_BAND_WIDTH = 4

# ── Load ONI ─────────────────────────────────────────────────────────
oni = pd.read_csv(DATA_DIR / "oni_classification.csv")

# ── ERA5 900 hPa ─────────────────────────────────────────────────────
ds = xr.open_dataset(DATA_DIR / "era5" /
                     "era5_monthly_means_900hpa_feb_1979_2023.nc")
u = ds["u"].squeeze("pressure_level")
q = ds["q"].squeeze("pressure_level")
lat = ds["latitude"].values
lon = ds["longitude"].values

times = pd.DatetimeIndex(ds["valid_time"].values)
years_era5 = times.year
unique_years = sorted(set(years_era5))
n_years = len(unique_years)

qu_all = (q * u).values
qu_annual = np.zeros((n_years, len(lat), len(lon)))
for i, yr in enumerate(unique_years):
    mask = years_era5 == yr
    qu_annual[i] = np.nanmean(qu_all[mask], axis=0)

# ── Extract flux at transects ────────────────────────────────────────
lat_mask = (lat <= max(LAT_BAND)) & (lat >= min(LAT_BAND))
flux_by_transect = {}
for tlon in TRANSECT_LONS:
    ilon = np.argmin(np.abs(lon - tlon))
    qu_transect = qu_annual[:, lat_mask, ilon]
    flux_by_transect[tlon] = -np.nanmean(qu_transect, axis=1)

# ── CHIRPS precipitation ─────────────────────────────────────────────
precip_by_transect = {t: [] for t in TRANSECT_LONS}
precip_years_list = []

for yr in range(1981, 2024):
    fpath = DATA_DIR / "chirps" / f"chirps-v2.0.{yr}.monthly.nc"
    if not fpath.exists():
        continue
    try:
        dsc = xr.open_dataset(fpath)
        feb_precip = dsc["precip"].isel(time=1)
        clat = dsc["latitude"].values
        clon = dsc["longitude"].values
        clat_mask = (clat <= max(LAT_BAND)) & (clat >= min(LAT_BAND))
        feb_days = 29 if calendar.isleap(yr) else 28
        for tlon in TRANSECT_LONS:
            half = CHIRPS_BAND_WIDTH / 2
            clon_mask = (clon >= tlon - half) & (clon <= tlon + half)
            band = feb_precip.values[np.ix_(clat_mask, clon_mask)]
            precip_by_transect[tlon].append(np.nanmean(band) / feb_days)
        precip_years_list.append(yr)
        dsc.close()
    except Exception:
        pass

for tlon in TRANSECT_LONS:
    precip_by_transect[tlon] = np.array(precip_by_transect[tlon])

# ── Composite by ENSO phase ─────────────────────────────────────────
def composite_by_phase(data_array, data_years, oni_df):
    results = {}
    for phase in ["El Nino", "La Nina", "Neutral"]:
        phase_years = oni_df[oni_df["phase"] == phase]["year"].values
        mask = np.isin(data_years, phase_years)
        results[phase] = data_array[mask]
    return results

flux_composites = {}
for tlon in TRANSECT_LONS:
    flux_composites[tlon] = composite_by_phase(
        flux_by_transect[tlon], np.array(unique_years), oni)

precip_composites = {}
for tlon in TRANSECT_LONS:
    precip_composites[tlon] = composite_by_phase(
        precip_by_transect[tlon], np.array(precip_years_list), oni)

# Flux retention ratio
retention = flux_by_transect[-64] / flux_by_transect[-48]
retention_composites = composite_by_phase(
    retention, np.array(unique_years), oni)

ret_en = retention_composites["El Nino"]
ret_ln = retention_composites["La Nina"]
_, p_val = stats.ttest_ind(ret_en, ret_ln, equal_var=False)

print(f"Retention (64W/48W): El Nino={ret_en.mean():.3f}, "
      f"La Nina={ret_ln.mean():.3f}, p={p_val:.4f}")

# ── Figure ───────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.5))

phases = ["El Nino", "La Nina", "Neutral"]
colors = {"El Nino": C_ELNINO, "La Nina": C_LANINA, "Neutral": C_NEUTRAL}
labels = {"El Nino": "El Ni\u00f1o", "La Nina": "La Ni\u00f1a",
          "Neutral": "Neutral"}
markers = {"El Nino": "o", "La Nina": "s", "Neutral": "D"}
x_pos = np.array([abs(t) for t in TRANSECT_LONS])
offsets = {"El Nino": -0.3, "La Nina": 0.0, "Neutral": 0.3}

# Panel (a): Moisture flux
for phase in phases:
    means = [flux_composites[t][phase].mean() * 1e3 for t in TRANSECT_LONS]
    ci = [1.96 * flux_composites[t][phase].std() /
          np.sqrt(len(flux_composites[t][phase])) * 1e3
          for t in TRANSECT_LONS]
    ax1.errorbar(x_pos + offsets[phase], means, yerr=ci,
                 color=colors[phase], marker=markers[phase], markersize=5,
                 linewidth=1.5, capsize=3, capthick=1, label=labels[phase])

ax1.set_xlabel("Longitude (\u00b0W)")
ax1.set_ylabel("Westward moisture flux\n"
               "(q\u00b7u, 10$^{-3}$ kg kg$^{-1}$ m s$^{-1}$)")
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f"{x}\u00b0W" for x in x_pos])
ax1.invert_xaxis()
ax1.legend(loc="upper left", frameon=False)
ax1.set_title("(a) 900 hPa moisture flux (Feb)", fontweight="bold")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.yaxis.grid(True, alpha=0.3, linewidth=0.5)

# Panel (b): CHIRPS precipitation
for phase in phases:
    means = [precip_composites[t][phase].mean() for t in TRANSECT_LONS]
    ci = [1.96 * precip_composites[t][phase].std() /
          np.sqrt(len(precip_composites[t][phase]))
          for t in TRANSECT_LONS]
    ax2.errorbar(x_pos + offsets[phase], means, yerr=ci,
                 color=colors[phase], marker=markers[phase], markersize=5,
                 linewidth=1.5, capsize=3, capthick=1, label=labels[phase])

ax2.set_xlabel("Longitude (\u00b0W)")
ax2.set_ylabel("Precipitation (mm day$^{-1}$)")
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f"{x}\u00b0W" for x in x_pos])
ax2.invert_xaxis()
ax2.legend(loc="upper right", frameon=False)
ax2.set_title("(b) CHIRPS precipitation (Feb)", fontweight="bold")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.yaxis.grid(True, alpha=0.3, linewidth=0.5)

plt.tight_layout()
outpath = FIG_DIR / "fig05_moisture_budget_transect.png"
fig.savefig(outpath, dpi=300, facecolor="white", bbox_inches="tight")
print(f"Saved: {outpath}")
plt.close()
print("Script 06 complete.")
