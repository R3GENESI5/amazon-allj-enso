"""
Nocturnal moisture flux priming of next-day convection.

Computes day-by-day pairs of nocturnal mean moisture flux magnitude
(00--06 UTC, day N) and next-afternoon CAPE (15--21 UTC, day N+1)
for the 10 key ENSO Februaries, then plots:
  (a) Scatter plot with regression lines by ENSO phase
  (b) Bar chart of Pearson correlation coefficients

Inputs:  data/era5/era5_pl_YYYY_feb.nc
         data/era5/era5_sfc_YYYY_feb.nc
Outputs: figures/fig09_priming_correlation.png

Reference: Section 3.9, Figure 9
"""

import warnings
warnings.filterwarnings("ignore")

import os
import zipfile
import tempfile
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "era5"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ELNINO_YEARS = [1983, 1992, 1998, 2010, 2016]
LANINA_YEARS = [1989, 1999, 2000, 2008, 2011]

C_ELNINO = "#D4443B"
C_LANINA = "#2E86AB"
C_GREY = "#888888"

# Amazon interior box
LAT_S, LAT_N = -5.0, 0.0
LON_W, LON_E = -60.0, -50.0

# ── Loaders ──────────────────────────────────────────────────────────
def load_sfc(year):
    fpath = DATA_DIR / f"era5_sfc_{year}_feb.nc"
    with open(fpath, "rb") as f:
        magic = f.read(4)
    if magic == b"PK\x03\x04":
        tmpdir = tempfile.mkdtemp()
        with zipfile.ZipFile(fpath) as zf:
            zf.extractall(tmpdir)
        ds_i = xr.open_dataset(os.path.join(
            tmpdir, "data_stream-oper_stepType-instant.nc"))
        ds_a = xr.open_dataset(os.path.join(
            tmpdir, "data_stream-oper_stepType-accum.nc"))
        return xr.merge([ds_i, ds_a])
    return xr.open_dataset(fpath)

def load_pl(year):
    return xr.open_dataset(DATA_DIR / f"era5_pl_{year}_feb.nc")

def sel_amazon(ds, var):
    da = ds[var]
    lat_name = "latitude" if "latitude" in da.dims else "lat"
    lon_name = "longitude" if "longitude" in da.dims else "lon"
    return da.sel({lat_name: slice(LAT_N, LAT_S),
                   lon_name: slice(LON_W, LON_E)})


# ── Compute priming pairs ────────────────────────────────────────────
def compute_priming_pairs(yr):
    ds_pl = load_pl(yr).sel(pressure_level=900.0, method="nearest")
    lat_name, lon_name = "latitude", "longitude"

    q = ds_pl["q"].sel({lat_name: slice(LAT_N, LAT_S),
                        lon_name: slice(LON_W, LON_E)})
    u = ds_pl["u"].sel({lat_name: slice(LAT_N, LAT_S),
                        lon_name: slice(LON_W, LON_E)})
    v = ds_pl["v"].sel({lat_name: slice(LAT_N, LAT_S),
                        lon_name: slice(LON_W, LON_E)})

    mf_mag = np.sqrt((q * u)**2 + (q * v)**2)
    mf_mean = mf_mag.mean(dim=[lat_name, lon_name])

    cape = sel_amazon(load_sfc(yr), "cape")
    cape_mean = cape.mean(dim=[lat_name, lon_name])

    time_coord = "valid_time"
    hours_mf = mf_mean[time_coord].dt.hour.values
    hours_cape = cape_mean[time_coord].dt.hour.values
    days_mf = mf_mean[time_coord].dt.day.values
    days_cape = cape_mean[time_coord].dt.day.values
    mf_vals = mf_mean.values
    cape_vals = cape_mean.values

    unique_days = sorted(set(days_mf))
    nightly_flux, nextday_cape = [], []
    for i, day in enumerate(unique_days[:-1]):
        next_day = unique_days[i + 1]
        mask_night = (days_mf == day) & (hours_mf >= 0) & (hours_mf <= 6)
        mask_aft = (days_cape == next_day) & (hours_cape >= 15) & (hours_cape <= 21)
        if mask_night.sum() > 0 and mask_aft.sum() > 0:
            nightly_flux.append(np.nanmean(mf_vals[mask_night]))
            nextday_cape.append(np.nanmean(cape_vals[mask_aft]))
    return np.array(nightly_flux), np.array(nextday_cape)

# Collect pairs
elnino_flux, elnino_cape = [], []
lanina_flux, lanina_cape = [], []

for yr in ELNINO_YEARS:
    print(f"  El Nino {yr}...")
    f, c = compute_priming_pairs(yr)
    elnino_flux.append(f)
    elnino_cape.append(c)

for yr in LANINA_YEARS:
    print(f"  La Nina {yr}...")
    f, c = compute_priming_pairs(yr)
    lanina_flux.append(f)
    lanina_cape.append(c)

elnino_flux = np.concatenate(elnino_flux) * 1000  # to g/kg m/s
elnino_cape = np.concatenate(elnino_cape)
lanina_flux = np.concatenate(lanina_flux) * 1000
lanina_cape = np.concatenate(lanina_cape)
all_flux = np.concatenate([elnino_flux, lanina_flux])
all_cape = np.concatenate([elnino_cape, lanina_cape])

def corr_with_p(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    return stats.pearsonr(x[mask], y[mask])

r_en, p_en = corr_with_p(elnino_flux, elnino_cape)
r_ln, p_ln = corr_with_p(lanina_flux, lanina_cape)
r_all, p_all = corr_with_p(all_flux, all_cape)

print(f"\nEl Nino: r={r_en:.3f}, p={p_en:.1e}")
print(f"La Nina: r={r_ln:.3f}, p={p_ln:.1e}")
print(f"All:     r={r_all:.3f}, p={p_all:.1e}")

# ── Figure ───────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 4.2),
                                gridspec_kw={"width_ratios": [1.6, 1]})

# (a) Scatter
ax1.scatter(elnino_flux, elnino_cape, c=C_ELNINO, s=12, alpha=0.35,
            edgecolors="none", label=f"El Ni\u00f1o (r={r_en:.2f})", zorder=3)
ax1.scatter(lanina_flux, lanina_cape, c=C_LANINA, s=12, alpha=0.35,
            edgecolors="none", label=f"La Ni\u00f1a (r={r_ln:.2f})", zorder=3)

for fx, fy, color in [(elnino_flux, elnino_cape, C_ELNINO),
                       (lanina_flux, lanina_cape, C_LANINA)]:
    mask = np.isfinite(fx) & np.isfinite(fy)
    slope, intercept = np.polyfit(fx[mask], fy[mask], 1)
    xfit = np.linspace(np.nanmin(fx), np.nanmax(fx), 100)
    ax1.plot(xfit, slope * xfit + intercept, color=color, lw=1.5,
             ls="--", zorder=4)

ax1.set_xlabel("Nocturnal moisture flux, day N\n"
               "(g kg$^{-1}$ m s$^{-1}$, 00\u201306 UTC)")
ax1.set_ylabel("Afternoon CAPE, day N+1 (J kg$^{-1}$, 15\u201321 UTC)")
ax1.set_title("(a) Priming: nocturnal flux \u2192 next-day CAPE",
              fontsize=9, fontweight="bold", loc="left")
ax1.legend(loc="upper left", fontsize=7, markerscale=2)

txt = (f"El Ni\u00f1o: r = {r_en:.3f}, p = {p_en:.1e}\n"
       f"La Ni\u00f1a: r = {r_ln:.3f}, p = {p_ln:.1e}")
ax1.text(0.98, 0.05, txt, transform=ax1.transAxes, fontsize=6.5,
         ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                   edgecolor="#cccccc", alpha=0.9))

# (b) Bar chart
categories = ["El Ni\u00f1o", "La Ni\u00f1a", "All years"]
r_values = [r_en, r_ln, r_all]
p_values = [p_en, p_ln, p_all]
bar_colors = [C_ELNINO, C_LANINA, C_GREY]

ax2.bar(categories, r_values, color=bar_colors, width=0.55,
        edgecolor="#333333", linewidth=0.5, zorder=3)

for i, (rv, pv) in enumerate(zip(r_values, p_values)):
    star = "***" if pv < 0.001 else "**" if pv < 0.01 else \
           "*" if pv < 0.05 else "n.s."
    offset = 0.015 if rv >= 0 else -0.015
    va = "bottom" if rv >= 0 else "top"
    ax2.text(i, rv + offset, f"{rv:.3f}\n{star}", ha="center", va=va,
             fontsize=7, fontweight="bold")

ax2.set_ylabel("Pearson r")
ax2.set_title("(b) Priming correlation strength", fontsize=9,
              fontweight="bold", loc="left")
ax2.axhline(0, color="#333333", lw=0.5, zorder=1)
ax2.set_ylim(min(r_values) - 0.12, max(r_values) + 0.12)

fig.subplots_adjust(top=0.88, wspace=0.35, bottom=0.15, left=0.10, right=0.96)
fig.suptitle("Nocturnal moisture flux priming of next-day convection",
             fontsize=11, fontweight="bold", y=0.98)

outpath = FIG_DIR / "fig09_priming_correlation.png"
fig.savefig(outpath, dpi=300, facecolor="white", bbox_inches="tight")
plt.close(fig)
print(f"Saved: {outpath}")
print("Script 10 complete.")
