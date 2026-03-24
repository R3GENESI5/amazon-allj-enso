"""
Deforestation-era precipitation and TWS analysis.

4-panel figure comparing the Arc of Deforestation vs intact interior:
  (a) February CHIRPS precipitation time series with ENSO shading
  (b) ONI vs February precipitation scatter
  (c) ENSO-adjusted secular precipitation trend
  (d) GRACE TWS comparison with linear trends

Inputs:  data/oni_classification.csv
         data/chirps/chirps-v2.0.YYYY.monthly.nc
         data/grace/GRCTellus.JPL.*.nc
Outputs: figures/fig06_deforestation_signal.png

Reference: Section 3.6, Figure 6
"""

import numpy as np
import netCDF4
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import stats
import statsmodels.api as sm
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CHIRPS_DIR = DATA_DIR / "chirps"

# Find GRACE file
grace_files = list((DATA_DIR / "grace").glob("GRCTellus.JPL.*.nc"))
if not grace_files:
    raise FileNotFoundError("No GRACE file found")
GRACE_FILE = str(grace_files[0])

ONI_FILE = DATA_DIR / "oni_classification.csv"

# Region definitions
ARC_LAT, ARC_LON = (-12, -5), (-55, -45)
INT_LAT, INT_LON = (-5, 0), (-65, -55)

C_ARC, C_INT = "#C45B28", "#2E7D32"
C_NINO, C_NINA = "#FFCDD2", "#BBDEFB"

# ── 1. Load ONI ──────────────────────────────────────────────────────
oni_df = pd.read_csv(ONI_FILE)
oni_dict = dict(zip(oni_df["year"], oni_df["oni"]))
phase_dict = dict(zip(oni_df["year"], oni_df["phase"]))

# ── 2. CHIRPS February precipitation ────────────────────────────────
def extract_region_mean(nc, var_name, time_idx, lat_bounds, lon_bounds):
    lats = nc.variables["latitude"][:]
    lons = nc.variables["longitude"][:]
    lat_mask = (lats >= lat_bounds[0]) & (lats <= lat_bounds[1])
    lon_mask = (lons >= lon_bounds[0]) & (lons <= lon_bounds[1])
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    data = nc.variables[var_name][time_idx,
                                  lat_idx[0]:lat_idx[-1]+1,
                                  lon_idx[0]:lon_idx[-1]+1]
    data = np.ma.filled(data, np.nan)
    return float(np.nanmean(data))

feb_arc, feb_int, valid_years = [], [], []
for yr in range(1981, 2025):
    fpath = CHIRPS_DIR / f"chirps-v2.0.{yr}.monthly.nc"
    try:
        nc = netCDF4.Dataset(str(fpath), "r")
        feb_arc.append(extract_region_mean(nc, "precip", 1, ARC_LAT, ARC_LON))
        feb_int.append(extract_region_mean(nc, "precip", 1, INT_LAT, INT_LON))
        valid_years.append(yr)
        nc.close()
    except Exception:
        pass

years_arr = np.array(valid_years)
feb_arc = np.array(feb_arc)
feb_int = np.array(feb_int)
print(f"CHIRPS: {len(valid_years)} years loaded")

# Running means
def running_mean(data, window=10):
    rm = np.full_like(data, np.nan, dtype=float)
    half = window // 2
    for i in range(half, len(data) - half):
        rm[i] = np.nanmean(data[i - half:i + half])
    return rm

rm_arc = running_mean(feb_arc, 10)
rm_int = running_mean(feb_int, 10)

# ── 3. ENSO-adjusted trend ──────────────────────────────────────────
oni_vals = np.array([oni_dict.get(yr, np.nan) for yr in valid_years])
mask = ~np.isnan(oni_vals)
time_var = years_arr[mask] - years_arr[mask].mean()
X = np.column_stack([time_var, oni_vals[mask]])
X = sm.add_constant(X)
model = sm.OLS(feb_arc[mask], X).fit()

enso_contribution = model.params[2] * oni_vals[mask]
residual_precip = feb_arc[mask] - enso_contribution
slope_resid, intercept_resid, _, p_resid, _ = stats.linregress(
    years_arr[mask], residual_precip)

# ── 4. GRACE TWS ────────────────────────────────────────────────────
grace = netCDF4.Dataset(GRACE_FILE, "r")
g_lats = grace.variables["lat"][:]
g_lons = grace.variables["lon"][:]
from netCDF4 import num2date
g_time_var = grace.variables["time"]
g_times = num2date(g_time_var[:], units=g_time_var.units,
                   calendar=getattr(g_time_var, "calendar", "standard"))
grace_time = pd.to_datetime([str(t) for t in g_times])

def to360(lon):
    return lon % 360

def grace_region_ts(lwe, sf, lats, lons, lat_b, lon_b_360):
    lat_m = (lats >= lat_b[0]) & (lats <= lat_b[1])
    lon_m = (lons >= lon_b_360[0]) & (lons <= lon_b_360[1])
    li = np.where(lat_m)[0]
    lo = np.where(lon_m)[0]
    ts = []
    for t in range(lwe.shape[0]):
        d = lwe[t, li[0]:li[-1]+1, lo[0]:lo[-1]+1]
        s = sf[li[0]:li[-1]+1, lo[0]:lo[-1]+1]
        d = np.ma.filled(d, np.nan) * np.ma.filled(s, np.nan)
        ts.append(float(np.nanmean(d)))
    return np.array(ts)

lwe = grace.variables["lwe_thickness"]
sf = grace.variables["scale_factor"]
arc_lon_360 = (to360(ARC_LON[0]), to360(ARC_LON[1]))
int_lon_360 = (to360(INT_LON[0]), to360(INT_LON[1]))
tws_arc = grace_region_ts(lwe, sf, g_lats, g_lons, ARC_LAT, arc_lon_360)
tws_int = grace_region_ts(lwe, sf, g_lats, g_lons, INT_LAT, int_lon_360)

grace_years_dec = np.array([t.year + t.month / 12.0 for t in grace_time])
va = ~np.isnan(tws_arc)
slope_a, int_a, _, p_a, se_a = stats.linregress(grace_years_dec[va], tws_arc[va])
vi = ~np.isnan(tws_int)
slope_i, int_i, _, p_i, se_i = stats.linregress(grace_years_dec[vi], tws_int[vi])
z_diff = (slope_a - slope_i) / np.sqrt(se_a**2 + se_i**2)
p_diff = 2 * (1 - stats.norm.cdf(abs(z_diff)))
grace.close()

# ── 5. Figure ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 11), facecolor="white")
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.28,
                       left=0.08, right=0.96, top=0.95, bottom=0.07)

# (a) February CHIRPS time series
ax1 = fig.add_subplot(gs[0, 0])
for yr in valid_years:
    phase = phase_dict.get(yr, "Neutral")
    if phase == "El Nino":
        ax1.axvspan(yr - 0.4, yr + 0.4, color=C_NINO, alpha=0.5, zorder=0)
    elif phase == "La Nina":
        ax1.axvspan(yr - 0.4, yr + 0.4, color=C_NINA, alpha=0.5, zorder=0)

ax1.plot(years_arr, feb_arc, "o", color=C_ARC, markersize=3.5, alpha=0.5)
ax1.plot(years_arr, feb_int, "s", color=C_INT, markersize=3.5, alpha=0.5)
ax1.plot(years_arr, rm_arc, color=C_ARC, linewidth=2.5, zorder=3,
         label="Arc of deforestation")
ax1.plot(years_arr, rm_int, color=C_INT, linewidth=2.5, zorder=3,
         label="Intact interior")
ax1.set_xlabel("Year")
ax1.set_ylabel("February precipitation (mm)")
ax1.set_title("(a) February precipitation: arc vs interior",
              fontweight="bold", loc="left")
ax1.legend(loc="lower left", fontsize=7.5)
ax1.set_xlim(1980, 2025)

# (b) Scatter ONI vs Feb precip (arc)
ax2 = fig.add_subplot(gs[0, 1])
oni_p = oni_vals[mask]
sc = ax2.scatter(oni_p, feb_arc[mask], c=years_arr[mask], cmap="YlOrBr",
                 s=45, edgecolors="k", linewidths=0.3, zorder=3)
cbar = plt.colorbar(sc, ax=ax2, shrink=0.8, pad=0.02)
cbar.set_label("Year", fontsize=9)
slope_oni, int_oni, r_oni, p_oni, _ = stats.linregress(oni_p, feb_arc[mask])
oni_range = np.linspace(oni_p.min() - 0.1, oni_p.max() + 0.1, 100)
ax2.plot(oni_range, int_oni + slope_oni * oni_range, "k--", lw=1.5, alpha=0.7)
ax2.set_xlabel("ONI (DJF)")
ax2.set_ylabel("February precipitation (mm)")
ax2.set_title(f"(b) ENSO vs Feb. precip (arc)\n"
              f"r = {r_oni:.2f}, p = {p_oni:.3f}",
              fontweight="bold", loc="left")

# (c) Residual trend
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(years_arr[mask], residual_precip, "o", color=C_ARC, markersize=4,
         alpha=0.5)
ax3.plot(years_arr[mask], running_mean(residual_precip, 10), color=C_ARC,
         linewidth=2.5, label="10-yr running mean")
trend_line = intercept_resid + slope_resid * years_arr[mask]
ax3.plot(years_arr[mask], trend_line, "k--", lw=1.5, alpha=0.7,
         label=f"Trend: {slope_resid:+.2f} mm/yr")
sig = "***" if p_resid < 0.001 else "**" if p_resid < 0.01 else \
      "*" if p_resid < 0.05 else "n.s."
ax3.set_xlabel("Year")
ax3.set_ylabel("ENSO-adjusted Feb. precip (mm)")
ax3.set_title(f"(c) Secular trend after ENSO removal\n"
              f"Trend: {slope_resid:+.2f} mm/yr (p = {p_resid:.3f}, {sig})",
              fontweight="bold", loc="left")
ax3.legend(fontsize=8, loc="best")
ax3.set_xlim(1980, 2025)

# (d) GRACE TWS
ax4 = fig.add_subplot(gs[1, 1])
s_arc = pd.Series(tws_arc, index=grace_time).rolling(
    12, center=True, min_periods=6).mean()
s_int = pd.Series(tws_int, index=grace_time).rolling(
    12, center=True, min_periods=6).mean()
ax4.plot(grace_time, tws_arc, color=C_ARC, lw=0.6, alpha=0.35)
ax4.plot(grace_time, tws_int, color=C_INT, lw=0.6, alpha=0.35)
ax4.plot(s_arc.index, s_arc.values, color=C_ARC, lw=2.5,
         label=f"Arc ({slope_a:+.2f} cm/yr, p={p_a:.3f})")
ax4.plot(s_int.index, s_int.values, color=C_INT, lw=2.5,
         label=f"Interior ({slope_i:+.2f} cm/yr, p={p_i:.3f})")
ax4.plot(grace_time, int_a + slope_a * grace_years_dec, "--",
         color=C_ARC, lw=1.2, alpha=0.7)
ax4.plot(grace_time, int_i + slope_i * grace_years_dec, "--",
         color=C_INT, lw=1.2, alpha=0.7)
ax4.axhline(0, color="gray", lw=0.5, ls=":")
ax4.set_xlabel("Year")
ax4.set_ylabel("TWS anomaly (cm eq. water)")
ds_sig = "***" if p_diff < 0.001 else "**" if p_diff < 0.01 else \
         "*" if p_diff < 0.05 else "n.s."
ax4.set_title(f"(d) GRACE TWS: arc vs interior\n"
              f"Slope difference: {ds_sig} (p = {p_diff:.3f})",
              fontweight="bold", loc="left")
ax4.legend(fontsize=8, loc="lower left")

outpath = FIG_DIR / "fig06_deforestation_signal.png"
plt.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {outpath}")
plt.close()
print("Script 07 complete.")
