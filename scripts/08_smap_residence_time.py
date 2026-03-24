"""
SMAP L4 root-zone soil moisture residence time analysis.

Compares soil moisture dynamics between the Arc of Deforestation and
the Intact Interior: seasonal cycle, lag-1 autocorrelation, post-wet-
season dry-down rates, and annual dry-season minimum trends.

Inputs:  data/smap/smap_l4_rootzone_sm_monthly_amazon_2015_2024.nc
Outputs: figures/fig07_smap_residence.png

Reference: Section 3.7, Figure 7
"""

import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SMAP_PATH = DATA_DIR / "smap" / "smap_l4_rootzone_sm_monthly_amazon_2015_2024.nc"

# Region definitions
ARC_LAT, ARC_LON = (-12, -5), (-55, -45)
INT_LAT, INT_LON = (-5, 0), (-65, -55)

# ── Load data ────────────────────────────────────────────────────────
ds = nc.Dataset(str(SMAP_PATH))
time_raw = ds.variables["time"][:]
lat = ds.variables["lat"][:]
lon = ds.variables["lon"][:]
sm = ds.variables["sm_rootzone"][:]
ds.close()

base_date = datetime(2000, 1, 1)
dates = np.array([base_date + timedelta(days=float(d)) for d in time_raw])
sm = np.where((sm < 0) | (sm > 1), np.nan, sm)

# Area weights
lat2d = np.broadcast_to(lat[np.newaxis, :, np.newaxis], sm.shape)
weights = np.cos(np.deg2rad(lat2d))

def regional_mean(sm, lat, lon, weights, lat_bounds, lon_bounds):
    lat_mask = (lat >= lat_bounds[0]) & (lat <= lat_bounds[1])
    lon_mask = (lon >= lon_bounds[0]) & (lon <= lon_bounds[1])
    region = sm[:, lat_mask, :][:, :, lon_mask]
    w = weights[:, lat_mask, :][:, :, lon_mask]
    w = np.where(np.isnan(region), 0, w)
    return np.nansum(region * w, axis=(1, 2)) / np.nansum(w, axis=(1, 2))

arc_ts = regional_mean(sm, lat, lon, weights, ARC_LAT, ARC_LON)
int_ts = regional_mean(sm, lat, lon, weights, INT_LAT, INT_LON)

# ── Residence time (e-folding dry-down) ──────────────────────────────
def compute_residence_times(ts):
    dsm = np.diff(ts)
    residence_times, decay_rates, peak_sm = [], [], []
    i = 0
    while i < len(dsm):
        if dsm[i] > 0:
            pk_idx = i + 1
            pk_val = ts[pk_idx]
            j = pk_idx
            while j < len(ts) - 1 and ts[j + 1] < ts[j]:
                j += 1
            dd_len = j - pk_idx
            if dd_len >= 2:
                t_months = np.arange(dd_len + 1, dtype=float)
                sm_decay = ts[pk_idx:pk_idx + dd_len + 1]
                sm_norm = sm_decay / pk_val
                with np.errstate(divide="ignore", invalid="ignore"):
                    ln_sm = np.log(sm_norm)
                valid = np.isfinite(ln_sm)
                if valid.sum() >= 2:
                    slope, _, _, _, _ = stats.linregress(
                        t_months[valid], ln_sm[valid])
                    if slope < 0:
                        tau = -1.0 / slope
                        if 0.5 < tau < 24:
                            residence_times.append(tau)
                            decay_rates.append(-slope)
                            peak_sm.append(pk_val)
            i = j if j > pk_idx else i + 1
        else:
            i += 1
    return np.array(residence_times), np.array(decay_rates), np.array(peak_sm)

arc_tau, arc_rate, arc_peak = compute_residence_times(arc_ts)
int_tau, int_rate, int_peak = compute_residence_times(int_ts)

t_stat, p_val = stats.ttest_ind(arc_tau, int_tau, equal_var=False)
print(f"Arc residence time: {np.mean(arc_tau):.2f} +/- {np.std(arc_tau):.2f} months")
print(f"Intact residence time: {np.mean(int_tau):.2f} +/- {np.std(int_tau):.2f} months")
print(f"Welch t = {t_stat:.3f}, p = {p_val:.4f}")

# SM anomaly decay segments for scatter
def get_decay_segments(ts):
    mean_sm = np.nanmean(ts)
    anom = ts - mean_sm
    dsm = np.diff(ts)
    peaks_anom, one_month_decay = [], []
    for i in range(len(dsm)):
        if dsm[i] > 0 and i + 1 < len(dsm):
            if dsm[i + 1] < 0:
                peaks_anom.append(anom[i + 1])
                one_month_decay.append(-dsm[i + 1])
    return np.array(peaks_anom), np.array(one_month_decay)

arc_pk_anom, arc_1m_decay = get_decay_segments(arc_ts)
int_pk_anom, int_1m_decay = get_decay_segments(int_ts)

# ── Figure ───────────────────────────────────────────────────────────
clr_arc = "#D95F02"
clr_int = "#1B9E77"

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5),
                         gridspec_kw={"width_ratios": [2.2, 1, 1.2]})

# (a) Time series
ax = axes[0]
ax.plot(dates, arc_ts, color=clr_arc, lw=1.3, label="Arc of deforestation")
ax.plot(dates, int_ts, color=clr_int, lw=1.3, label="Intact interior")
ax.set_ylabel(r"Root-zone SM (m$^3$ m$^{-3}$)")
ax.set_xlabel("Year")
ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_title("(a) Monthly root-zone soil moisture", fontsize=10, loc="left")
ax.grid(True, alpha=0.3, lw=0.5)

# (b) Box plot of residence times
ax = axes[1]
bp = ax.boxplot([arc_tau, int_tau], labels=["Arc", "Intact"],
                patch_artist=True, widths=0.5,
                medianprops=dict(color="black", lw=1.5))
bp["boxes"][0].set_facecolor(clr_arc)
bp["boxes"][0].set_alpha(0.7)
bp["boxes"][1].set_facecolor(clr_int)
bp["boxes"][1].set_alpha(0.7)
ax.set_ylabel("Residence time (months)")
sig = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"
ax.set_title(f"(b) Dry-down e-folding time\n({sig})", fontsize=10, loc="left")
ax.grid(True, axis="y", alpha=0.3, lw=0.5)

# (c) Scatter: SM anomaly vs 1-month decay
ax = axes[2]
ax.scatter(arc_pk_anom, arc_1m_decay, c=clr_arc, s=30, alpha=0.7,
           edgecolors="white", lw=0.3, label="Arc", zorder=3)
ax.scatter(int_pk_anom, int_1m_decay, c=clr_int, s=30, alpha=0.7,
           edgecolors="white", lw=0.3, label="Intact", zorder=3)
for xd, yd, clr in [(arc_pk_anom, arc_1m_decay, clr_arc),
                     (int_pk_anom, int_1m_decay, clr_int)]:
    if len(xd) > 2:
        sl, ic, _, _, _ = stats.linregress(xd, yd)
        xfit = np.linspace(xd.min(), xd.max(), 50)
        ax.plot(xfit, sl * xfit + ic, color=clr, lw=1.5, ls="--", alpha=0.8)
ax.set_xlabel(r"SM anomaly at peak (m$^3$ m$^{-3}$)")
ax.set_ylabel(r"1-month SM loss (m$^3$ m$^{-3}$)")
ax.set_title("(c) Anomaly vs. decay rate", fontsize=10, loc="left")
ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
ax.grid(True, alpha=0.3, lw=0.5)

plt.tight_layout()
outpath = FIG_DIR / "fig07_smap_residence.png"
fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {outpath}")
plt.close()
print("Script 08 complete.")
