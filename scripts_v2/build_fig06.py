"""Fig 6 — Deforestation signal: arc vs intact interior.

(a) Feb CHIRPS precip time series, arc vs interior (10-yr running mean).
(b) ENSO–precip scatter for the arc (colored by year), with linear fit.
(c) Residual precip after ENSO removal, with secular trend line.
(d) GRACE TWS, arc vs interior, monthly + 12-month rolling mean, with trends.
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import netCDF4
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import stats
import statsmodels.api as sm

from fig_style_v2 import set_style, panel_label, COL2

CHIRPS_DIR = r"D:/amazon paper/data/chirps"
GRACE_NC = r"D:/amazon paper/data/grace/GRCTellus.JPL.200204_202512.GLO.RL06.3M.MSCNv04CRI.nc"
ONI_CSV = r"D:/amazon paper/data/oni_classification.csv"
OUT_PNG = r"D:/Projects/research-programme/manuscripts/amazon/p4_allj_enso/figures_v2/fig06_deforestation_signal.png"
OUT_PDF = OUT_PNG.replace(".png", ".pdf")

ARC_LAT = (-12, -5);   ARC_LON = (-55, -45)
INT_LAT = (-5,  0);    INT_LON = (-65, -55)

C_ARC = "#d95f02"
C_INT = "#1b7837"
C_NINO = "#f4cccc"
C_NINA = "#cfe2f3"


def region_mean(nc, var, t, lat_b, lon_b):
    lats = nc.variables["latitude"][:]; lons = nc.variables["longitude"][:]
    lm = np.where((lats >= lat_b[0]) & (lats <= lat_b[1]))[0]
    lo = np.where((lons >= lon_b[0]) & (lons <= lon_b[1]))[0]
    data = np.ma.filled(nc.variables[var][t, lm[0]:lm[-1]+1, lo[0]:lo[-1]+1], np.nan)
    return float(np.nanmean(data))


def running_mean(a, window=10):
    rm = np.full_like(a, np.nan, dtype=float)
    h = window // 2
    for i in range(h, len(a) - h):
        rm[i] = np.nanmean(a[i - h:i + h])
    return rm


def main():
    set_style()
    oni = pd.read_csv(ONI_CSV)
    oni_d = dict(zip(oni["year"], oni["oni"]))
    phase_d = dict(zip(oni["year"], oni["phase"]))

    print("loading CHIRPS arc + interior...")
    yrs, arc, intr = [], [], []
    for yr in range(1981, 2025):
        fp = f"{CHIRPS_DIR}/chirps-v2.0.{yr}.monthly.nc"
        if not os.path.exists(fp): continue
        nc = netCDF4.Dataset(fp)
        arc.append(region_mean(nc, "precip", 1, ARC_LAT, ARC_LON))
        intr.append(region_mean(nc, "precip", 1, INT_LAT, INT_LON))
        yrs.append(yr); nc.close()
    yrs = np.array(yrs); arc = np.array(arc); intr = np.array(intr)
    rm_arc = running_mean(arc, 10); rm_int = running_mean(intr, 10)

    oni_vals = np.array([oni_d.get(y, np.nan) for y in yrs])
    mask = ~np.isnan(oni_vals)
    X = sm.add_constant(np.column_stack([yrs[mask] - yrs[mask].mean(), oni_vals[mask]]))
    model = sm.OLS(arc[mask], X).fit()
    enso_contrib = model.params[2] * oni_vals[mask]
    resid = arc[mask] - enso_contrib
    slope_r, int_r, r_r, p_r, _ = stats.linregress(yrs[mask], resid)
    slope_oni, int_oni, r_oni, p_oni, _ = stats.linregress(oni_vals[mask], arc[mask])

    print("loading GRACE arc + interior...")
    g = netCDF4.Dataset(GRACE_NC)
    g_lat = g.variables["lat"][:]; g_lon = g.variables["lon"][:]
    g_t = netCDF4.num2date(g.variables["time"][:], g.variables["time"].units,
                           only_use_cftime_datetimes=False)
    g_time = pd.to_datetime([str(t) for t in g_t])

    def grace_ts(lat_b, lon_b_360):
        lm = np.where((g_lat >= lat_b[0]) & (g_lat <= lat_b[1]))[0]
        lo = np.where((g_lon >= lon_b_360[0]) & (g_lon <= lon_b_360[1]))[0]
        lwe = g.variables["lwe_thickness"]
        sf = g.variables["scale_factor"]
        ts = []
        for t in range(lwe.shape[0]):
            d = np.ma.filled(lwe[t, lm[0]:lm[-1]+1, lo[0]:lo[-1]+1], np.nan)
            s = np.ma.filled(sf[lm[0]:lm[-1]+1, lo[0]:lo[-1]+1], np.nan)
            ts.append(float(np.nanmean(d * s)))
        return np.array(ts)
    to360 = lambda x: x % 360
    tws_a = grace_ts(ARC_LAT, (to360(ARC_LON[0]), to360(ARC_LON[1])))
    tws_i = grace_ts(INT_LAT, (to360(INT_LON[0]), to360(INT_LON[1])))
    g.close()
    g_dec = np.array([t.year + t.month / 12 for t in g_time])
    va = ~np.isnan(tws_a); vi = ~np.isnan(tws_i)
    sa, ia, _, pa, ea = stats.linregress(g_dec[va], tws_a[va])
    si, ii, _, pi, ei = stats.linregress(g_dec[vi], tws_i[vi])
    z = (sa - si) / np.sqrt(ea**2 + ei**2)
    pdiff = 2 * (1 - stats.norm.cdf(abs(z)))

    fig = plt.figure(figsize=(COL2, COL2 * 0.78))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           left=0.07, right=0.97, top=0.94, bottom=0.07,
                           wspace=0.28, hspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # (a) precip time series — drop scatter dots to reduce clutter; keep running mean + light ENSO shading
    for y in yrs:
        phase = phase_d.get(int(y), "Neutral")
        if phase == "El Nino":
            ax1.axvspan(y - 0.4, y + 0.4, color=C_NINO, alpha=0.5, linewidth=0, zorder=0)
        elif phase == "La Nina":
            ax1.axvspan(y - 0.4, y + 0.4, color=C_NINA, alpha=0.5, linewidth=0, zorder=0)
    ax1.plot(yrs, arc, color=C_ARC, linewidth=0.4, alpha=0.6)
    ax1.plot(yrs, intr, color=C_INT, linewidth=0.4, alpha=0.6)
    ax1.plot(yrs, rm_arc, color=C_ARC, linewidth=1.8, label="Arc of deforestation")
    ax1.plot(yrs, rm_int, color=C_INT, linewidth=1.8, label="Intact interior")
    ax1.set_xlim(1981, 2024)
    ax1.set_xlabel("Year"); ax1.set_ylabel("February precipitation (mm)")
    ax1.tick_params(labelsize=8); ax1.grid(True, alpha=0.2, linewidth=0.3)
    handles_main = [
        plt.Line2D([], [], color=C_ARC, linewidth=1.6, label="Arc"),
        plt.Line2D([], [], color=C_INT, linewidth=1.6, label="Intact interior"),
    ]
    handles_enso = [
        Patch(facecolor=C_NINO, alpha=0.7, label="El Niño"),
        Patch(facecolor=C_NINA, alpha=0.7, label="La Niña"),
    ]
    ax1.legend(handles=handles_main + handles_enso, loc="lower right", fontsize=7, ncol=2)
    panel_label(ax1, "a")

    # (b) ONI vs arc precip scatter
    sc = ax2.scatter(oni_vals[mask], arc[mask], c=yrs[mask], cmap="YlOrBr",
                     s=24, edgecolor="black", linewidth=0.3, zorder=3)
    cb = plt.colorbar(sc, ax=ax2, shrink=0.85, pad=0.02)
    cb.set_label("Year", fontsize=8); cb.ax.tick_params(labelsize=7)
    xr_ = np.linspace(oni_vals[mask].min() - 0.1, oni_vals[mask].max() + 0.1, 50)
    ax2.plot(xr_, int_oni + slope_oni * xr_, "--", color="0.3", linewidth=1.0)
    ax2.axvline(0, color="0.6", linewidth=0.4, linestyle=":")
    ax2.set_xlabel("ONI (DJF)"); ax2.set_ylabel("Arc Feb precipitation (mm)")
    ax2.tick_params(labelsize=8); ax2.grid(True, alpha=0.2, linewidth=0.3)
    ax2.text(0.97, 0.03,
             f"r = {r_oni:.2f}\np = {p_oni:.3f}",
             transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=7, color="0.25")
    panel_label(ax2, "b")

    # (c) residual trend
    ax3.plot(yrs[mask], resid, "o", color=C_ARC, markersize=2.6, alpha=0.55, markeredgecolor="none")
    rm_resid = running_mean(resid, 10)
    ax3.plot(yrs[mask], rm_resid, color=C_ARC, linewidth=1.8, label="10-yr running mean")
    trend = int_r + slope_r * yrs[mask]
    ax3.plot(yrs[mask], trend, "--", color="0.25", linewidth=1.0,
             label=f"Trend  {slope_r:+.2f} mm/yr")
    sig = "n.s." if p_r >= 0.05 else "*" if p_r >= 0.01 else "**" if p_r >= 0.001 else "***"
    ax3.set_xlim(1981, 2024)
    ax3.set_xlabel("Year"); ax3.set_ylabel("ENSO-adjusted Feb precip (mm)")
    ax3.tick_params(labelsize=8); ax3.grid(True, alpha=0.2, linewidth=0.3)
    ax3.legend(loc="upper left", fontsize=7)
    ax3.text(0.97, 0.03, f"p = {p_r:.3f} ({sig})",
             transform=ax3.transAxes, ha="right", va="bottom",
             fontsize=7, color="0.25")
    panel_label(ax3, "c")

    # (d) GRACE arc vs interior
    sa_arc = pd.Series(tws_a, index=g_time).rolling(12, center=True, min_periods=6).mean()
    sa_int = pd.Series(tws_i, index=g_time).rolling(12, center=True, min_periods=6).mean()
    ax4.plot(g_time, tws_a, color=C_ARC, linewidth=0.4, alpha=0.35)
    ax4.plot(g_time, tws_i, color=C_INT, linewidth=0.4, alpha=0.35)
    ax4.plot(sa_arc.index, sa_arc.values, color=C_ARC, linewidth=1.8,
             label=f"Arc   {sa:+.2f} cm/yr (p={pa:.2f})")
    ax4.plot(sa_int.index, sa_int.values, color=C_INT, linewidth=1.8,
             label=f"Int.  {si:+.2f} cm/yr (p={pi:.2f})")
    ax4.plot(g_time, ia + sa * g_dec, "--", color=C_ARC, linewidth=0.8, alpha=0.65)
    ax4.plot(g_time, ii + si * g_dec, "--", color=C_INT, linewidth=0.8, alpha=0.65)
    ax4.axhline(0, color="0.5", linewidth=0.4, linestyle=":")
    ax4.set_xlabel("Year"); ax4.set_ylabel("TWS anomaly (cm eq. water)")
    ax4.tick_params(labelsize=8); ax4.grid(True, alpha=0.2, linewidth=0.3)
    ax4.legend(loc="lower left", fontsize=7)
    ax4.text(0.97, 0.97,
             f"Δslope: p = {pdiff:.3f}",
             transform=ax4.transAxes, ha="right", va="top",
             fontsize=7, color="0.25", style="italic")
    panel_label(ax4, "d")

    fig.text(0.5, 0.985,
             "Deforestation signal in Feb precipitation and TWS: arc vs intact interior",
             ha="center", va="top", fontsize=10)

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
