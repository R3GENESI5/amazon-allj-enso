"""Fig 7 — SMAP L4 root-zone soil moisture residence time: arc vs intact interior."""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
import numpy as np
import netCDF4
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from scipy import stats

from fig_style_v2 import set_style, panel_label, COL2

SMAP_NC = r"D:/amazon paper/data/smap/smap_l4_rootzone_sm_monthly_amazon_2015_2024.nc"
OUT_PNG = r"D:/Projects/research-programme/manuscripts/amazon/p4_allj_enso/figures_v2/fig07_smap_residence.png"
OUT_PDF = OUT_PNG.replace(".png", ".pdf")

ARC_LAT = (-12, -5); ARC_LON = (-55, -45)
INT_LAT = (-5, 0);   INT_LON = (-65, -55)

C_ARC = "#d95f02"
C_INT = "#1b9e77"


def regional_mean(sm, lat, lon, lat_b, lon_b):
    lat_m = (lat >= lat_b[0]) & (lat <= lat_b[1])
    lon_m = (lon >= lon_b[0]) & (lon <= lon_b[1])
    region = sm[:, lat_m, :][:, :, lon_m]
    w = np.cos(np.deg2rad(lat[lat_m]))[None, :, None]
    wt = np.where(np.isnan(region), 0, np.broadcast_to(w, region.shape))
    return np.nansum(region * wt, axis=(1, 2)) / np.nansum(wt, axis=(1, 2))


def compute_residence(ts):
    dsm = np.diff(ts)
    tau, peaks = [], []
    i = 0
    while i < len(dsm):
        if dsm[i] > 0:
            pk = i + 1
            pk_val = ts[pk]
            j = pk
            while j < len(ts) - 1 and ts[j + 1] < ts[j]:
                j += 1
            dd = j - pk
            if dd >= 2:
                tt = np.arange(dd + 1, dtype=float)
                norm = ts[pk:pk + dd + 1] / pk_val
                with np.errstate(divide="ignore", invalid="ignore"):
                    ln = np.log(norm)
                ok = np.isfinite(ln)
                if ok.sum() >= 2:
                    slope, *_ = stats.linregress(tt[ok], ln[ok])
                    if slope < 0:
                        t = -1.0 / slope
                        if 0.5 < t < 24:
                            tau.append(t); peaks.append(pk_val)
            i = j if j > pk else i + 1
        else:
            i += 1
    return np.array(tau), np.array(peaks)


def decay_segments(ts):
    dsm = np.diff(ts)
    anom = ts - np.nanmean(ts)
    pk_anom, d1 = [], []
    for i in range(len(dsm)):
        if dsm[i] > 0 and i + 1 < len(dsm) and dsm[i + 1] < 0:
            pk_anom.append(anom[i + 1]); d1.append(-dsm[i + 1])
    return np.array(pk_anom), np.array(d1)


def main():
    set_style()
    ds = netCDF4.Dataset(SMAP_NC)
    t = ds.variables["time"][:]; lat = ds.variables["lat"][:]; lon = ds.variables["lon"][:]
    sm = ds.variables["sm_rootzone"][:]; ds.close()
    sm = np.where((sm < 0) | (sm > 1), np.nan, sm)
    base = datetime(2000, 1, 1)
    dates = np.array([base + timedelta(days=float(d)) for d in t])

    arc_ts = regional_mean(sm, lat, lon, ARC_LAT, ARC_LON)
    int_ts = regional_mean(sm, lat, lon, INT_LAT, INT_LON)

    arc_tau, _ = compute_residence(arc_ts)
    int_tau, _ = compute_residence(int_ts)
    tstat, p = stats.ttest_ind(arc_tau, int_tau, equal_var=False)

    arc_pa, arc_d1 = decay_segments(arc_ts)
    int_pa, int_d1 = decay_segments(int_ts)

    fig = plt.figure(figsize=(COL2, COL2 * 0.75))
    gs = gridspec.GridSpec(
        2, 2, figure=fig, height_ratios=[1, 1.1],
        left=0.08, right=0.97, top=0.92, bottom=0.08,
        wspace=0.30, hspace=0.40,
    )
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    ax1.plot(dates, arc_ts, color=C_ARC, linewidth=1.2, label="Arc of deforestation")
    ax1.plot(dates, int_ts, color=C_INT, linewidth=1.2, label="Intact interior")
    ax1.set_ylabel(r"Root-zone SM (m$^3$ m$^{-3}$)")
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    # Push y-range up so legend at top doesn't clip data
    y0, y1 = ax1.get_ylim()
    ax1.set_ylim(y0, y1 + 0.08 * (y1 - y0))
    ax1.legend(loc="upper right", fontsize=8, ncol=2, frameon=True,
               framealpha=0.92, edgecolor="0.7", facecolor="white")
    ax1.grid(True, alpha=0.25, linewidth=0.3)
    ax1.tick_params(labelsize=8)
    panel_label(ax1, "a")

    bp = ax2.boxplot(
        [arc_tau, int_tau], positions=[1, 2], widths=0.55, patch_artist=True,
        medianprops=dict(color="black", linewidth=1.2),
        boxprops=dict(linewidth=0.6),
        whiskerprops=dict(linewidth=0.6),
        capprops=dict(linewidth=0.6),
        flierprops=dict(marker="o", markersize=3, alpha=0.5,
                        markerfacecolor="0.5", markeredgecolor="none"),
    )
    bp["boxes"][0].set_facecolor(C_ARC); bp["boxes"][0].set_alpha(0.65)
    bp["boxes"][1].set_facecolor(C_INT); bp["boxes"][1].set_alpha(0.65)
    ax2.set_xticks([1, 2]); ax2.set_xticklabels(["Arc", "Intact"])
    ax2.set_ylabel("e-folding τ (months)")
    sig = f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"
    ax2.text(0.5, 1.02, f"Arc vs Intact: {sig}",
             transform=ax2.transAxes, ha="center", va="bottom",
             fontsize=7, color="0.25", style="italic")
    ax2.grid(True, axis="y", alpha=0.25, linewidth=0.3)
    ax2.tick_params(labelsize=8)
    panel_label(ax2, "b")

    for xd, yd, c, name in [
        (arc_pa, arc_d1, C_ARC, "Arc"),
        (int_pa, int_d1, C_INT, "Intact"),
    ]:
        ax3.scatter(xd, yd, color=c, s=18, alpha=0.7, edgecolor="white", linewidth=0.3, label=name)
        if len(xd) > 2:
            sl, ic, *_ = stats.linregress(xd, yd)
            xf = np.linspace(xd.min(), xd.max(), 30)
            ax3.plot(xf, sl * xf + ic, color=c, linewidth=1.0, linestyle="--", alpha=0.8)
    ax3.set_xlabel(r"SM anomaly at peak (m$^3$ m$^{-3}$)")
    ax3.set_ylabel(r"1-month SM loss (m$^3$ m$^{-3}$)")
    ax3.legend(loc="upper left", fontsize=7)
    ax3.grid(True, alpha=0.25, linewidth=0.3)
    ax3.tick_params(labelsize=8)
    panel_label(ax3, "c")

    fig.text(0.5, 0.97,
             "SMAP L4 root-zone soil moisture residence time: arc vs intact interior, 2015–2024",
             ha="center", va="top", fontsize=10)

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"saved {OUT_PNG}  (n_arc={len(arc_tau)}, n_int={len(int_tau)}, p={p:.4f})")


if __name__ == "__main__":
    main()
