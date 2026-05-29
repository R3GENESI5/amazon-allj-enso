"""Fig 5 — coast-to-interior moisture flux vs CHIRPS precipitation, by ENSO phase."""

from __future__ import annotations

import os, sys, calendar
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from fig_style_v2 import set_style, panel_label, classify_years, COL2

ERA5_MONTHLY = r"D:/amazon paper/data/era5/era5_monthly_means_900hpa_feb_1979_2023.nc"
CHIRPS_DIR = r"D:/amazon paper/data/chirps"
ONI_CSV = r"D:/amazon paper/data/oni_classification.csv"
OUT_PNG = r"D:/Projects/research-programme/manuscripts/amazon/p4_allj_enso/figures_v2/fig05_moisture_budget.png"
OUT_PDF = OUT_PNG.replace(".png", ".pdf")

TRANSECT_LONS = [-48, -52, -56, -60, -64]
LAT_BAND = (-5, 0)
CHIRPS_BAND_WIDTH = 4

C_EN = "#d62828"; C_LN = "#1d4e89"; C_NT = "#777777"
MARK = {"El Nino": "o", "La Nina": "s", "Neutral": "D"}
LABEL = {"El Nino": "El Niño", "La Nina": "La Niña", "Neutral": "Neutral"}
COLOR = {"El Nino": C_EN, "La Nina": C_LN, "Neutral": C_NT}
OFFSET = {"El Nino": -0.4, "La Nina": 0.0, "Neutral": 0.4}


def load_flux():
    ds = xr.open_dataset(ERA5_MONTHLY)
    u = ds["u"].squeeze("pressure_level")
    q = ds["q"].squeeze("pressure_level")
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    times = pd.DatetimeIndex(ds["valid_time"].values)
    yrs = times.year
    uniq_yrs = np.array(sorted(set(yrs)))
    qu = (q * u).values
    n = len(uniq_yrs)
    qu_y = np.empty((n, len(lat), len(lon)))
    for i, y in enumerate(uniq_yrs):
        qu_y[i] = np.nanmean(qu[yrs == y], axis=0)
    lat_mask = (lat <= LAT_BAND[1]) & (lat >= LAT_BAND[0])
    out = {}
    for tl in TRANSECT_LONS:
        il = int(np.argmin(np.abs(lon - tl)))
        out[tl] = -np.nanmean(qu_y[:, lat_mask, il], axis=1)
    return uniq_yrs, out


def load_precip():
    out = {tl: [] for tl in TRANSECT_LONS}
    years = []
    for yr in range(1981, 2025):
        f = os.path.join(CHIRPS_DIR, f"chirps-v2.0.{yr}.monthly.nc")
        if not os.path.exists(f):
            continue
        ds = xr.open_dataset(f)
        feb = ds["precip"].isel(time=1)
        clat = ds["latitude"].values
        clon = ds["longitude"].values
        clat_mask = (clat <= LAT_BAND[1]) & (clat >= LAT_BAND[0])
        feb_days = 29 if calendar.isleap(yr) else 28
        half = CHIRPS_BAND_WIDTH / 2
        for tl in TRANSECT_LONS:
            clon_mask = (clon >= tl - half) & (clon <= tl + half)
            band = feb.values[np.ix_(clat_mask, clon_mask)]
            out[tl].append(np.nanmean(band) / feb_days)
        years.append(yr)
        ds.close()
    return np.array(years), {k: np.array(v) for k, v in out.items()}


def composite_phase(data, yrs, oni):
    return {p: data[np.isin(yrs, oni[oni["phase"] == p]["year"].values)]
            for p in ["El Nino", "La Nina", "Neutral"]}


def main():
    set_style()
    print("loading flux...")
    yrs_e, flux_by_lon = load_flux()
    print("loading CHIRPS...")
    yrs_c, precip_by_lon = load_precip()
    oni = pd.read_csv(ONI_CSV)

    flux_c = {tl: composite_phase(flux_by_lon[tl], yrs_e, oni) for tl in TRANSECT_LONS}
    prec_c = {tl: composite_phase(precip_by_lon[tl], yrs_c, oni) for tl in TRANSECT_LONS}

    retention = flux_by_lon[-64] / flux_by_lon[-48]
    ret_c = composite_phase(retention, yrs_e, oni)
    t, p = stats.ttest_ind(ret_c["El Nino"], ret_c["La Nina"], equal_var=False)

    x = np.array([abs(t) for t in TRANSECT_LONS])

    fig = plt.figure(figsize=(COL2, COL2 * 0.42))
    gs = gridspec.GridSpec(
        1, 3, figure=fig, width_ratios=[1.5, 1.5, 1],
        left=0.06, right=0.98, top=0.88, bottom=0.16, wspace=0.32,
    )
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    for ax, src, ylabel in [
        (ax1, flux_c, r"Westward q·u  (10$^{-3}$ kg kg$^{-1}$ m s$^{-1}$)"),
        (ax2, prec_c, r"Precipitation  (mm day$^{-1}$)"),
    ]:
        for phase in ["El Nino", "La Nina", "Neutral"]:
            means, ci = [], []
            for tl in TRANSECT_LONS:
                d = src[tl][phase] * (1e3 if src is flux_c else 1)
                means.append(np.nanmean(d))
                ci.append(1.96 * np.nanstd(d) / np.sqrt(len(d)))
            ax.errorbar(
                x + OFFSET[phase], means, yerr=ci,
                color=COLOR[phase], marker=MARK[phase], markersize=4,
                linewidth=1.2, capsize=2, capthick=0.8,
                label=LABEL[phase],
                zorder=3 if phase == "El Nino" else 2,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{xi}°W" for xi in x])
        ax.invert_xaxis()
        ax.set_xlabel("Longitude")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.3)
        ax.tick_params(labelsize=8)
    ax1.legend(loc="upper left", fontsize=7)
    panel_label(ax1, "a")
    panel_label(ax2, "b")

    # Retention ratio bar with 95% CI as panel (c) — clean, dedicated panel
    phases = ["El Nino", "Neutral", "La Nina"]
    means_r = [np.nanmean(ret_c[p]) for p in phases]
    ci_r = [1.96 * np.nanstd(ret_c[p]) / np.sqrt(len(ret_c[p])) for p in phases]
    bar_colors = [COLOR[p] for p in phases]
    bars = ax3.bar(
        range(3), means_r, yerr=ci_r,
        color=bar_colors, alpha=0.7, edgecolor="black", linewidth=0.5,
        capsize=3, error_kw=dict(elinewidth=0.7),
    )
    ax3.set_xticks(range(3))
    ax3.set_xticklabels([LABEL[p] for p in phases])
    ax3.set_ylabel("Flux retention  (64°W / 48°W)")
    ax3.axhline(1.0, color="0.5", linewidth=0.5, linestyle=":")
    ax3.set_ylim(0.0, max(means_r) * 1.35)
    sig = f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"
    ax3.text(0.5, 1.02, f"El Niño vs La Niña: {sig}",
             transform=ax3.transAxes, ha="center", va="bottom",
             fontsize=7, color="0.25", style="italic")
    ax3.grid(True, axis="y", alpha=0.25, linewidth=0.3)
    ax3.tick_params(labelsize=8)
    panel_label(ax3, "c")

    fig.text(0.5, 0.985,
             "Coast-to-interior moisture budget by ENSO phase (February)",
             ha="center", va="top", fontsize=10)

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
