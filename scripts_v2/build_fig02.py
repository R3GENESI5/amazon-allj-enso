"""Fig 2 — February CHIRPS precipitation: coast-to-interior transect by ENSO phase.

Panels
------
(a) Transect along 0-3°S band from 70°W to 44°W; mean ± 1.96·SE for each ENSO phase.
(b) Interior (70-58°W) minus coast (52-44°W) precipitation gradient as boxplot.
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from fig_style_v2 import set_style, panel_label, classify_years, COL2

CHIRPS_DIR = r"D:/amazon paper/data/chirps"
OUT_PNG = r"D:/Projects/research-programme/manuscripts/amazon/p4_allj_enso/figures_v2/fig02_chirps_transect.png"
OUT_PDF = OUT_PNG.replace(".png", ".pdf")

LAT_MIN, LAT_MAX = -3.0, 0.0
LON_MIN, LON_MAX = -70.0, -44.0
COAST = (-52, -44)
INTERIOR = (-70, -58)

C_EN = "#d62828"
C_LN = "#1d4e89"
C_NT = "#777777"


def load_transects() -> tuple[np.ndarray, dict[int, np.ndarray]]:
    out: dict[int, np.ndarray] = {}
    lons = None
    for yr in range(1981, 2025):
        f = os.path.join(CHIRPS_DIR, f"chirps-v2.0.{yr}.monthly.nc")
        if not os.path.exists(f):
            continue
        ds = xr.open_dataset(f)
        feb = ds["precip"].isel(time=1)
        lat_sel = (ds.latitude >= LAT_MIN) & (ds.latitude <= LAT_MAX)
        lon_sel = (ds.longitude >= LON_MIN) & (ds.longitude <= LON_MAX)
        region = feb.where(lat_sel & lon_sel, drop=True)
        transect = region.mean(dim="latitude").values
        if lons is None:
            lons = region.longitude.values
        out[yr] = transect
        ds.close()
    return lons, out


def main() -> None:
    set_style()
    print("loading CHIRPS...")
    lons, transects = load_transects()
    years = classify_years()

    def stack(yr_list: list[int]) -> np.ndarray:
        return np.array([transects[y] for y in yr_list if y in transects])

    en = stack(years["elnino"])
    ln = stack(years["lanina"])
    nt = stack(years["neutral"])
    print(f"  n_en={len(en)}, n_ln={len(ln)}, n_nt={len(nt)}")

    def m_se(a):
        return np.nanmean(a, 0), np.nanstd(a, 0) / np.sqrt(len(a))

    en_m, en_se = m_se(en)
    ln_m, ln_se = m_se(ln)
    nt_m, nt_se = m_se(nt)

    coast_mask = (lons >= COAST[0]) & (lons <= COAST[1])
    int_mask = (lons >= INTERIOR[0]) & (lons <= INTERIOR[1])

    def gradient(arr):
        return np.nanmean(arr[:, int_mask], 1) - np.nanmean(arr[:, coast_mask], 1)

    g_en, g_ln, g_nt = gradient(en), gradient(ln), gradient(nt)
    t, p = stats.ttest_ind(g_en, g_ln, equal_var=False)

    fig = plt.figure(figsize=(COL2, COL2 * 0.44))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2.6, 1],
                           left=0.07, right=0.97, top=0.83, bottom=0.16, wspace=0.20)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    ax1.axvspan(*COAST, color="#bcd7ec", alpha=0.35, zorder=0, linewidth=0)
    ax1.axvspan(*INTERIOR, color="#cfe6cf", alpha=0.35, zorder=0, linewidth=0)
    ax1.text(np.mean(COAST), ax1.get_ylim()[1], "coast",
             ha="center", va="bottom", fontsize=7, color="#3a6a8f", transform=ax1.get_xaxis_transform())
    ax1.text(np.mean(INTERIOR), ax1.get_ylim()[1], "interior",
             ha="center", va="bottom", fontsize=7, color="#3a6a3a", transform=ax1.get_xaxis_transform())

    for m, se, c, name, n in [
        (en_m, en_se, C_EN, "El Niño", len(en)),
        (ln_m, ln_se, C_LN, "La Niña", len(ln)),
        (nt_m, nt_se, C_NT, "Neutral", len(nt)),
    ]:
        ax1.fill_between(lons, m - 1.96 * se, m + 1.96 * se, alpha=0.18, color=c, linewidth=0)
        ax1.plot(lons, m, color=c, linewidth=1.6, label=f"{name}  (n = {n})")

    ax1.set_xlim(LON_MAX, LON_MIN)  # invert (coast on left)
    ax1.set_xlabel("Longitude (°W)")
    ax1.set_ylabel("February precipitation (mm month$^{-1}$)")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.25, linewidth=0.3)
    ax1.tick_params(labelsize=8)
    # Convert longitude tick labels to positive W
    ax1.set_xticks([-45, -50, -55, -60, -65, -70])
    ax1.set_xticklabels(["45", "50", "55", "60", "65", "70"])
    panel_label(ax1, "a")

    data = [g_en, g_nt, g_ln]
    colors = [C_EN, C_NT, C_LN]
    bp = ax2.boxplot(
        data, positions=[1, 2, 3], widths=0.55, patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white",
                       markeredgecolor="black", markersize=4),
        medianprops=dict(color="black", linewidth=1.2),
        flierprops=dict(marker="o", markersize=3, alpha=0.5,
                        markerfacecolor="0.5", markeredgecolor="none"),
        boxprops=dict(linewidth=0.6),
        whiskerprops=dict(linewidth=0.6),
        capprops=dict(linewidth=0.6),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(["El Niño", "Neutral", "La Niña"])
    ax2.set_ylabel("Interior − Coast (mm month$^{-1}$)")
    ax2.axhline(0, color="0.3", linewidth=0.5)
    ax2.grid(True, axis="y", alpha=0.25, linewidth=0.3)
    ax2.tick_params(labelsize=8)

    sig = f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"
    ax2.text(0.5, 1.02, f"El Niño vs La Niña: {sig}",
             transform=ax2.transAxes, ha="center", va="bottom",
             fontsize=7, color="0.25", style="italic")
    panel_label(ax2, "b")

    fig.text(0.5, 0.96,
             "February CHIRPS precipitation along the 0–3°S transect by ENSO phase, 1981–2024",
             ha="center", va="top", fontsize=10)

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
