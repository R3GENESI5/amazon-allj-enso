"""Fig 3 — GRACE/GRACE-FO terrestrial water storage anomaly over the Amazon basin.

Monthly TWS, 12-month running mean, ENSO-period shading.
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from fig_style_v2 import set_style, panel_label, COL2

GRACE_NC = r"D:/amazon paper/data/grace/GRCTellus.JPL.200204_202512.GLO.RL06.3M.MSCNv04CRI.nc"
ONI_CSV = r"D:/amazon paper/data/oni_classification.csv"
OUT_PNG = r"D:/Projects/research-programme/manuscripts/amazon/p4_allj_enso/figures_v2/fig03_grace_tws.png"
OUT_PDF = OUT_PNG.replace(".png", ".pdf")

LAT_RANGE = (-15, 5)
LON_RANGE_180 = (-75, -45)
LON_RANGE_360 = (285, 315)


def load_tws() -> tuple[pd.DatetimeIndex, np.ndarray]:
    ds = xr.open_dataset(GRACE_NC)
    if float(ds.lon.max()) > 180:
        lon_min, lon_max = LON_RANGE_360
    else:
        lon_min, lon_max = LON_RANGE_180
    sel = (ds.lat >= LAT_RANGE[0]) & (ds.lat <= LAT_RANGE[1]) \
        & (ds.lon >= lon_min) & (ds.lon <= lon_max)
    land = ds["land_mask"].where(sel, drop=True)
    scl = ds["scale_factor"].where(sel, drop=True)
    tws = ds["lwe_thickness"].where(sel, drop=True) * scl
    tws = tws.where(land == 1)
    w = np.cos(np.deg2rad(tws.lat))
    mean = tws.weighted(w).mean(dim=["lat", "lon"])
    times = pd.DatetimeIndex(ds["time"].values)
    return times, mean.values


def main() -> None:
    set_style()
    print("loading GRACE...")
    times, tws = load_tws()
    print(f"  n = {len(tws)} months ({times[0]:%Y-%m} → {times[-1]:%Y-%m})")
    s = pd.Series(tws, index=times)
    rm = s.rolling(12, center=True, min_periods=6).mean()

    oni = pd.read_csv(ONI_CSV)

    fig, ax = plt.subplots(figsize=(COL2, COL2 * 0.32))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.20)

    t0, t1 = times[0], times[-1]
    for _, row in oni.iterrows():
        yr = int(row["year"]); phase = row["phase"]
        if yr < t0.year - 1 or yr > t1.year + 1:
            continue
        start = datetime(yr - 1, 7, 1); end = datetime(yr, 6, 30)
        if phase == "El Nino":
            ax.axvspan(start, end, color="#f4cccc", alpha=0.45, linewidth=0, zorder=0)
        elif phase == "La Nina":
            ax.axvspan(start, end, color="#cfe2f3", alpha=0.45, linewidth=0, zorder=0)
    ax.set_xlim(t0, t1)

    ax.axhline(0, color="0.5", linewidth=0.5)
    ax.plot(times, tws, color="#2a2a2a", linewidth=0.6, alpha=0.7, label="monthly TWS")
    ax.plot(rm.index, rm.values, color="#c62828", linewidth=1.6, label="12-month running mean")

    notable = {"2010 EN": (2010, 7), "2016 EN": (2016, 5), "2024 EN": (2024, 7)}
    ymax = ax.get_ylim()[1]
    for label, (yr, mo) in notable.items():
        ax.annotate(
            label,
            xy=(datetime(yr, mo, 1), ymax * 0.92),
            ha="center", va="top", fontsize=7, color="#a00000", fontweight="bold",
        )

    ax.set_ylabel("TWS anomaly (cm eq. water)")
    ax.set_xlabel("")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2, linewidth=0.3)

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#f4cccc", alpha=0.7, label="El Niño period"),
        Patch(facecolor="#cfe2f3", alpha=0.7, label="La Niña period"),
        plt.Line2D([], [], color="#2a2a2a", linewidth=0.8, label="Monthly TWS"),
        plt.Line2D([], [], color="#c62828", linewidth=1.6, label="12-month running mean"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=7, ncol=4,
              bbox_to_anchor=(0.0, -0.32), frameon=False)

    fig.text(0.5, 0.96,
             "GRACE/GRACE-FO terrestrial water storage anomaly over the Amazon basin (5°N–15°S, 75–45°W)",
             ha="center", va="top", fontsize=10)
    panel_label(ax, "")  # single panel; skip label

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
