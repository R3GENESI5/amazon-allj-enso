"""Fig 8 — Diurnal cycles by ENSO phase, 10 selected ENSO Februaries.

Four panels:
  (a) 900 hPa moisture flux at 52°W, 0–5°S
  (b) CAPE area mean over 2–8°S, 60–50°W
  (c) BLH area mean over same box
  (d) 500 hPa ω over same box
"""

from __future__ import annotations

import os, sys, gc, zipfile, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from fig_style_v2 import set_style, panel_label, COL2

DATA_DIR = r"D:/amazon paper/data/era5"
OUT_PNG = r"D:/Projects/research-programme/manuscripts/amazon/p4_allj_enso/figures_v2/fig08_diurnal_enso.png"
OUT_PDF = OUT_PNG.replace(".png", ".pdf")
CACHE = r"D:/Projects/research-programme/manuscripts/amazon/p4_allj_enso/figures_v2/_fig08_cache.npz"

EN = [1983, 1992, 1998, 2010, 2016]
LN = [1989, 1999, 2000, 2008, 2011]
C_EN = "#d62828"; C_LN = "#1d4e89"

BOX_LAT = (-8.0, -2.0); BOX_LON = (-60.0, -50.0)
TRANSECT_LON = -52.0
TRANS_LAT = (-5.0, 0.0)


def load_sfc(y):
    fp = os.path.join(DATA_DIR, f"era5_sfc_{y}_feb.nc")
    with open(fp, "rb") as f:
        magic = f.read(4)
    if magic == b"PK\x03\x04":
        td = tempfile.mkdtemp()
        with zipfile.ZipFile(fp) as zf:
            zf.extractall(td)
        a = xr.open_dataset(os.path.join(td, "data_stream-oper_stepType-instant.nc"))
        b = xr.open_dataset(os.path.join(td, "data_stream-oper_stepType-accum.nc"))
        return xr.merge([a, b])
    return xr.open_dataset(fp)


def load_pl(y):
    return xr.open_dataset(os.path.join(DATA_DIR, f"era5_pl_{y}_feb.nc"))


def to_diurnal(da):
    v = da.values
    n = len(v) // 24
    return np.nanmean(v[:n * 24].reshape(n, 24), axis=0)


def process_year(y):
    sfc = load_sfc(y); pl = load_pl(y)
    lat = "latitude"; lon = "longitude"
    box_lats = pl[lat].sel({lat: slice(BOX_LAT[1], BOX_LAT[0])})
    wt_box = np.cos(np.deg2rad(box_lats))

    pl900 = pl.sel(pressure_level=900.0, method="nearest")
    q = pl900["q"].sel({lon: TRANSECT_LON}, method="nearest").sel({lat: slice(TRANS_LAT[1], TRANS_LAT[0])})
    u = pl900["u"].sel({lon: TRANSECT_LON}, method="nearest").sel({lat: slice(TRANS_LAT[1], TRANS_LAT[0])})
    v = pl900["v"].sel({lon: TRANSECT_LON}, method="nearest").sel({lat: slice(TRANS_LAT[1], TRANS_LAT[0])})
    mag = np.sqrt((q * u) ** 2 + (q * v) ** 2)
    wt_tr = np.cos(np.deg2rad(mag[lat]))
    d_mflux = to_diurnal(mag.weighted(wt_tr).mean(dim=lat))

    cape = sfc["cape"].sel({lat: slice(BOX_LAT[1], BOX_LAT[0]),
                            lon: slice(BOX_LON[0], BOX_LON[1])})
    d_cape = to_diurnal(cape.weighted(wt_box).mean(dim=[lat, lon]))

    blh = sfc["blh"].sel({lat: slice(BOX_LAT[1], BOX_LAT[0]),
                          lon: slice(BOX_LON[0], BOX_LON[1])})
    d_blh = to_diurnal(blh.weighted(wt_box).mean(dim=[lat, lon]))

    pl500 = pl.sel(pressure_level=500.0, method="nearest")
    w = pl500["w"].sel({lat: slice(BOX_LAT[1], BOX_LAT[0]),
                        lon: slice(BOX_LON[0], BOX_LON[1])})
    d_omega = to_diurnal(w.weighted(wt_box).mean(dim=[lat, lon]))

    sfc.close(); pl.close(); del sfc, pl, pl900, pl500; gc.collect()
    return d_mflux, d_cape, d_blh, d_omega


def compute_all():
    if os.path.exists(CACHE):
        print(f"loading cache {CACHE}")
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}
    out = {k: {"elnino": [], "lanina": []} for k in ["mflux", "cape", "blh", "omega"]}
    for phase, ys in [("elnino", EN), ("lanina", LN)]:
        for y in ys:
            print(f"  {phase} {y}...")
            mf, cp, bh, om = process_year(y)
            out["mflux"][phase].append(mf); out["cape"][phase].append(cp)
            out["blh"][phase].append(bh); out["omega"][phase].append(om)
    arrs = {}
    for k in out:
        for ph in out[k]:
            arrs[f"{k}_{ph}"] = np.array(out[k][ph])
    np.savez(CACHE, **arrs)
    return arrs


def main():
    set_style()
    print("processing ERA5 hourly ENSO years (cached after first run)...")
    arrs = compute_all()

    fig = plt.figure(figsize=(COL2, COL2 * 0.65))
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        left=0.08, right=0.97, top=0.88, bottom=0.10,
        wspace=0.28, hspace=0.45,
    )
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    hours = np.arange(24)

    def plot_panel(ax, key, ylabel, scale=1.0, invert=False):
        for phase, color, label in [("elnino", C_EN, "El Niño"), ("lanina", C_LN, "La Niña")]:
            arr = arrs[f"{key}_{phase}"] * scale
            mn = np.nanmean(arr, 0); sd = np.nanstd(arr, 0)
            ax.plot(hours, mn, color=color, linewidth=1.6, label=label, zorder=3)
            ax.fill_between(hours, mn - sd, mn + sd, color=color, alpha=0.15, linewidth=0, zorder=2)
        if invert:
            ax.invert_yaxis()
        ax.set_xlabel("Hour (UTC)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 23)
        ax.set_xticks(np.arange(0, 24, 3))
        ax.grid(True, alpha=0.25, linewidth=0.3)
        ax.tick_params(labelsize=8)

    plot_panel(axes[0], "mflux", r"900 hPa |q·V|  (g kg$^{-1}$ m s$^{-1}$)", scale=1e3)
    panel_label(axes[0], "a", x=-0.14, y=1.08)
    axes[0].legend(loc="lower center", fontsize=7, ncol=2,
                   bbox_to_anchor=(0.5, 1.02), frameon=False)
    plot_panel(axes[1], "cape",  r"CAPE  (J kg$^{-1}$)")
    panel_label(axes[1], "b", x=-0.14, y=1.08)
    plot_panel(axes[2], "blh",   r"BLH  (m)")
    panel_label(axes[2], "c", x=-0.14, y=1.08)
    plot_panel(axes[3], "omega", r"$\omega_{500}$  (Pa s$^{-1}$)", invert=True)
    panel_label(axes[3], "d", x=-0.14, y=1.08)

    fig.text(0.5, 0.985,
             "Mean diurnal cycle (February) over the Amazon convective zone — 5 El Niño vs 5 La Niña years",
             ha="center", va="top", fontsize=10)
    fig.text(0.5, 0.005, "Local time (Amazon interior) = UTC − 4",
             ha="center", va="bottom", fontsize=7, color="0.45", style="italic")

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
