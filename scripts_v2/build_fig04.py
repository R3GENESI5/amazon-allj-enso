"""Fig 4 — February 500 hPa vertical velocity (omega) composites by ENSO phase."""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import netCDF4
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import patheffects
import cartopy.crs as ccrs
import cmocean
from scipy import stats

from fig_style_v2 import set_style, panel_label, add_map_features, classify_years, COL2

ERA5_UPPER = r"D:/amazon paper/data/era5/era5_monthly_means_upper_feb_1979_2023.nc"
OUT_PNG = r"D:/Projects/research-programme/manuscripts/amazon/p4_allj_enso/figures_v2/fig04_omega500.png"
OUT_PDF = OUT_PNG.replace(".png", ".pdf")


def load_omega_500():
    ds = netCDF4.Dataset(ERA5_UPPER)
    vt = netCDF4.num2date(ds.variables["valid_time"][:], ds.variables["valid_time"].units,
                          only_use_cftime_datetimes=False)
    lat = ds.variables["latitude"][:]
    lon = ds.variables["longitude"][:]
    plevs = ds.variables["pressure_level"][:]
    iz = int(np.argmin(np.abs(plevs - 500)))
    w = ds.variables["w"][:, iz]  # Pa/s
    ds.close()
    years = np.array([t.year for t in vt])
    months = np.array([t.month for t in vt])
    feb = months == 2
    out = {}
    for y in np.unique(years[feb]):
        m = feb & (years == y)
        out[y] = np.asarray(w[m]).mean(axis=0)
    return lat, lon, out


def composite(field_by_year, yrs):
    arr = np.stack([field_by_year[y] for y in yrs if y in field_by_year])
    return arr.mean(0), arr


def draw_panel(ax, lat, lon, w, label, *, vmin, vmax, draw_boxes=False):
    LON, LAT = np.meshgrid(lon, lat)
    pcm = ax.pcolormesh(LON, LAT, w * 100,  # to 10^-2 Pa/s
                        cmap=cmocean.cm.curl, vmin=vmin, vmax=vmax,
                        transform=ccrs.PlateCarree(), shading="auto", rasterized=True)
    add_map_features(ax, draw_arc=draw_boxes, draw_interior=draw_boxes)
    panel_label(ax, label)
    return pcm


def draw_diff(ax, lat, lon, dw, p, label, *, vlim):
    LON, LAT = np.meshgrid(lon, lat)
    pcm = ax.pcolormesh(LON, LAT, dw * 100,
                        cmap=cmocean.cm.curl, vmin=-vlim, vmax=vlim,
                        transform=ccrs.PlateCarree(), shading="auto", rasterized=True)
    add_map_features(ax, draw_arc=True, draw_interior=True)
    sig = p < 0.05
    LON_s, LAT_s = np.meshgrid(lon[::8], lat[::8])
    sig_s = sig[::8, ::8]
    ax.scatter(LON_s[sig_s], LAT_s[sig_s], s=0.5, color="black", alpha=0.4,
               transform=ccrs.PlateCarree())
    panel_label(ax, label)
    return pcm


def main():
    set_style()
    print("loading ERA5 upper-level...")
    lat, lon, w = load_omega_500()
    years = classify_years()
    en_m, en_arr = composite(w, years["elnino"])
    ln_m, ln_arr = composite(w, years["lanina"])
    nt_m, _      = composite(w, years["neutral"])
    diff = en_m - ln_m
    p = stats.ttest_ind(en_arr, ln_arr, axis=0, equal_var=False).pvalue

    stack = np.stack([en_m, ln_m, nt_m]) * 100
    vmax = float(np.nanpercentile(np.abs(stack), 98))
    vlim = float(np.nanpercentile(np.abs(diff * 100), 98))

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(COL2, COL2 * 0.78))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           left=0.05, right=0.97, top=0.94, bottom=0.10,
                           wspace=0.12, hspace=0.18)
    axes = [fig.add_subplot(gs[i, j], projection=proj) for i in range(2) for j in range(2)]

    pcm_abs = draw_panel(axes[0], lat, lon, en_m, "a", vmin=-vmax, vmax=vmax)
    axes[0].set_title(f"El Niño  (n = {len(years['elnino'])})", fontsize=9)
    draw_panel(axes[1], lat, lon, ln_m, "b", vmin=-vmax, vmax=vmax)
    axes[1].set_title(f"La Niña  (n = {len(years['lanina'])})", fontsize=9)
    draw_panel(axes[2], lat, lon, nt_m, "c", vmin=-vmax, vmax=vmax)
    axes[2].set_title(f"Neutral  (n = {len(years['neutral'])})", fontsize=9)
    pcm_d = draw_diff(axes[3], lat, lon, diff, p, "d", vlim=vlim)
    axes[3].set_title("El Niño − La Niña", fontsize=9)

    cax1 = fig.add_axes([0.07, 0.05, 0.40, 0.018])
    cb1 = fig.colorbar(pcm_abs, cax=cax1, orientation="horizontal")
    cb1.set_label(r"$\omega_{500}$  (10$^{-2}$ Pa s$^{-1}$,  positive = subsidence)", fontsize=8)
    cb1.ax.tick_params(labelsize=7)

    cax2 = fig.add_axes([0.55, 0.05, 0.40, 0.018])
    cb2 = fig.colorbar(pcm_d, cax=cax2, orientation="horizontal")
    cb2.set_label(r"$\Delta\omega_{500}$  (10$^{-2}$ Pa s$^{-1}$)", fontsize=8)
    cb2.ax.tick_params(labelsize=7)

    fig.text(0.5, 0.985,
             "February 500 hPa vertical velocity (ω) composites by ENSO phase, 1979–2023",
             ha="center", va="top", fontsize=10)

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
