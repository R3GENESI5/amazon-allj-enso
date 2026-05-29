"""Fig 1 — February 900 hPa moisture flux composites by ENSO phase.

Four panels:
  (a) El Niño composite     (b) La Niña composite
  (c) Neutral composite     (d) El Niño minus La Niña (with significance stippling)

Inputs
------
* D:/amazon paper/data/era5/era5_monthly_means_900hpa_feb_1979_2023.nc
  - dims: valid_time (1080 = 45 yr × 24 hour-of-day), pressure_level (1=900 hPa),
          latitude, longitude
  - vars: u, v, q
* D:/amazon paper/data/oni_classification.csv
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import netCDF4
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import patheffects
import cartopy.crs as ccrs
import cmocean
from scipy import stats

from fig_style_v2 import (
    set_style, panel_label, add_map_features,
    classify_years, ERA5_MONTHLY, COL2,
)

OUT_PNG = r"D:/Projects/research-programme/manuscripts/amazon/p4_allj_enso/figures_v2/fig01_moisture_flux_composites.png"
OUT_PDF = OUT_PNG.replace(".png", ".pdf")


def load_yearly_flux():
    """Return dict with year-indexed Feb-mean q*u and q*v fields, plus lat/lon."""
    ds = netCDF4.Dataset(ERA5_MONTHLY)
    vt = netCDF4.num2date(
        ds.variables["valid_time"][:],
        ds.variables["valid_time"].units,
        only_use_cftime_datetimes=False,
    )
    lat = ds.variables["latitude"][:]
    lon = ds.variables["longitude"][:]
    u = ds.variables["u"][:, 0]   # (time, lat, lon)
    v = ds.variables["v"][:, 0]
    q = ds.variables["q"][:, 0]
    ds.close()

    years = np.array([t.year for t in vt])
    months = np.array([t.month for t in vt])
    feb_mask = months == 2

    yrs = np.unique(years[feb_mask])
    fu = {}
    fv = {}
    qm = {}
    for y in yrs:
        m = feb_mask & (years == y)
        # Average across the 24 hour-of-day samples for this year's Feb monthly-mean.
        u_y = np.asarray(u[m]).mean(axis=0)
        v_y = np.asarray(v[m]).mean(axis=0)
        q_y = np.asarray(q[m]).mean(axis=0)
        fu[y] = q_y * u_y * 1e3   # kg/kg * m/s -> g/kg * m/s
        fv[y] = q_y * v_y * 1e3
        qm[y] = q_y
    return lat, lon, fu, fv, qm


def composite(field_by_year: dict, years: list[int]) -> np.ndarray:
    arr = np.stack([field_by_year[y] for y in years if y in field_by_year])
    return arr.mean(axis=0), arr


def t_test_diff(arr_a: np.ndarray, arr_b: np.ndarray) -> np.ndarray:
    t, p = stats.ttest_ind(arr_a, arr_b, axis=0, equal_var=False)
    return p


QUIVER_KW = dict(
    scale=1400, width=0.0022, headwidth=4.0, headlength=4.5, headaxislength=4.0,
)
QUIVER_KW_DIFF = dict(
    scale=600, width=0.0022, headwidth=4.0, headlength=4.5, headaxislength=4.0,
)


def draw_map_panel(ax, lat, lon, fu, fv, label, *, cmap, vmin, vmax,
                   draw_boxes=False, draw_cities=False, ref_arrow=False):
    mag = np.hypot(fu, fv)
    LON, LAT = np.meshgrid(lon, lat)
    pcm = ax.pcolormesh(
        LON, LAT, mag,
        cmap=cmap, vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree(), shading="auto", rasterized=True,
    )
    add_map_features(ax, draw_arc=draw_boxes, draw_interior=draw_boxes, draw_cities=draw_cities)
    step = 12
    q = ax.quiver(
        LON[::step, ::step], LAT[::step, ::step],
        fu[::step, ::step], fv[::step, ::step],
        transform=ccrs.PlateCarree(),
        color="white", edgecolor="0.15", linewidth=0.3,
        **QUIVER_KW,
    )
    if ref_arrow:
        qk = ax.quiverkey(
            q, X=0.84, Y=0.94, U=100,
            label=r"100 g kg$^{-1}$ m s$^{-1}$",
            labelpos="W", fontproperties={"size": 7}, color="black",
        )
        qk.text.set_path_effects([patheffects.withStroke(linewidth=2.0, foreground="white")])
    panel_label(ax, label)
    return pcm


def draw_diff_panel(ax, lat, lon, du, dv, p, label, *, vlim, draw_cities=False):
    LON, LAT = np.meshgrid(lon, lat)
    pcm = ax.pcolormesh(
        LON, LAT, du,
        cmap=cmocean.cm.balance, vmin=-vlim, vmax=vlim,
        transform=ccrs.PlateCarree(), shading="auto", rasterized=True,
    )
    add_map_features(ax, draw_arc=True, draw_interior=True, draw_cities=draw_cities)
    step = 12
    q = ax.quiver(
        LON[::step, ::step], LAT[::step, ::step],
        du[::step, ::step], dv[::step, ::step],
        transform=ccrs.PlateCarree(),
        color="black",
        **QUIVER_KW_DIFF,
    )
    qk = ax.quiverkey(
        q, X=0.84, Y=0.94, U=20,
        label=r"20 g kg$^{-1}$ m s$^{-1}$",
        labelpos="W", fontproperties={"size": 7}, color="black",
    )
    qk.text.set_path_effects([patheffects.withStroke(linewidth=2.0, foreground="white")])
    sig = p < 0.05
    LON_s, LAT_s = np.meshgrid(lon[::8], lat[::8])
    sig_s = sig[::8, ::8]
    ax.scatter(
        LON_s[sig_s], LAT_s[sig_s],
        s=0.5, color="black", alpha=0.45, transform=ccrs.PlateCarree(),
    )
    panel_label(ax, label, color="black")
    return pcm


def main() -> None:
    set_style()
    print("loading ERA5...")
    lat, lon, fu, fv, qm = load_yearly_flux()
    years = classify_years()

    en_u_mean, en_u_arr = composite(fu, years["elnino"])
    en_v_mean, en_v_arr = composite(fv, years["elnino"])
    ln_u_mean, ln_u_arr = composite(fu, years["lanina"])
    ln_v_mean, ln_v_arr = composite(fv, years["lanina"])
    nu_u_mean, _ = composite(fu, years["neutral"])
    nu_v_mean, _ = composite(fv, years["neutral"])

    print(f"  n El Niño = {len(years['elnino'])}, n La Niña = {len(years['lanina'])}, "
          f"n Neutral = {len(years['neutral'])}")

    diff_u = en_u_mean - ln_u_mean
    diff_v = en_v_mean - ln_v_mean
    p_u = t_test_diff(en_u_arr, ln_u_arr)

    mag_all = np.hypot(
        np.stack([en_u_mean, ln_u_mean, nu_u_mean]),
        np.stack([en_v_mean, ln_v_mean, nu_v_mean]),
    )
    vmax = float(np.nanpercentile(mag_all, 99))
    diff_lim = float(np.nanpercentile(np.abs(diff_u), 98))

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(COL2, COL2 * 0.78))
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        left=0.05, right=0.97, top=0.94, bottom=0.10,
        wspace=0.12, hspace=0.18,
    )
    ax_a = fig.add_subplot(gs[0, 0], projection=proj)
    ax_b = fig.add_subplot(gs[0, 1], projection=proj)
    ax_c = fig.add_subplot(gs[1, 0], projection=proj)
    ax_d = fig.add_subplot(gs[1, 1], projection=proj)

    cmap = cmocean.cm.dense

    pcm_abs = draw_map_panel(ax_a, lat, lon, en_u_mean, en_v_mean, "a",
                             cmap=cmap, vmin=0, vmax=vmax, draw_cities=True)
    ax_a.set_title(f"El Niño  (n = {len(years['elnino'])})", fontsize=9)

    draw_map_panel(ax_b, lat, lon, ln_u_mean, ln_v_mean, "b",
                   cmap=cmap, vmin=0, vmax=vmax)
    ax_b.set_title(f"La Niña  (n = {len(years['lanina'])})", fontsize=9)

    draw_map_panel(ax_c, lat, lon, nu_u_mean, nu_v_mean, "c",
                   cmap=cmap, vmin=0, vmax=vmax, ref_arrow=True)
    ax_c.set_title(f"Neutral  (n = {len(years['neutral'])})", fontsize=9)

    pcm_diff = draw_diff_panel(ax_d, lat, lon, diff_u, diff_v, p_u, "d",
                               vlim=diff_lim)
    ax_d.set_title("El Niño − La Niña", fontsize=9)

    cax1 = fig.add_axes([0.07, 0.05, 0.40, 0.018])
    cb1 = fig.colorbar(pcm_abs, cax=cax1, orientation="horizontal")
    cb1.set_label(r"|q·V|$_{900}$  (g kg$^{-1}$ m s$^{-1}$)", fontsize=8)
    cb1.ax.tick_params(labelsize=7)

    cax2 = fig.add_axes([0.55, 0.05, 0.40, 0.018])
    cb2 = fig.colorbar(pcm_diff, cax=cax2, orientation="horizontal")
    cb2.set_label(r"Δ zonal moisture flux  (g kg$^{-1}$ m s$^{-1}$)", fontsize=8)
    cb2.ax.tick_params(labelsize=7)

    fig.text(0.5, 0.985,
             "February 900 hPa moisture flux composites by ENSO phase, 1979–2023",
             ha="center", va="top", fontsize=10)

    print(f"saving {OUT_PNG}")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print("done")


if __name__ == "__main__":
    main()
