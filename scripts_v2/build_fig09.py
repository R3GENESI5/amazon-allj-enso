"""Fig 9 — Nocturnal moisture flux priming of next-day afternoon CAPE.

For each day N in the 10 selected ENSO Februaries, compute:
  - mean nocturnal 900 hPa moisture flux magnitude (00–06 UTC, day N)
  - mean afternoon CAPE (15–21 UTC, day N+1)
Pair (flux_N, CAPE_N+1) → Pearson r by ENSO phase.
"""

from __future__ import annotations

import os, sys, gc, zipfile, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from fig_style_v2 import set_style, panel_label, COL2

DATA_DIR = r"D:/amazon paper/data/era5"
OUT_PNG = r"D:/Projects/research-programme/manuscripts/amazon/p4_allj_enso/figures_v2/fig09_priming.png"
OUT_PDF = OUT_PNG.replace(".png", ".pdf")
CACHE = r"D:/Projects/research-programme/manuscripts/amazon/p4_allj_enso/figures_v2/_fig09_cache.npz"

EN = [1983, 1992, 1998, 2010, 2016]
LN = [1989, 1999, 2000, 2008, 2011]
C_EN = "#d62828"; C_LN = "#1d4e89"; C_GREY = "#777777"

LAT = (-5.0, 0.0); LON = (-60.0, -50.0)


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


def pairs_for_year(y):
    sfc = load_sfc(y); pl = load_pl(y)
    pl900 = pl.sel(pressure_level=900.0, method="nearest")
    q = pl900["q"].sel(latitude=slice(LAT[1], LAT[0]), longitude=slice(LON[0], LON[1]))
    u = pl900["u"].sel(latitude=slice(LAT[1], LAT[0]), longitude=slice(LON[0], LON[1]))
    v = pl900["v"].sel(latitude=slice(LAT[1], LAT[0]), longitude=slice(LON[0], LON[1]))
    mf_mag = np.sqrt((q * u) ** 2 + (q * v) ** 2)
    mf_mean = mf_mag.mean(dim=["latitude", "longitude"])
    cape = sfc["cape"].sel(latitude=slice(LAT[1], LAT[0]), longitude=slice(LON[0], LON[1]))
    cape_mean = cape.mean(dim=["latitude", "longitude"])

    t = mf_mean["valid_time"]
    hrs_mf = t.dt.hour.values
    dys_mf = t.dt.day.values
    t2 = cape_mean["valid_time"]
    hrs_c = t2.dt.hour.values
    dys_c = t2.dt.day.values

    days = sorted(set(dys_mf))
    out_flux, out_cape = [], []
    for i, d in enumerate(days[:-1]):
        nd = days[i + 1]
        m_n = (dys_mf == d) & (hrs_mf >= 0) & (hrs_mf <= 6)
        m_a = (dys_c == nd) & (hrs_c >= 15) & (hrs_c <= 21)
        if m_n.sum() and m_a.sum():
            out_flux.append(float(np.nanmean(mf_mean.values[m_n])))
            out_cape.append(float(np.nanmean(cape_mean.values[m_a])))
    sfc.close(); pl.close(); del sfc, pl; gc.collect()
    return np.array(out_flux), np.array(out_cape)


def compute_all():
    if os.path.exists(CACHE):
        print(f"loading cache {CACHE}")
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}
    en_f, en_c, ln_f, ln_c = [], [], [], []
    for y in EN:
        print(f"  EN {y}...")
        f, c = pairs_for_year(y); en_f.append(f); en_c.append(c)
    for y in LN:
        print(f"  LN {y}...")
        f, c = pairs_for_year(y); ln_f.append(f); ln_c.append(c)
    out = dict(
        en_f=np.concatenate(en_f), en_c=np.concatenate(en_c),
        ln_f=np.concatenate(ln_f), ln_c=np.concatenate(ln_c),
    )
    np.savez(CACHE, **out)
    return out


def main():
    set_style()
    print("loading priming pairs (cached)...")
    a = compute_all()
    en_f = a["en_f"] * 1000; en_c = a["en_c"]
    ln_f = a["ln_f"] * 1000; ln_c = a["ln_c"]
    all_f = np.concatenate([en_f, ln_f]); all_c = np.concatenate([en_c, ln_c])

    def corr(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        r, p = stats.pearsonr(x[m], y[m])
        return r, p, m.sum()

    r_en, p_en, n_en = corr(en_f, en_c)
    r_ln, p_ln, n_ln = corr(ln_f, ln_c)
    r_al, p_al, n_al = corr(all_f, all_c)

    fig = plt.figure(figsize=(COL2, COL2 * 0.42))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.6, 1],
                           left=0.07, right=0.97, top=0.90, bottom=0.18, wspace=0.30)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    ax1.scatter(en_f, en_c, color=C_EN, s=10, alpha=0.45, edgecolor="none",
                label=f"El Niño  (n = {n_en})", zorder=3)
    ax1.scatter(ln_f, ln_c, color=C_LN, s=10, alpha=0.45, edgecolor="none",
                label=f"La Niña  (n = {n_ln})", zorder=3)
    for fx, fy, c in [(en_f, en_c, C_EN), (ln_f, ln_c, C_LN)]:
        m = np.isfinite(fx) & np.isfinite(fy)
        slope, intercept = np.polyfit(fx[m], fy[m], 1)
        xf = np.linspace(np.nanmin(fx), np.nanmax(fx), 80)
        ax1.plot(xf, slope * xf + intercept, color=c, linewidth=1.2, linestyle="--", zorder=4)
    ax1.set_xlabel("Nocturnal moisture flux, day N\n(g kg$^{-1}$ m s$^{-1}$, 00–06 UTC)")
    ax1.set_ylabel("Afternoon CAPE, day N+1\n(J kg$^{-1}$, 15–21 UTC)")
    ax1.legend(loc="upper left", fontsize=7, markerscale=1.6)
    ax1.grid(True, alpha=0.25, linewidth=0.3)
    ax1.tick_params(labelsize=8)
    panel_label(ax1, "a")

    cats = ["El Niño", "La Niña", "All years"]
    rvals = [r_en, r_ln, r_al]
    pvals = [p_en, p_ln, p_al]
    bcols = [C_EN, C_LN, C_GREY]
    bars = ax2.bar(cats, rvals, color=bcols, width=0.55,
                   edgecolor="black", linewidth=0.4, alpha=0.85)
    for i, (rv, pv) in enumerate(zip(rvals, pvals)):
        star = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else "n.s."
        off = 0.015 if rv >= 0 else -0.015
        va = "bottom" if rv >= 0 else "top"
        ax2.text(i, rv + off, f"{rv:.3f}\n{star}",
                 ha="center", va=va, fontsize=7, fontweight="bold")
    ax2.set_ylabel("Pearson r")
    ax2.axhline(0, color="0.3", linewidth=0.5)
    ax2.set_ylim(min(rvals) - 0.12, max(rvals) + 0.14)
    ax2.grid(True, axis="y", alpha=0.25, linewidth=0.3)
    ax2.tick_params(labelsize=8)
    panel_label(ax2, "b")

    fig.text(0.5, 0.97,
             "Nocturnal moisture flux primes next-day afternoon CAPE — sign reverses across ENSO phase",
             ha="center", va="top", fontsize=10)

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"saved {OUT_PNG}  (EN r={r_en:.3f} p={p_en:.3f}; LN r={r_ln:.3f} p={p_ln:.3f})")


if __name__ == "__main__":
    main()
