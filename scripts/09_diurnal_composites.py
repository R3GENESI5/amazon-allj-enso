"""
ERA5 hourly diurnal cycle composites for 10 key ENSO years (February).

Memory-efficient: processes one year at a time, stores only diurnal arrays.
4-panel figure:
  (a) 900 hPa moisture flux magnitude at 52W transect (0--5S mean)
  (b) CAPE area mean (2S--8S, 60W--50W)
  (c) BLH area mean (same region)
  (d) 500 hPa omega area mean (same region)

Inputs:  data/era5/era5_pl_YYYY_feb.nc
         data/era5/era5_sfc_YYYY_feb.nc
Outputs: figures/fig08_diurnal_enso.png

Reference: Section 3.8, Figure 8
"""

import warnings
warnings.filterwarnings("ignore")

import os
import gc
import zipfile
import tempfile
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "era5"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ELNINO_YEARS = [1983, 1992, 1998, 2010, 2016]
LANINA_YEARS = [1989, 1999, 2000, 2008, 2011]

C_ELNINO = "#D4443B"
C_LANINA = "#2E86AB"

# Domains
BOX_LAT_S, BOX_LAT_N = -8.0, -2.0
BOX_LON_W, BOX_LON_E = -60.0, -50.0
TRANSECT_LON = -52.0
TRANS_LAT_S, TRANS_LAT_N = -5.0, 0.0
LOCAL_OFFSET = -4

# ── Loaders ──────────────────────────────────────────────────────────
def load_sfc(year):
    fpath = DATA_DIR / f"era5_sfc_{year}_feb.nc"
    with open(fpath, "rb") as f:
        magic = f.read(4)
    if magic == b"PK\x03\x04":
        tmpdir = tempfile.mkdtemp()
        with zipfile.ZipFile(fpath) as zf:
            zf.extractall(tmpdir)
        ds_i = xr.open_dataset(os.path.join(
            tmpdir, "data_stream-oper_stepType-instant.nc"))
        ds_a = xr.open_dataset(os.path.join(
            tmpdir, "data_stream-oper_stepType-accum.nc"))
        return xr.merge([ds_i, ds_a])
    return xr.open_dataset(fpath)

def load_pl(year):
    return xr.open_dataset(DATA_DIR / f"era5_pl_{year}_feb.nc")

def to_diurnal(da_1d):
    vals = da_1d.values
    n = len(vals) // 24
    return np.nanmean(vals[:n * 24].reshape(n, 24), axis=0)

def process_year(yr):
    sfc = load_sfc(yr)
    pl = load_pl(yr)
    lat, lon = "latitude", "longitude"

    box_lats = pl[lat].sel({lat: slice(BOX_LAT_N, BOX_LAT_S)})
    wt_box = np.cos(np.deg2rad(box_lats))

    # (a) Moisture flux at 52W, 0-5S, 900 hPa
    pl900 = pl.sel(pressure_level=900.0, method="nearest")
    q_t = pl900["q"].sel({lon: TRANSECT_LON}, method="nearest").sel(
        {lat: slice(TRANS_LAT_N, TRANS_LAT_S)})
    u_t = pl900["u"].sel({lon: TRANSECT_LON}, method="nearest").sel(
        {lat: slice(TRANS_LAT_N, TRANS_LAT_S)})
    v_t = pl900["v"].sel({lon: TRANSECT_LON}, method="nearest").sel(
        {lat: slice(TRANS_LAT_N, TRANS_LAT_S)})
    mag = np.sqrt((q_t * u_t)**2 + (q_t * v_t)**2)
    wt_trans = np.cos(np.deg2rad(mag[lat]))
    d_mflux = to_diurnal(mag.weighted(wt_trans).mean(dim=lat))

    # (b) CAPE
    cape = sfc["cape"].sel({lat: slice(BOX_LAT_N, BOX_LAT_S),
                            lon: slice(BOX_LON_W, BOX_LON_E)})
    d_cape = to_diurnal(cape.weighted(wt_box).mean(dim=[lat, lon]))

    # (c) BLH
    blh = sfc["blh"].sel({lat: slice(BOX_LAT_N, BOX_LAT_S),
                          lon: slice(BOX_LON_W, BOX_LON_E)})
    d_blh = to_diurnal(blh.weighted(wt_box).mean(dim=[lat, lon]))

    # (d) Omega 500 hPa
    pl500 = pl.sel(pressure_level=500.0, method="nearest")
    w = pl500["w"].sel({lat: slice(BOX_LAT_N, BOX_LAT_S),
                        lon: slice(BOX_LON_W, BOX_LON_E)})
    d_omega = to_diurnal(w.weighted(wt_box).mean(dim=[lat, lon]))

    sfc.close()
    pl.close()
    del sfc, pl, pl900, pl500
    gc.collect()
    return {"mflux": d_mflux, "cape": d_cape, "blh": d_blh, "omega": d_omega}

# ── Compute diurnal cycles ───────────────────────────────────────────
print("Computing diurnal cycles year by year...")
results = {k: {"elnino": [], "lanina": []}
           for k in ["mflux", "cape", "blh", "omega"]}

for yr in ELNINO_YEARS:
    print(f"  El Nino {yr}...")
    d = process_year(yr)
    for k in results:
        results[k]["elnino"].append(d[k])

for yr in LANINA_YEARS:
    print(f"  La Nina {yr}...")
    d = process_year(yr)
    for k in results:
        results[k]["lanina"].append(d[k])

for k in results:
    for phase in ["elnino", "lanina"]:
        results[k][phase] = np.array(results[k][phase])

# ── Figure ───────────────────────────────────────────────────────────
UTC_HOURS = np.arange(24)
fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.0), constrained_layout=True)

def plot_panel(ax, key, ylabel, title, letter, scale=1.0, invert=False):
    for phase, color, label in [("elnino", C_ELNINO, "El Ni\u00f1o"),
                                 ("lanina", C_LANINA, "La Ni\u00f1a")]:
        arr = results[key][phase] * scale
        mn = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0, ddof=0)
        ax.plot(UTC_HOURS, mn, color=color, lw=1.8, label=label, zorder=3)
        ax.fill_between(UTC_HOURS, mn - sd, mn + sd,
                        color=color, alpha=0.13, zorder=2)
    if invert:
        ax.invert_yaxis()

    ax.axvspan(21, 22, alpha=0.12, color="#C25B56", zorder=0)
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"({letter}) {title}", fontsize=9, fontweight="bold",
                 loc="left")
    ax.set_xlim(0, 23)
    ax.set_xticks(np.arange(0, 24, 3))
    ax.legend(loc="best", fontsize=7)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    lt = np.arange(0, 24, 6)
    ax2.set_xticks(lt)
    ax2.set_xticklabels([f"{(h + LOCAL_OFFSET) % 24:02d}" for h in lt],
                        fontsize=6, color="#666666")
    ax2.set_xlabel("Local time (UTC\u22124)", fontsize=6, color="#666666")
    ax2.tick_params(axis="x", length=2)

plot_panel(axes[0, 0], "mflux",
           "|q$\\cdot$V|$_{900}$ (g kg$^{-1}$ m s$^{-1}$)",
           "900 hPa moisture flux at 52\u00b0W", "a", scale=1000.0)
plot_panel(axes[0, 1], "cape", "CAPE (J kg$^{-1}$)",
           "CAPE (2\u20138\u00b0S, 60\u201350\u00b0W)", "b")
plot_panel(axes[1, 0], "blh", "BLH (m)",
           "Boundary layer height", "c")
plot_panel(axes[1, 1], "omega", "\u03c9$_{500}$ (Pa s$^{-1}$)",
           "500 hPa vertical velocity", "d", invert=True)

fig.suptitle("Diurnal cycles during February \u2014 ENSO composites "
             "(5 yr each, \u00b11\u03c3 shading)",
             fontsize=10, fontweight="bold", y=1.02)

outpath = FIG_DIR / "fig08_diurnal_enso.png"
fig.savefig(outpath, dpi=300, facecolor="white", bbox_inches="tight")
plt.close(fig)
print(f"Saved: {outpath}")
print("Script 09 complete.")
