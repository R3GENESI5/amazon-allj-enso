"""Nature-style figure module for the Amazon ALLJ paper (P4).

Conventions
-----------
* Helvetica/Arial 9 pt body, 10 pt panel labels (bold), no bold titles.
* Column widths: 90 mm (single), 120 mm (1.5), 180 mm (double). 300 DPI.
* Perceptually uniform colormaps via cmocean.
* Panel labels (a, b, ...) bold, top-left, *inside* axes, with a white halo.
* Annotations belong in margins or captions, never on data.
* Maps: PlateCarree, country borders + Amazon-bbox box, light graticules.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patheffects
import cartopy.crs as ccrs
import cartopy.feature as cfeature

MM = 1 / 25.4
COL1, COL15, COL2 = 90 * MM, 120 * MM, 180 * MM

DATA_ROOT = r"D:/amazon paper/data"
ERA5_MONTHLY = os.path.join(DATA_ROOT, "era5", "era5_monthly_means_900hpa_feb_1979_2023.nc")
ONI_CSV = os.path.join(DATA_ROOT, "oni_classification.csv")

AMAZON_BBOX = dict(west=-75, east=-40, south=-15, north=6)
ARC_BBOX = dict(west=-55, east=-45, south=-12, north=-5)
INTERIOR_BBOX = dict(west=-65, east=-55, south=-5, north=0)


def set_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.dpi": 300,
        "figure.dpi": 120,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def panel_label(ax, label: str, *, x: float = 0.02, y: float = 0.97, color: str = "black") -> None:
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10, fontweight="bold", color=color,
        path_effects=[patheffects.withStroke(linewidth=2.0, foreground="white")],
    )


CITIES = [
    ("Belém",   -48.5, -1.45),
    ("Manaus",  -60.0, -3.10),
    ("Santarém", -54.7, -2.44),
    ("Porto Velho", -63.9, -8.76),
]


def add_map_features(
    ax, *, bbox=AMAZON_BBOX,
    draw_arc: bool = False, draw_interior: bool = False,
    draw_rivers: bool = True, draw_cities: bool = False,
) -> None:
    ax.set_extent([bbox["west"], bbox["east"], bbox["south"], bbox["north"]], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5, edgecolor="0.25")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.3, edgecolor="0.5")
    if draw_rivers:
        ax.add_feature(
            cfeature.RIVERS.with_scale("50m"),
            edgecolor="white", linewidth=0.45, alpha=0.7, zorder=3,
        )
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.7", alpha=0.5, linestyle=":")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}
    if draw_arc:
        _draw_box(ax, ARC_BBOX, color="#d95f02", label="arc of deforestation")
    if draw_interior:
        _draw_box(ax, INTERIOR_BBOX, color="#1b9e77", label="intact interior")
    if draw_cities:
        for name, lon, lat in CITIES:
            ax.plot(lon, lat, "o", color="white", markeredgecolor="black",
                    markeredgewidth=0.5, markersize=3.2, transform=ccrs.PlateCarree(), zorder=6)
            ax.text(
                lon + 0.35, lat + 0.25, name,
                fontsize=7, color="black",
                transform=ccrs.PlateCarree(), zorder=7,
                path_effects=[patheffects.withStroke(linewidth=1.6, foreground="white")],
            )


def _draw_box(ax, bbox, *, color: str, label: str | None = None) -> None:
    xs = [bbox["west"], bbox["east"], bbox["east"], bbox["west"], bbox["west"]]
    ys = [bbox["south"], bbox["south"], bbox["north"], bbox["north"], bbox["south"]]
    ax.plot(xs, ys, color=color, linewidth=1.0, transform=ccrs.PlateCarree(), zorder=5)


def discrete_levels(vmin: float, vmax: float, n: int = 11, symmetric: bool = False) -> np.ndarray:
    if symmetric:
        m = max(abs(vmin), abs(vmax))
        return np.linspace(-m, m, n)
    return np.linspace(vmin, vmax, n)


def load_oni() -> pd.DataFrame:
    return pd.read_csv(ONI_CSV)


def classify_years() -> dict[str, list[int]]:
    df = load_oni()
    return {
        "elnino": df.loc[df["phase"] == "El Nino", "year"].tolist(),
        "lanina": df.loc[df["phase"] == "La Nina", "year"].tolist(),
        "neutral": df.loc[df["phase"] == "Neutral", "year"].tolist(),
    }
