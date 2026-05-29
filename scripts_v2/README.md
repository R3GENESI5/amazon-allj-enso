# ALLJ — figure build scripts

Source code for the nine figures in **"ENSO Modulation of the Amazonian Low-Level Jet: More Moisture, Less Rain, and the Role of Land Surface Reception"** (Shahid 2026, EarthArXiv preprint).

Updated plotting code (added in release v1.0.2). Same composites as the original `scripts/` pipeline; only the figure layer was rewritten with cmocean colormaps, Amazon river overlays, basin/city labels, and margin-positioned annotations so no text overlaps data. All versions are archived under the Zenodo concept DOI [10.5281/zenodo.19209347](https://doi.org/10.5281/zenodo.19209347).

## Layout

| File | Output |
|---|---|
| `fig_style_v2.py` | Shared Matplotlib style (Helvetica/Arial 9 pt, cmocean colormaps, basin overlays). All build scripts import this. |
| `build_fig01.py` | Fig 1 — Feb 900 hPa moisture flux composites by ENSO phase |
| `build_fig02.py` | Fig 2 — CHIRPS coast-to-interior transect by ENSO phase |
| `build_fig03.py` | Fig 3 — GRACE/GRACE-FO Amazon-basin TWS anomaly, 2002–2025 |
| `build_fig04.py` | Fig 4 — Feb 500 hPa ω composites by ENSO phase |
| `build_fig05.py` | Fig 5 — Coast-to-interior moisture flux + precip + retention ratio |
| `build_fig06.py` | Fig 6 — Deforestation signal: arc vs intact in CHIRPS + GRACE |
| `build_fig07.py` | Fig 7 — SMAP L4 root-zone SM residence time, arc vs intact |
| `build_fig08.py` | Fig 8 — Diurnal cycles (moisture flux, CAPE, BLH, ω) by ENSO phase |
| `build_fig09.py` | Fig 9 — Nocturnal flux ↔ next-day CAPE priming correlation |

Output figures land in `../figures_v2/` as 300 DPI PNG plus PDF.

## Running

Each script is standalone:

```bash
python build_fig01.py
```

Fig 8 and Fig 9 cache intermediate diurnal arrays in `../figures_v2/_fig0X_cache.npz` after the first run; subsequent runs are instant.

To rebuild everything:

```bash
for n in 01 02 03 04 05 06 07 08 09; do python build_fig${n}.py; done
```

## Data paths (hard-coded)

| Variable | Path |
|---|---|
| `ERA5_MONTHLY` | `D:/amazon paper/data/era5/era5_monthly_means_900hpa_feb_1979_2023.nc` |
| `ERA5_UPPER` | `D:/amazon paper/data/era5/era5_monthly_means_upper_feb_1979_2023.nc` |
| ERA5 hourly | `D:/amazon paper/data/era5/era5_{sfc,pl}_{YEAR}_feb.nc` (10 ENSO years) |
| `CHIRPS_DIR` | `D:/amazon paper/data/chirps/chirps-v2.0.{YEAR}.monthly.nc` |
| `GRACE_NC` | `D:/amazon paper/data/grace/GRCTellus.JPL.200204_202512.GLO.RL06.3M.MSCNv04CRI.nc` |
| `SMAP_NC` | `D:/amazon paper/data/smap/smap_l4_rootzone_sm_monthly_amazon_2015_2024.nc` |
| `ONI_CSV` | `D:/amazon paper/data/oni_classification.csv` |

DOIs for each dataset are in the **Data and Code Availability** section of the manuscript.

## Dependencies

`numpy`, `pandas`, `xarray`, `netCDF4`, `matplotlib`, `cartopy`, `cmocean`, `scipy`, `statsmodels`. Tested with Python 3.12 on Windows.

## License

MIT.
