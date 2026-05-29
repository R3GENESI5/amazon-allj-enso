# ENSO Modulation of the Amazonian Low-Level Jet

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19209348.svg)](https://doi.org/10.5281/zenodo.19209348)

Code and analysis scripts for:

**Shahid, A. B., et al. (2026). ENSO Modulation of the Amazonian Low-Level Jet: More Moisture, Less Rain, and the Role of Land Surface Reception.**

## Citation

If you use this code, please cite:

> Shahid, A. B., et al. (2026). Analysis code for: ENSO Modulation of the Amazonian Low-Level Jet. Zenodo. https://doi.org/10.5281/zenodo.19209348

## Requirements

- Python 3.9+
- See `environment.yml` for full dependencies

## Data

See `data/README.md` for instructions on obtaining the required datasets.

## Layout

This repository contains two generations of code:

- **`scripts/`** — Original analysis pipeline (v1.0.0 / v1.0.1). 12 numbered scripts that download raw data, classify ENSO years, compute composites, and produce the first round of figures.
- **`scripts_v2/`** — Updated plotting code (added in v1.0.2). Same underlying datasets and composites; rewritten figure layer with Helvetica/Arial typography, cmocean perceptually uniform colormaps, Amazon river overlays, basin/city labels, and margin-positioned annotations so no text overlaps data. Output figures are 300 DPI PNG + PDF in `figures_v2/`. See `scripts_v2/README.md`.

If you want to reproduce the latest figures, use `scripts_v2/`. If you want the original analysis pipeline that downloads and ingests the raw data, use `scripts/`. All versions archived under Zenodo concept DOI [10.5281/zenodo.19209347](https://doi.org/10.5281/zenodo.19209347).

## Reproducing the original analysis

Scripts are numbered in order of execution:

1. `00_download_data.py` -- Download all required datasets (requires API credentials for CDS, Earthdata)
2. `01_enso_classification.py` -- Classify years by ENSO phase using the ONI index
3. `02_moisture_flux_composites.py` -- 45-year ERA5 moisture flux composites by ENSO phase (Fig. 1)
4. `03_chirps_transect.py` -- CHIRPS precipitation coast-to-interior transect (Fig. 2)
5. `04_grace_tws.py` -- GRACE terrestrial water storage anomaly time series (Fig. 3)
6. `05_subsidence_omega.py` -- 500 hPa omega (vertical velocity) composites (Fig. 4)
7. `06_moisture_budget.py` -- Moisture flux convergence and retention analysis (Fig. 5)
8. `07_deforestation_signal.py` -- Deforestation-era precipitation and TWS trends (Fig. 6)
9. `08_smap_residence_time.py` -- SMAP soil moisture residence time analysis (Fig. 7)
10. `09_diurnal_composites.py` -- ERA5 hourly diurnal cycle composites (Fig. 8)
11. `10_priming_correlation.py` -- Nocturnal moisture flux priming of next-day CAPE (Fig. 9)
12. `11_nocturnal_partition.py` -- Nocturnal vs daytime moisture flux partition (Fig. 10)

All scripts use relative paths and can be run from the repository root:

```bash
cd repo/
python scripts/01_enso_classification.py
```

Shared domain definitions are in `scripts/domains.py`.

## Attribution

Analysis assisted by Claude (Anthropic). All scientific decisions, interpretations, and conclusions are the responsibility of the authors.

## License

MIT License
