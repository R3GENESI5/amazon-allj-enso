# Data sources

This directory should contain the following datasets. They are not included in
the repository due to size. Use `scripts/00_download_data.py` to download
them automatically, or follow the manual instructions below.

## Required datasets

### 1. ONI Classification (auto-generated)

Run `scripts/01_enso_classification.py` to download the ONI index from NOAA
and produce `data/oni_classification.csv`.

### 2. ERA5 Monthly Means (900 hPa and upper levels)

- Source: Copernicus Climate Data Store (CDS)
- Product: `reanalysis-era5-pressure-levels-monthly-means`
- Variables: u, v, specific humidity (900 hPa); vertical velocity, temperature (300/500/700 hPa)
- Period: February 1979--2023
- Registration: https://cds.climate.copernicus.eu/

Place files in `data/era5/`.

### 3. ERA5 Hourly (10 key ENSO years)

- Source: CDS
- Product: `reanalysis-era5-pressure-levels` and `reanalysis-era5-single-levels`
- Variables: u, v, q at 900 hPa; omega at 500 hPa; CAPE, BLH, SLHF
- Years: El Nino (1983, 1992, 1998, 2010, 2016), La Nina (1989, 1999, 2000, 2008, 2011)
- Month: February only
- Registration: https://cds.climate.copernicus.eu/

Place files in `data/era5/`.

### 4. CHIRPS v2.0

- Source: Climate Hazards Center, UC Santa Barbara
- URL: https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/netcdf/byYear/
- Files: `chirps-v2.0.YYYY.monthly.nc` for 1981--2024
- No registration required

Place files in `data/chirps/`.

### 5. GRACE/GRACE-FO TWS

- Source: NASA PODAAC
- Dataset: TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.1_V3
- File: `GRCTellus.JPL.*.GLO.RL06.*.MSCNv04CRI.nc`
- Registration: https://urs.earthdata.nasa.gov/

Place files in `data/grace/`.

### 6. SMAP L4 Root-Zone Soil Moisture

- Source: NASA Earthdata (NSIDC DAAC)
- Product: SPL4SMGP v008
- The download script produces a monthly-mean Amazon subset via OPeNDAP
- Registration: https://urs.earthdata.nasa.gov/

Place files in `data/smap/`.
