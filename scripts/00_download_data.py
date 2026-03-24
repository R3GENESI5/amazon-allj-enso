"""
Download all datasets required for the ALLJ paper.

This script orchestrates downloads from multiple sources:
  - ERA5 monthly means (CDS API) -- 900 hPa and upper levels
  - ERA5 hourly data for 10 key ENSO years (CDS API)
  - CHIRPS v2.0 monthly precipitation (UCSB HTTP)
  - GRACE/GRACE-FO mascon TWS (NASA Earthdata)
  - SMAP L4 root-zone soil moisture (NASA Earthdata OPeNDAP)

Prerequisites:
  - CDS API key: ~/.cdsapirc  (https://cds.climate.copernicus.eu/)
  - NASA Earthdata: ~/.netrc   (https://urs.earthdata.nasa.gov/)

Outputs:
  data/era5/    -- ERA5 NetCDF files
  data/chirps/  -- CHIRPS monthly NetCDF files
  data/grace/   -- GRACE mascon NetCDF
  data/smap/    -- SMAP monthly-mean Amazon subset

Reference: Section 2 (Data and Methods)
"""

import os
import sys
import time
import calendar
import pathlib
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

ERA5_DIR = DATA_DIR / "era5"
CHIRPS_DIR = DATA_DIR / "chirps"
GRACE_DIR = DATA_DIR / "grace"
SMAP_DIR = DATA_DIR / "smap"

for d in [ERA5_DIR, CHIRPS_DIR, GRACE_DIR, SMAP_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ===================================================================
# 1. ERA5 MONTHLY MEANS (requires cdsapi)
# ===================================================================
def download_era5_monthly():
    """Download ERA5 monthly-averaged-by-hour-of-day for all Februaries."""
    try:
        import cdsapi
    except ImportError:
        print("WARNING: cdsapi not installed. Skipping ERA5 downloads.")
        print("  Install with: pip install cdsapi")
        return

    c = cdsapi.Client()
    ALL_YEARS = [str(y) for y in range(1979, 2024)]
    ALL_HOURS = [f"{h:02d}:00" for h in range(24)]
    AREA = [15, -80, -20, -30]  # N, W, S, E

    # 900 hPa winds + specific humidity
    f1 = ERA5_DIR / "era5_monthly_means_900hpa_feb_1979_2023.nc"
    if not f1.exists():
        print("[ERA5] Downloading 900 hPa (u, v, q) monthly means...")
        c.retrieve(
            "reanalysis-era5-pressure-levels-monthly-means",
            {
                "product_type": "monthly_averaged_reanalysis_by_hour_of_day",
                "variable": ["u_component_of_wind", "v_component_of_wind",
                             "specific_humidity"],
                "pressure_level": "900",
                "year": ALL_YEARS, "month": "02", "time": ALL_HOURS,
                "area": AREA, "format": "netcdf",
            },
            str(f1),
        )
        print(f"  Saved: {f1}")
    else:
        print(f"[ERA5] Already have: {f1.name}")

    # Upper levels: 500 hPa omega + temperature
    f2 = ERA5_DIR / "era5_monthly_means_upper_feb_1979_2023.nc"
    if not f2.exists():
        print("[ERA5] Downloading upper-level (omega, T) monthly means...")
        c.retrieve(
            "reanalysis-era5-pressure-levels-monthly-means",
            {
                "product_type": "monthly_averaged_reanalysis_by_hour_of_day",
                "variable": ["vertical_velocity", "temperature"],
                "pressure_level": ["300", "500", "700"],
                "year": ALL_YEARS, "month": "02", "time": ALL_HOURS,
                "area": AREA, "format": "netcdf",
            },
            str(f2),
        )
        print(f"  Saved: {f2}")
    else:
        print(f"[ERA5] Already have: {f2.name}")


# ===================================================================
# 2. ERA5 HOURLY (10 key ENSO years)
# ===================================================================
def download_era5_hourly():
    """Download ERA5 hourly PL and surface data for key ENSO Februaries."""
    try:
        import cdsapi
    except ImportError:
        return

    c = cdsapi.Client()
    ELNINO_YEARS = [1983, 1992, 1998, 2010, 2016]
    LANINA_YEARS = [1989, 1999, 2000, 2008, 2011]
    ALL_YEARS = ELNINO_YEARS + LANINA_YEARS
    ALL_HOURS = [f"{h:02d}:00" for h in range(24)]
    AREA = [6, -75, -12, -40]

    def feb_days(year):
        n = calendar.monthrange(year, 2)[1]
        return [f"{d:02d}" for d in range(1, n + 1)]

    for year in ALL_YEARS:
        # Pressure levels
        out_pl = ERA5_DIR / f"era5_pl_{year}_feb.nc"
        if not out_pl.exists():
            print(f"[ERA5] PL {year}...")
            c.retrieve("reanalysis-era5-pressure-levels", {
                "product_type": "reanalysis",
                "variable": ["u_component_of_wind", "v_component_of_wind",
                             "specific_humidity", "vertical_velocity"],
                "pressure_level": ["500", "900"],
                "year": str(year), "month": "02",
                "day": feb_days(year), "time": ALL_HOURS,
                "area": AREA, "format": "netcdf",
            }, str(out_pl))

        # Surface
        out_sfc = ERA5_DIR / f"era5_sfc_{year}_feb.nc"
        if not out_sfc.exists():
            print(f"[ERA5] SFC {year}...")
            c.retrieve("reanalysis-era5-single-levels", {
                "product_type": "reanalysis",
                "variable": ["convective_available_potential_energy",
                             "boundary_layer_height",
                             "surface_latent_heat_flux"],
                "year": str(year), "month": "02",
                "day": feb_days(year), "time": ALL_HOURS,
                "area": AREA, "format": "netcdf",
            }, str(out_sfc))


# ===================================================================
# 3. CHIRPS v2.0
# ===================================================================
def download_chirps():
    """Download CHIRPS v2.0 global monthly precipitation NetCDF files."""
    BASE_URL = ("https://data.chc.ucsb.edu/products/CHIRPS-2.0/"
                "global_monthly/netcdf/byYear/")

    for year in range(1981, 2025):
        filename = f"chirps-v2.0.{year}.monthly.nc"
        dest = CHIRPS_DIR / filename
        if dest.exists() and dest.stat().st_size > 10_000_000:
            continue
        url = BASE_URL + filename
        print(f"[CHIRPS] {filename}...")
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "Mozilla/5.0")
            with urllib.request.urlopen(req, timeout=120) as resp:
                with open(str(dest), "wb") as f:
                    while True:
                        chunk = resp.read(256 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
            print(f"  OK ({dest.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:
            print(f"  FAILED: {e}")


# ===================================================================
# 4. GRACE/GRACE-FO
# ===================================================================
def download_grace():
    """Download GRACE-FO JPL mascon data (requires Earthdata credentials)."""
    FILENAME = ("GRCTellus.JPL.200204_202312.GLO.RL06.1M."
                "MSCNv04CRI.nc")
    dest = GRACE_DIR / FILENAME
    if dest.exists():
        print(f"[GRACE] Already have: {FILENAME}")
        return

    LANDING = ("https://podaac.jpl.nasa.gov/dataset/"
               "TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.1_V3")

    print(f"[GRACE] Automated download requires NASA Earthdata credentials.")
    print(f"  Please download manually from:\n  {LANDING}")
    print(f"  Place the file in: {GRACE_DIR}")


# ===================================================================
# 5. SMAP
# ===================================================================
def download_smap():
    """Download SMAP L4 root-zone soil moisture (requires Earthdata)."""
    dest = SMAP_DIR / "smap_l4_rootzone_sm_monthly_amazon_2015_2024.nc"
    if dest.exists():
        print(f"[SMAP] Already have: {dest.name}")
        return

    print("[SMAP] SMAP download requires NASA Earthdata credentials")
    print("  and the requests + netCDF4 packages.")
    print("  Run the full SMAP download script separately if needed.")
    print("  See data/README.md for manual download instructions.")


# ===================================================================
# MAIN
# ===================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ALLJ Paper -- Data Download")
    print("=" * 60)

    print("\n--- ERA5 Monthly Means ---")
    download_era5_monthly()

    print("\n--- ERA5 Hourly (ENSO years) ---")
    download_era5_hourly()

    print("\n--- CHIRPS ---")
    download_chirps()

    print("\n--- GRACE ---")
    download_grace()

    print("\n--- SMAP ---")
    download_smap()

    print("\n" + "=" * 60)
    print("Download script complete.")
    print("See data/README.md for any datasets requiring manual download.")
