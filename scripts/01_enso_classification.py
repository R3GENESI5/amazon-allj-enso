"""
Classify years by ENSO phase using the Oceanic Nino Index (ONI).

Downloads the ONI index from NOAA CPC and classifies each DJF season
(1979--2023) as El Nino (ONI > 0.5), La Nina (ONI < -0.5), or Neutral.

Inputs:  ONI data from NOAA CPC (downloaded automatically)
Outputs: data/oni_classification.csv
         data/key_enso_years.txt

Reference: Section 2.1
"""

import csv
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Download ONI data from NOAA CPC
url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
print(f"Downloading ONI from {url}...")
response = urllib.request.urlopen(url)
lines = response.read().decode("utf-8").strip().split("\n")

# Parse: columns are SEAS YR TOTAL ANOM
# We want DJF season which represents the Feb state
records = []
for line in lines[1:]:
    parts = line.split()
    if len(parts) >= 4:
        seas = parts[0]
        yr = int(parts[1])
        anom = float(parts[3])
        if seas == "DJF" and 1979 <= yr <= 2023:
            if anom > 0.5:
                phase = "El Nino"
            elif anom < -0.5:
                phase = "La Nina"
            else:
                phase = "Neutral"
            records.append({"year": yr, "season": seas,
                            "oni": anom, "phase": phase})

el_ninos = sorted([r for r in records if r["phase"] == "El Nino"],
                  key=lambda x: -x["oni"])
la_ninas = sorted([r for r in records if r["phase"] == "La Nina"],
                  key=lambda x: x["oni"])
neutrals = [r for r in records if r["phase"] == "Neutral"]

print(f"\nENSO classification for DJF Februaries (1979-2023):")
print(f"  El Nino years: {len(el_ninos)}")
print(f"  La Nina years: {len(la_ninas)}")
print(f"  Neutral years: {len(neutrals)}")

print(f"\nTop 5 strongest El Nino Februaries:")
for r in el_ninos[:5]:
    print(f"  {r['year']}: ONI = {r['oni']:+.2f}")

print(f"\nTop 5 strongest La Nina Februaries:")
for r in la_ninas[:5]:
    print(f"  {r['year']}: ONI = {r['oni']:+.2f}")

# Save to CSV
outpath = DATA_DIR / "oni_classification.csv"
with open(outpath, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["year", "season", "oni", "phase"])
    writer.writeheader()
    for r in sorted(records, key=lambda x: x["year"]):
        writer.writerow(r)
print(f"\nSaved: {outpath}")

# Save key years for hourly download
key_elnino = [r["year"] for r in el_ninos[:5]]
key_lanina = [r["year"] for r in la_ninas[:5]]
print(f"\nKey years for hourly analysis:")
print(f"  El Nino: {key_elnino}")
print(f"  La Nina: {key_lanina}")

keypath = DATA_DIR / "key_enso_years.txt"
with open(keypath, "w") as f:
    f.write(f"elnino: {','.join(map(str, key_elnino))}\n")
    f.write(f"lanina: {','.join(map(str, key_lanina))}\n")
print(f"Saved: {keypath}")
