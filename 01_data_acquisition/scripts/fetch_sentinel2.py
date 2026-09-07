"""
fetch_sentinel2.py

Stage 1 acquisition script for Project 1 (LULC Classification).

What this does:
1. Authenticates to Google Earth Engine (GEE) using your registered project.
2. Defines the AOI: Kolkata Metropolitan Area + East Kolkata Wetlands.
3. Pulls Sentinel-2 L2A imagery for Nov-Feb (dry season), filters cloudy
   scenes, and builds a median composite across 10 spectral bands.
4. Pulls matching ESA WorldCover land cover labels for the same AOI.
5. Downloads both as GeoTIFFs DIRECTLY TO LOCAL DISK — no Google Drive
   involved. See 01_data_acquisition/README.md for why this approach was
   chosen over Export.image.toDrive.

Run this from a local Python environment (Mac M4). This step is CPU-only
(no GPU needed) — it's just querying and downloading, not training.

Requirements:
    pip install earthengine-api geemap geedim

Before running:
    - You must have completed GEE registration (see 01_data_acquisition/README.md)
    - Replace PROJECT_ID below if it ever changes
"""

import os
import ee
import geemap

# ---------------------------------------------------------------------------
# 1. AUTHENTICATION
# ---------------------------------------------------------------------------
# First-time run: this opens a browser window to authenticate your Google
# account. After that, credentials are cached locally and you won't need to
# re-authenticate every run.

PROJECT_ID = "n8nlovetppanu"  # Tamojeet's registered GEE project ID

ee.Authenticate()  # opens browser auth flow on first run; no-op after that
ee.Initialize(project=PROJECT_ID)

print(f"Earth Engine initialized with project: {PROJECT_ID}")


# ---------------------------------------------------------------------------
# 2. DEFINE AREA OF INTEREST (AOI)
# ---------------------------------------------------------------------------
# Bounding box roughly covering Kolkata Metropolitan Area + East Kolkata
# Wetlands. This is a rectangle for simplicity — precise enough for a first
# model. (Coordinates are [longitude, latitude] pairs, GEE convention.)
#
# Rough bounds:
#   West  -> 88.20  (west of central Kolkata)
#   East  -> 88.55  (east, past the wetlands)
#   South -> 22.40  (south of wetlands)
#   North -> 22.70  (north Kolkata / Dum Dum area)

aoi = ee.Geometry.Rectangle([88.20, 22.40, 88.55, 22.70])

print("AOI defined: Kolkata Metropolitan Area + East Kolkata Wetlands")


# ---------------------------------------------------------------------------
# 3. SENTINEL-2 L2A IMAGE COLLECTION
# ---------------------------------------------------------------------------
# Band list explained in docs/glossary.md — covers visible, red-edge, NIR,
# and SWIR. We exclude atmospheric-correction-only bands (B1, B9, B10).

BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

# Cloud filtering: Sentinel-2 L2A includes a per-scene cloud percentage
# metadata field. We filter to scenes with <20% cloud cover, then take a
# median composite across all remaining scenes to further suppress residual
# cloud/shadow noise on a per-pixel basis.

s2_collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(aoi)
    .filterDate("2025-11-01", "2026-02-28")  # dry season window
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    .select(BANDS)
)

print(f"Sentinel-2 scenes matching filter: {s2_collection.size().getInfo()}")

# Median composite: for each pixel, take the median value across all
# remaining cloud-filtered scenes. This is a simple, standard way to produce
# a clean, mostly cloud-free single image from many time-stamped scenes.
s2_composite = s2_collection.median().clip(aoi)

# IMPORTANT: median() upconverts pixel values to float64 (8 bytes/pixel)
# even though the source Sentinel-2 bands are natively int16. Left
# uncorrected, this roughly quadruples file size (a ~230MB composite
# becomes ~1.1GB) and can push a direct download over Earth Engine's
# per-request size limit. Surface reflectance values comfortably fit in
# int16, so we cast back down before exporting.
s2_composite = s2_composite.toInt16()

print("Sentinel-2 median composite created (cast to int16 to control file size).")


# ---------------------------------------------------------------------------
# 4. ESA WORLDCOVER LABELS
# ---------------------------------------------------------------------------
# WorldCover v200 covers 2021 — the most recent version available at time of
# writing. It's a single global static map (not time-filtered like Sentinel-2).

worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().clip(aoi)

print("ESA WorldCover labels loaded for AOI.")


# ---------------------------------------------------------------------------
# 5. DOWNLOAD DIRECTLY TO LOCAL DISK (no Google Drive involved)
# ---------------------------------------------------------------------------
# geemap.download_ee_image() (NOT ee_export_image) is used here deliberately.
# Earth Engine caps any single direct-download request at 48MB
# (50,331,648 bytes). download_ee_image() automatically splits a larger
# image into tiles under that cap, downloads each tile, and stitches them
# back into one local GeoTIFF — handling exactly the case we hit during
# development, where a naive single-request download failed outright for
# an image just over that limit.

export_scale = 10  # meters per pixel, matches WorldCover + finest S2 bands

output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(output_dir, exist_ok=True)

s2_local_path = os.path.join(output_dir, "kolkata_sentinel2_composite.tif")
label_local_path = os.path.join(output_dir, "kolkata_worldcover_labels.tif")

print("Downloading Sentinel-2 composite to local disk (tiled automatically if needed)...")
geemap.download_ee_image(
    s2_composite,
    filename=s2_local_path,
    scale=export_scale,
    region=aoi,
    crs="EPSG:4326",
)

print("Downloading WorldCover labels to local disk...")
geemap.download_ee_image(
    worldcover,
    filename=label_local_path,
    scale=export_scale,
    region=aoi,
    crs="EPSG:4326",
)

print(f"Done. Files saved locally at:")
print(f"  {os.path.abspath(s2_local_path)}")
print(f"  {os.path.abspath(label_local_path)}")
print("Nothing was written to Google Drive.")