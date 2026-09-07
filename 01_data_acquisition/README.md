# Stage 1: Data Acquisition

## Goal
Pull two aligned datasets for the same geographic area and time window:
1. **Sentinel-2 L2A imagery** (the input — what the model sees)
2. **Reference land cover labels** (the target — what the model should predict)

## Why two separate sources?
Sentinel-2 satellites capture raw reflectance data — they don't tell you
"this pixel is a forest." A label dataset (ESA WorldCover or Dynamic World)
is a *separately produced* map that already classifies land cover, usually
made by a mix of expert rules, other AI models, and validation surveys. We
use it as **ground truth**: our model looks at the Sentinel-2 pixels and
tries to learn to predict what WorldCover/Dynamic World already labeled.

This is standard practice in remote sensing ML — you rarely hand-label
satellite imagery yourself from scratch; you piggyback on existing
authoritative land cover products.

## Key decisions made

| Decision | Choice | Why |
|---|---|---|
| Sentinel-2 processing level | L2A (surface reflectance) | Already atmospherically corrected — see glossary. Using raw L1C would mean atmospheric haze/aerosols contaminate the spectral signal, corrupting indices like NDVI. |
| Bands used | B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 | Covers visible (B2-B4), red-edge (B5-B7), NIR (B8, B8A), and SWIR (B11, B12). This is the full useful spectral range Sentinel-2 offers at ≤20m native resolution. B1 (coastal aerosol), B9 (water vapor), B10 (cirrus) are excluded — they're designed for atmospheric correction itself, not land surface classification. |
| Label source | ESA WorldCover (primary), Dynamic World (secondary/validation) | WorldCover gives a stable, well-validated annual map. Dynamic World gives near-real-time probabilistic labels — useful for a sanity-check comparison later. |
| Resolution | 10m | Matches WorldCover's native resolution and Sentinel-2's finest band resolution (B2,B3,B4,B8 are natively 10m; others need resampling — see Stage 2). |
| Data access method | Google Earth Engine (GEE) | Avoids downloading massive raw satellite archives locally — GEE lets us query, filter, and export only the exact area/time/bands we need, server-side. |

## What "study area and time window" means (and why it matters)

Sentinel-2 covers the *entire planet*, repeatedly, since 2015. Before pulling
any data we must define:
- **AOI (Area of Interest)** — a bounding box or polygon of the region to study
- **Time window** — since land cover changes seasonally (crops grow/get
  harvested, leaves fall), and clouds vary by season, we need a defined date
  range, usually a few months, to get a mostly cloud-free composite

**This is the one open decision before I can write acquisition code** — see
the question below.

## Study area (AOI) — decided

**Kolkata Metropolitan Area + East Kolkata Wetlands**, covering:
- Hooghly River (open water)
- Dense urban core of Kolkata (built-up)
- East Kolkata Wetlands — a Ramsar-designated wetland site that naturally
  treats a large share of the city's sewage through fish ponds and
  agriculture (locally called *bheris*) — a strong geography talking point
  beyond just "diverse land cover"
- Wetland-adjacent cropland and urban green patches

This gives good class diversity (water, built-up, cropland, vegetation,
wetland) in a single compact ~areas, with a clear geographic narrative for
Honors framing.

**Time window**: Nov–Feb (dry/winter season in this region) — minimizes
monsoon cloud cover for a cleaner composite.

## Google Earth Engine setup — done

- **Project ID**: `n8nlovetppanu`
- Registered for **noncommercial use**, on the **Community Tier** (free)
- Verified with a test query (`USGS/SRTMGL1_003`) returning valid image
  metadata in the Code Editor console

## Data storage decision: local disk, not Google Drive

The Earth Engine Python API's standard export method
(`Export.image.toDrive`) routes files through Google Drive. For this AOI
at 10m resolution across 10 bands, the composite is roughly **~230MB**
(≈12M pixels × 10 bands × 2 bytes) and the WorldCover labels are **~12MB**
— manageable individually, but with 8 projects in this portfolio, letting
every export land in Drive by default adds up fast and clutters an account
used for other things.

**Decision**: `fetch_sentinel2.py` uses `geemap.ee_export_image()` instead,
which streams data directly from Earth Engine's servers to local disk —
no Drive folder involved at all. For larger regions, geemap automatically
splits the download into tiles server-side and stitches them back together
transparently.

**Downstream data handling (GitHub/Colab)**: for now, data stays local only
and gets uploaded to Colab manually per session when full training runs are
needed. Once the pipeline reaches the training stage, the plan is to push
the processed data alongside the Colab notebooks (`.ipynb` files) to GitHub
together, rather than deciding on a raw-data storage strategy prematurely.
This is a deliberate "decide when you have more information" call — noted
here so the reasoning isn't lost by the time Stage 4 (training) starts.

**Note for future stages**: raw multiband GeoTIFFs of this size are too
large for a plain git repo without Git LFS (GitHub hard-blocks files over
100MB). This is worth revisiting explicitly at that point rather than
finding out from a rejected push.

## Gotcha encountered: `median()` silently balloons file size

While building this stage, the composite export failed with:
```
Total request size (1172440440 bytes) must be less than or equal to 50331648 bytes.
```

**Root cause**: calling `.median()` on an `ImageCollection` upconverts pixel
values from the source's native `int16` to `float64` — 4x the storage per
pixel — even though nothing about surface reflectance values requires that
precision. A composite estimated at ~230MB actually came out to ~1.1GB.

**Fix**: cast the composite back to `int16` with `.toInt16()` immediately
after computing the median, since Sentinel-2 surface reflectance values fit
comfortably in that range. This is now done in `fetch_sentinel2.py`.

**Secondary issue this exposed**: Earth Engine caps any single direct-download
request at 48MB (50,331,648 bytes) — `geemap.ee_export_image()` makes one
such request and fails outright if the image is larger, rather than
partially succeeding. `geemap.download_ee_image()` was used instead, which
automatically tiles a large image into requests under that cap and stitches
the results into one local file — this is the correct tool for any image
export whose size isn't tiny and known in advance.

**Takeaway for later projects**: any reducer that changes a data type
(median, mean) should be checked against the source band's dtype before
export, not assumed to preserve it.

## Stage 1 result — complete ✅

Final downloaded files, in `01_data_acquisition/data/` (not tracked in git —
see storage decision above):

| File | Size | Description |
|---|---|---|
| `kolkata_sentinel2_composite.tif` | 183.5 MB | 10-band Sentinel-2 L2A median composite, int16, Nov 2025–Feb 2026, 10m resolution |
| `kolkata_worldcover_labels.tif` | 2.5 MB | ESA WorldCover v200 labels, same AOI, same resolution |

Both downloaded via `geemap.download_ee_image()`, which depends on the
`geedim` package for automatic tiling (140 tiles for the composite, 14 for
the labels) — installed as an implicit dependency, now added to the
requirements list above explicitly.

**Next stage**: `02_preprocessing/` — resample/align bands if needed,
compute NDVI/NDWI/NDBI as auxiliary channels, tile both files into
fixed-size patches (e.g. 64x64 or 128x128) for model training.

## Workflow for this stage

1. Define AOI + time window (pending your input)
2. Authenticate to Google Earth Engine
3. Query Sentinel-2 L2A collection, filter by AOI/date/cloud cover
4. Create a median composite (reduces cloud noise by taking the per-pixel
   median across many cloud-filtered scenes)
5. Query matching ESA WorldCover labels for the same AOI
6. Export both as GeoTIFF files, aligned to the same grid
7. Save a preview screenshot of the AOI + RGB composite + label map here in
   this folder for documentation

## Files in this folder
- `scripts/` — GEE Python API scripts for data pull (to be added)
- `README.md` — this file