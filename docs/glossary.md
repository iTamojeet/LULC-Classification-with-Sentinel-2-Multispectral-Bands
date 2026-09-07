# Glossary

Living document. Terms are added as they come up in each stage's README.
Organized alphabetically. Each entry has a plain-language explanation first,
then technical precision if needed.

---

### Atmospheric correction
Raw satellite sensor readings are contaminated by scattering and absorption
from the atmosphere (haze, water vapor, aerosols) between the ground and the
satellite. Atmospheric correction is a processing step that removes this
distortion, converting raw "top-of-atmosphere" brightness values into
"surface reflectance" — an estimate of what the light would look like if
measured right at ground level. This is why we use **Sentinel-2 L2A** data
(see below) instead of raw L1C data — L2A has already been atmospherically
corrected.

### Band (spectral band)
A single channel of light captured by the satellite sensor at a specific
wavelength range. A normal camera has 3 bands (Red, Green, Blue). Sentinel-2
has 13 bands spanning visible light through short-wave infrared. Each band
is stored as a separate grayscale image; combining bands produces the full
multispectral image.

### Dynamic World
A near-real-time global land cover dataset produced by Google/WRI, updated
continuously from incoming Sentinel-2 imagery using an AI model. Provides
per-pixel class probabilities (9 classes) at 10m resolution. Used here as an
alternative or complement to ESA WorldCover for reference labels.

### ESA WorldCover
A global land cover map produced by the European Space Agency, at 10m
resolution, with 11 land cover classes (tree cover, cropland, built-up,
water, etc.), derived from a full year of Sentinel-1 and Sentinel-2 data.
Used in this project as **ground truth labels** — i.e., the "correct answers"
our model is trained to predict.

### L1C vs L2A (Sentinel-2 processing levels)
- **L1C**: Top-of-atmosphere reflectance — raw-ish, not corrected for
  atmospheric effects.
- **L2A**: Surface (bottom-of-atmosphere) reflectance — atmospherically
  corrected. This project uses L2A exclusively.

### LULC (Land Use / Land Cover)
"Land cover" refers to the physical material on the surface (forest, water,
bare soil, concrete). "Land use" refers to how humans use that land
(residential, agricultural, industrial). Datasets like ESA WorldCover
technically map land *cover*, though the two terms are often used together.

### NDVI (Normalized Difference Vegetation Index)
`NDVI = (NIR - Red) / (NIR + Red)`
A formula combining the Near-Infrared and Red bands. Healthy vegetation
strongly reflects NIR light and absorbs Red light (for photosynthesis), so
NDVI is high (close to +1) over healthy plants and low or negative over
water, bare soil, or built-up areas. One of the oldest and most widely used
remote sensing indices in geography.

### NDWI (Normalized Difference Water Index)
`NDWI = (Green - NIR) / (Green + NIR)`
Highlights open water bodies, which absorb NIR strongly and reflect green
light, giving high NDWI values over water and low values over land.

### NDBI (Normalized Difference Built-up Index)
`NDBI = (SWIR - NIR) / (SWIR + NIR)`
Highlights built-up/urban surfaces, which reflect short-wave infrared (SWIR)
more strongly than near-infrared, unlike vegetation.

### Reflectance
The fraction of incoming sunlight that a surface reflects back, per
wavelength. This is the actual physical quantity satellites measure (after
correction) — different materials have distinct "reflectance signatures"
across wavelengths, which is the whole basis for spectral classification.

### Resampling
Sentinel-2 bands are captured at different native resolutions (10m, 20m,
60m per pixel depending on the band). Resampling converts all bands to a
common resolution (10m in this project) so they can be stacked into one
multi-channel image.

### Sentinel-2
A pair of European Space Agency satellites that image the entire Earth's
land surface every 5 days, capturing 13 spectral bands at 10-60m resolution.
The primary data source for this project.

### Spectral index
A formula that combines two or more bands to highlight a specific
land-surface property (vegetation, water, built-up area). NDVI, NDWI, and
NDBI (above) are all spectral indices.
