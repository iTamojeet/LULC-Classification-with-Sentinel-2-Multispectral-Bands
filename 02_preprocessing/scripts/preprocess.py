"""
preprocess.py — Stage 2 for Project 1 (LULC Classification)

1. Loads Sentinel-2 composite + WorldCover labels
2. Computes NDVI, NDWI, NDBI as extra channels (13 total input channels)
3. Verifies pixel alignment between image and labels
4. Tiles both into 128x128 patches
5. Splits patches into train/val/test using spatial blocks (not random) to
   prevent adjacent-tile leakage between splits
6. Saves patches as .npy arrays

Requirements: pip install rasterio numpy scikit-learn
"""

import os
import numpy as np
import rasterio

TILE_SIZE = 128
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
GRID_BLOCK = 4  # NxN tile blocks assigned wholesale to one split

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "01_data_acquisition", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "patches")

COMPOSITE_PATH = os.path.join(DATA_DIR, "kolkata_sentinel2_composite.tif")
LABELS_PATH = os.path.join(DATA_DIR, "kolkata_worldcover_labels.tif")


def compute_indices(bands):
    # bands order: B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12 (indices 0-9)
    blue, green, red = bands[0], bands[1], bands[2]
    nir, swir1 = bands[6], bands[8]
    eps = 1e-6
    ndvi = (nir - red) / (nir + red + eps)
    ndwi = (green - nir) / (green + nir + eps)
    ndbi = (swir1 - nir) / (swir1 + nir + eps)
    return np.stack([ndvi, ndwi, ndbi], axis=0)


def load_data():
    with rasterio.open(COMPOSITE_PATH) as src:
        composite = src.read().astype(np.float32)
        composite_transform = src.transform
        composite_shape = (src.height, src.width)

    with rasterio.open(LABELS_PATH) as src:
        labels = src.read(1)
        label_shape = (src.height, src.width)

    if composite_shape != label_shape:
        raise ValueError(
            f"Shape mismatch: composite {composite_shape} vs labels {label_shape}. "
            "Re-export both at the same scale/region before tiling."
        )

    return composite, labels, composite_transform


def tile_arrays(image, labels, tile_size):
    """Yield (row_block, col_block, image_tile, label_tile) for full tiles only."""
    _, h, w = image.shape
    n_rows, n_cols = h // tile_size, w // tile_size
    for r in range(n_rows):
        for c in range(n_cols):
            y0, y1 = r * tile_size, (r + 1) * tile_size
            x0, x1 = c * tile_size, (c + 1) * tile_size
            yield r, c, image[:, y0:y1, x0:x1], labels[y0:y1, x0:x1]


def assign_split(row_block, col_block, grid_block):
    """Deterministically assign a spatial block of tiles to train/val/test."""
    return BLOCK_SPLIT_MAP[(row_block // grid_block, col_block // grid_block)]


def build_block_split_map(image_shape, tile_size, grid_block, seed=42):
    """Enumerate all spatial blocks, shuffle deterministically, assign
    proportionally to train/val/test. Fixes the modulo-hash approach, which
    doesn't guarantee correct ratios over a small number of blocks."""
    _, h, w = image_shape
    n_rows, n_cols = h // tile_size, w // tile_size
    n_block_rows = (n_rows // grid_block) + 1
    n_block_cols = (n_cols // grid_block) + 1

    blocks = [(br, bc) for br in range(n_block_rows) for bc in range(n_block_cols)]
    rng = np.random.RandomState(seed)
    rng.shuffle(blocks)

    n = len(blocks)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])

    split_map = {}
    for i, block in enumerate(blocks):
        if i < n_train:
            split_map[block] = "train"
        elif i < n_train + n_val:
            split_map[block] = "val"
        else:
            split_map[block] = "test"
    return split_map


def main():
    composite, labels, _ = load_data()
    indices = compute_indices(composite)
    full_stack = np.concatenate([composite, indices], axis=0)  # 13 channels

    global BLOCK_SPLIT_MAP
    BLOCK_SPLIT_MAP = build_block_split_map(full_stack.shape, TILE_SIZE, GRID_BLOCK)

    for split in SPLIT_RATIOS:
        os.makedirs(os.path.join(OUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, split, "labels"), exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}
    for r, c, img_tile, lbl_tile in tile_arrays(full_stack, labels, TILE_SIZE):
        split = assign_split(r, c, GRID_BLOCK)
        tile_id = f"r{r}_c{c}"
        np.save(os.path.join(OUT_DIR, split, "images", f"{tile_id}.npy"), img_tile)
        np.save(os.path.join(OUT_DIR, split, "labels", f"{tile_id}.npy"), lbl_tile)
        counts[split] += 1

    print(f"Total input channels per tile: {full_stack.shape[0]}")
    print(f"Tiles created — train: {counts['train']}, val: {counts['val']}, test: {counts['test']}")
    print(f"Saved to: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()