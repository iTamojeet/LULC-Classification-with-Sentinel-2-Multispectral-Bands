# Stage 2: Preprocessing

## What this does
- Computes NDVI, NDWI, NDBI from the composite → 10 bands become 13 channels
- Verifies image/label pixel alignment
- Tiles both into 128x128 patches
- Splits patches into train/val/test using spatial blocks (4x4 tile blocks
  assigned wholesale to one split) — prevents adjacent tiles from leaking
  across splits, which random-per-tile splitting would allow

## Decisions
| Decision | Choice | Why |
|---|---|---|
| Tile size | 128x128 | Default starting point; open to tuning (64/256) if model performance suggests otherwise |
| Split method | Spatial blocks | Random splitting lets near-identical neighboring tiles appear in both train and test, inflating validation scores |
| Split ratio | 70/15/15 | Standard default |

## Run
```
pip install rasterio numpy
python scripts/preprocess.py
```

## Output
`02_preprocessing/patches/{train,val,test}/{images,labels}/*.npy`

## Result — complete ✅
780 tiles total: train 552 (70.8%), val 112 (14.4%), test 116 (14.9%).
13 channels/tile (10 bands + NDVI/NDWI/NDBI).

**Next**: `03_model_architecture/` — RGB baseline + 13-channel U-Net/ResNet.