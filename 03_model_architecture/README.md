# Stage 3: Model Architecture

## What this does
Defines the two models for the core comparison: RGB-only vs full multispectral.

| Model | Input channels | Pretrained weights |
|---|---|---|
| RGB baseline | 3 (B4,B3,B2) | Full ImageNet |
| Multispectral | 13 (10 bands + NDVI/NDWI/NDBI) | Partial (only first 3 channels) |

## Decisions
| Decision | Choice | Why |
|---|---|---|
| Library | segmentation-models-pytorch | Handles arbitrary in_channels natively; avoids manually patching a pretrained model's first conv layer |
| Architecture | U-Net | Standard for pixel-wise segmentation |
| Encoder | ResNet34 | Good accuracy/speed tradeoff on Colab GPU |
| Classes | 11 (ESA WorldCover taxonomy) | Matches label source; see `WORLDCOVER_CLASSES` in `model.py` |

## Known caveat
ImageNet pretrained weights only apply to 3 channels. For the 13-channel
model, smp Kaiming-initializes the extra 10 channels — the multispectral
model does not get full pretrained benefit. This is expected and should be
noted when interpreting RGB-vs-multispectral results, not treated as a bug.

## Run
```
pip install segmentation-models-pytorch torch
python scripts/model.py
```
Prints parameter counts for both models as a sanity check.

**Next**: `04_training/` — train both models on Google Colab.
