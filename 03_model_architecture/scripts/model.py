"""
model.py — Stage 3 for Project 1 (LULC Classification)

Defines two models for the RGB-vs-multispectral comparison:
  - build_rgb_baseline():      3 input channels (B4,B3,B2), full ImageNet pretrained weights
  - build_multispectral_model(): 13 input channels (10 bands + NDVI/NDWI/NDBI)

Both use U-Net (segmentation-models-pytorch) with a ResNet34 encoder.

Caveat: ImageNet pretrained weights only cover 3 input channels. When
in_channels=13, smp initializes the first conv layer's extra 10 channels
using Kaiming init (not pretrained) — so only partial pretrained benefit
applies to the multispectral model. This is expected and worth noting in
the RGB-vs-multispectral results writeup, not a bug.

Requirements: pip install segmentation-models-pytorch torch
"""

import segmentation_models_pytorch as smp

# ESA WorldCover raw pixel values -> contiguous class indices (0-10)
WORLDCOVER_CLASSES = {
    10: 0,   # Tree cover
    20: 1,   # Shrubland
    30: 2,   # Grassland
    40: 3,   # Cropland
    50: 4,   # Built-up
    60: 5,   # Bare / sparse vegetation
    70: 6,   # Snow and ice
    80: 7,   # Permanent water bodies
    90: 8,   # Herbaceous wetland
    95: 9,   # Mangroves
    100: 10, # Moss and lichen
}
NUM_CLASSES = len(WORLDCOVER_CLASSES)


def remap_labels(label_array):
    """Map raw WorldCover pixel values to contiguous 0-10 class indices."""
    remapped = label_array.copy()
    for raw_val, class_idx in WORLDCOVER_CLASSES.items():
        remapped[label_array == raw_val] = class_idx
    return remapped


def build_rgb_baseline(num_classes=NUM_CLASSES, encoder="resnet34"):
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    )


def build_multispectral_model(num_classes=NUM_CLASSES, encoder="resnet34", in_channels=13):
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet",  # partial: only first 3 channels get pretrained weights
        in_channels=in_channels,
        classes=num_classes,
    )


if __name__ == "__main__":
    rgb_model = build_rgb_baseline()
    ms_model = build_multispectral_model()
    print(f"RGB baseline params: {sum(p.numel() for p in rgb_model.parameters()):,}")
    print(f"Multispectral model params: {sum(p.numel() for p in ms_model.parameters()):,}")
    print(f"Num classes: {NUM_CLASSES}")
