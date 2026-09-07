"""
evaluate.py — Stage 5 for Project 1 (LULC Classification)

Loads both trained models, runs on the test split, reports:
  - Per-class IoU and pixel accuracy
  - Overall mean IoU
  - Confusion matrix
  - Class pixel distribution (to explain imbalance-driven results)

Run in Colab, same session as training (models already in memory) or
standalone by loading checkpoints from Drive.
"""

import os
import numpy as np
import torch
from collections import Counter

PATCH_DIR = "/content/drive/MyDrive/patches"
CHECKPOINT_DIR = "/content/drive/MyDrive/patches/checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "Tree cover", "Shrubland", "Grassland", "Cropland", "Built-up",
    "Bare/sparse veg", "Snow/ice", "Water", "Wetland", "Mangroves", "Moss/lichen",
]
NUM_CLASSES = len(CLASS_NAMES)


def load_test_set(channels_mode):
    from train import TileDataset  # reuse Stage 4 dataset class
    return TileDataset(PATCH_DIR, "test", channels=channels_mode)


def check_class_distribution(dataset):
    counts = Counter()
    for i in range(len(dataset)):
        _, label = dataset[i]
        for c in label.unique().tolist():
            counts[c] += (label == c).sum().item()
    total = sum(counts.values())
    print("Class pixel distribution in test set:")
    for c in range(NUM_CLASSES):
        pct = 100 * counts.get(c, 0) / total if total else 0
        print(f"  {CLASS_NAMES[c]:20s}: {pct:5.2f}%")
    return counts


def compute_iou_per_class(model, dataset, batch_size=8):
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()

    intersection = np.zeros(NUM_CLASSES)
    union = np.zeros(NUM_CLASSES)
    correct = 0
    total = 0
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            labels = labels.numpy()

            for c in range(NUM_CLASSES):
                pred_c = preds == c
                label_c = labels == c
                intersection[c] += np.logical_and(pred_c, label_c).sum()
                union[c] += np.logical_or(pred_c, label_c).sum()

            correct += (preds == labels).sum()
            total += labels.size

            for t, p in zip(labels.flatten(), preds.flatten()):
                confusion[t, p] += 1

    iou_per_class = intersection / np.maximum(union, 1)
    pixel_accuracy = correct / total
    mean_iou = iou_per_class[union > 0].mean()  # only over classes present

    return {
        "iou_per_class": iou_per_class,
        "mean_iou": mean_iou,
        "pixel_accuracy": pixel_accuracy,
        "confusion": confusion,
    }


def print_results(name, results):
    print(f"\n=== {name} ===")
    print(f"Pixel accuracy: {results['pixel_accuracy']:.4f}")
    print(f"Mean IoU (present classes): {results['mean_iou']:.4f}")
    print("Per-class IoU:")
    for c in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[c]:20s}: {results['iou_per_class'][c]:.4f}")


if __name__ == "__main__":
    import sys
    sys.path.append("../../03_model_architecture/scripts")
    from model import build_rgb_baseline, build_multispectral_model

    test_ds_rgb = load_test_set("rgb")
    test_ds_ms = load_test_set("all")

    check_class_distribution(test_ds_ms)

    rgb_model = build_rgb_baseline().to(DEVICE)
    rgb_model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "rgb_baseline_best.pt")))
    rgb_results = compute_iou_per_class(rgb_model, test_ds_rgb)
    print_results("RGB baseline", rgb_results)

    ms_model = build_multispectral_model().to(DEVICE)
    ms_model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "multispectral_best.pt")))
    ms_results = compute_iou_per_class(ms_model, test_ds_ms)
    print_results("Multispectral", ms_results)

    print(f"\nDelta (multispectral - RGB) mean IoU: {ms_results['mean_iou'] - rgb_results['mean_iou']:.4f}")
