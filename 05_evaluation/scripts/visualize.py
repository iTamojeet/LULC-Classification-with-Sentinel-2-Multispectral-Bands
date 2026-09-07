"""
visualize.py — Stage 5 addition for Project 1 (LULC Classification)

Generates 4 presentation visuals, entirely local (CPU inference is fine at
this scale):
  1. study_area.png            — RGB composite of the AOI
  2. prediction_examples.png   — input | ground truth | RGB pred | MS pred, for tiles with water/wetland
  3. confusion_matrices.png    — normalized confusion matrix, both models
  4. iou_comparison.png        — grouped bar chart, RGB vs multispectral IoU per class

Requirements: pip install rasterio torch segmentation-models-pytorch matplotlib
"""

import os
import sys
import numpy as np
import torch
import rasterio
import matplotlib.pyplot as plt

sys.path.append("../../03_model_architecture/scripts")
from model import build_rgb_baseline, build_multispectral_model, remap_labels

DATA_DIR = "../../01_data_acquisition/data"
PATCH_DIR = "../../02_preprocessing/patches"
CHECKPOINT_DIR = "../../04_training/checkpoints"
OUT_DIR = "../outputs"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["Tree cover", "Shrubland", "Grassland", "Cropland", "Built-up",
               "Bare/sparse veg", "Snow/ice", "Water", "Wetland", "Mangroves", "Moss/lichen"]
NUM_CLASSES = len(CLASS_NAMES)

CLASS_COLORS = np.array([
    [34, 139, 34], [128, 128, 0], [154, 205, 50], [255, 215, 0], [128, 128, 128],
    [210, 180, 140], [255, 255, 255], [30, 144, 255], [0, 128, 128], [85, 107, 47], [211, 211, 211],
]) / 255.0


def colorize(label_2d):
    return CLASS_COLORS[label_2d]


def load_model(name, build_fn):
    model = build_fn().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, f"{name}_best.pt"), map_location=DEVICE))
    model.eval()
    return model


def predict(model, image_13ch, rgb_only=False):
    img = image_13ch[[2, 1, 0], :, :] if rgb_only else image_13ch
    with torch.no_grad():
        t = torch.from_numpy(img).float().unsqueeze(0).to(DEVICE)
        pred = model(t).argmax(dim=1).squeeze(0).cpu().numpy()
    return pred


def stretch(rgb):
    lo, hi = np.percentile(rgb, (2, 98))
    return np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)


# ---------------------------------------------------------------------------
# 1. Study area map
# ---------------------------------------------------------------------------
def plot_study_area():
    with rasterio.open(os.path.join(DATA_DIR, "kolkata_sentinel2_composite.tif")) as src:
        bands = src.read([3, 2, 1]).astype(np.float32)  # B4,B3,B2 -> RGB
    rgb = stretch(np.transpose(bands, (1, 2, 0)))
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb)
    plt.title("Study Area: Kolkata Metropolitan Area + East Kolkata Wetlands")
    plt.axis("off")
    plt.savefig(os.path.join(OUT_DIR, "study_area.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved study_area.png")


# ---------------------------------------------------------------------------
# 2. Prediction examples (tiles with water/wetland)
# ---------------------------------------------------------------------------
def plot_prediction_examples(rgb_model, ms_model, n_examples=3):
    label_dir = os.path.join(PATCH_DIR, "test", "labels")
    image_dir = os.path.join(PATCH_DIR, "test", "images")
    files = sorted(os.listdir(label_dir))

    scored = []
    for f in files:
        raw_label = np.load(os.path.join(label_dir, f))
        water_wetland_frac = np.isin(raw_label, [80, 90]).mean()
        scored.append((water_wetland_frac, f))
    scored.sort(reverse=True)
    chosen = [f for _, f in scored[:n_examples]]

    fig, axes = plt.subplots(len(chosen), 4, figsize=(16, 4 * len(chosen)))
    col_titles = ["Input (RGB)", "Ground truth", "RGB-model pred", "Multispectral pred"]

    for row, fname in enumerate(chosen):
        image = np.load(os.path.join(image_dir, fname))
        raw_label = np.load(os.path.join(label_dir, fname))
        label = remap_labels(raw_label)

        rgb_img = stretch(np.transpose(image[[2, 1, 0], :, :], (1, 2, 0)))
        pred_rgb_model = predict(rgb_model, image, rgb_only=True)
        pred_ms_model = predict(ms_model, image, rgb_only=False)

        panels = [rgb_img, colorize(label), colorize(pred_rgb_model), colorize(pred_ms_model)]
        for col, (panel, title) in enumerate(zip(panels, col_titles)):
            ax = axes[row, col] if len(chosen) > 1 else axes[col]
            ax.imshow(panel)
            ax.axis("off")
            if row == 0:
                ax.set_title(title)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "prediction_examples.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved prediction_examples.png")


# ---------------------------------------------------------------------------
# 3 & 4. Confusion matrices + IoU bar chart (single pass over test set)
# ---------------------------------------------------------------------------
def evaluate_full(model, rgb_only):
    label_dir = os.path.join(PATCH_DIR, "test", "labels")
    image_dir = os.path.join(PATCH_DIR, "test", "images")
    files = sorted(os.listdir(label_dir))

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    intersection = np.zeros(NUM_CLASSES)
    union = np.zeros(NUM_CLASSES)

    for fname in files:
        image = np.load(os.path.join(image_dir, fname))
        label = remap_labels(np.load(os.path.join(label_dir, fname)))
        pred = predict(model, image, rgb_only=rgb_only)

        for t, p in zip(label.flatten(), pred.flatten()):
            confusion[t, p] += 1
        for c in range(NUM_CLASSES):
            pred_c, label_c = pred == c, label == c
            intersection[c] += np.logical_and(pred_c, label_c).sum()
            union[c] += np.logical_or(pred_c, label_c).sum()

    iou = intersection / np.maximum(union, 1)
    return confusion, iou, union


def plot_confusion_matrices(rgb_conf, ms_conf, present_classes):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, conf, title in zip(axes, [rgb_conf, ms_conf], ["RGB baseline", "Multispectral"]):
        sub = conf[np.ix_(present_classes, present_classes)].astype(np.float32)
        sub = sub / np.maximum(sub.sum(axis=1, keepdims=True), 1)  # row-normalize
        im = ax.imshow(sub, cmap="Blues", vmin=0, vmax=1)
        labels = [CLASS_NAMES[c] for c in present_classes]
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved confusion_matrices.png")


def plot_iou_comparison(rgb_iou, ms_iou, present_classes):
    labels = [CLASS_NAMES[c] for c in present_classes]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, rgb_iou[present_classes], width, label="RGB baseline")
    plt.bar(x + width / 2, ms_iou[present_classes], width, label="Multispectral")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("IoU")
    plt.title("Per-class IoU: RGB baseline vs Multispectral")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "iou_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved iou_comparison.png")


if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    plot_study_area()

    rgb_model = load_model("rgb_baseline", build_rgb_baseline)
    ms_model = load_model("multispectral", build_multispectral_model)

    plot_prediction_examples(rgb_model, ms_model)

    print("Running full test-set evaluation for confusion matrices + IoU chart...")
    rgb_conf, rgb_iou, rgb_union = evaluate_full(rgb_model, rgb_only=True)
    ms_conf, ms_iou, ms_union = evaluate_full(ms_model, rgb_only=False)

    present_classes = [c for c in range(NUM_CLASSES) if rgb_union[c] > 0 or ms_union[c] > 0]

    plot_confusion_matrices(rgb_conf, ms_conf, present_classes)
    plot_iou_comparison(rgb_iou, ms_iou, present_classes)

    print(f"\nAll visuals saved to {os.path.abspath(OUT_DIR)}")