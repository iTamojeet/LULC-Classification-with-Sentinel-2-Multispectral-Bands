# Stage 5: Evaluation

## What this does
- Reports test-set class pixel distribution (explains loss behavior via imbalance)
- Per-class IoU + pixel accuracy for both models
- Confusion matrix
- Direct RGB vs multispectral comparison — the core result of Project 1

## Colab — run as 4 cells (same session as training, or fresh with checkpoints already in Drive)

**Cell 1 — class names + config**
```python
import os, numpy as np, torch
from collections import Counter

CLASS_NAMES = ["Tree cover","Shrubland","Grassland","Cropland","Built-up",
               "Bare/sparse veg","Snow/ice","Water","Wetland","Mangroves","Moss/lichen"]
NUM_CLASSES = len(CLASS_NAMES)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

**Cell 2 — class distribution check**
```python
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

test_ds_ms = TileDataset(PATCH_DIR, "test", channels="all")
check_class_distribution(test_ds_ms)
```

**Cell 3 — IoU/accuracy function**
```python
from torch.utils.data import DataLoader

def compute_iou_per_class(model, dataset, batch_size=8):
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
    mean_iou = iou_per_class[union > 0].mean()
    return {"iou_per_class": iou_per_class, "mean_iou": mean_iou,
            "pixel_accuracy": pixel_accuracy, "confusion": confusion}

def print_results(name, results):
    print(f"\n=== {name} ===")
    print(f"Pixel accuracy: {results['pixel_accuracy']:.4f}")
    print(f"Mean IoU (present classes): {results['mean_iou']:.4f}")
    print("Per-class IoU:")
    for c in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[c]:20s}: {results['iou_per_class'][c]:.4f}")
```

**Cell 4 — run evaluation on both models**
```python
test_ds_rgb = TileDataset(PATCH_DIR, "test", channels="rgb")

rgb_model = build_rgb_baseline().to(DEVICE)
rgb_model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "rgb_baseline_best.pt")))
rgb_results = compute_iou_per_class(rgb_model, test_ds_rgb)
print_results("RGB baseline", rgb_results)

ms_model = build_multispectral_model().to(DEVICE)
ms_model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "multispectral_best.pt")))
ms_results = compute_iou_per_class(ms_model, test_ds_ms)
print_results("Multispectral", ms_results)

print(f"\nDelta (multispectral - RGB) mean IoU: {ms_results['mean_iou'] - rgb_results['mean_iou']:.4f}")
```

Note: Cells 2-4 assume `TileDataset`, `PATCH_DIR`, `CHECKPOINT_DIR`, `build_rgb_baseline`, `build_multispectral_model` are already defined — true if run right after Stage 4's cells in the same notebook. If running fresh, re-paste Stage 4 Cells 3-4 first.

## Result — final (unweighted loss, 30 epochs)

| Metric | RGB baseline | Multispectral | Delta |
|---|---|---|---|
| Pixel accuracy | 0.7682 | 0.7678 | ~0 |
| Mean IoU (present classes) | 0.3235 | 0.3313 | +0.0078 |
| Tree cover IoU | 0.6464 | 0.6442 | ~0 |
| Cropland IoU | 0.5216 | 0.4761 | -0.0455 |
| Built-up IoU | 0.7705 | 0.7785 | +0.0080 |
| Water IoU | 0.5286 | 0.6185 | **+0.0899** |
| Grassland IoU | 0.1210 | 0.1334 | +0.0124 |

**Conclusion**: multispectral input improves LULC segmentation over RGB
alone, most clearly on Water (+9% IoU) — consistent with the hypothesis
that SWIR/NIR bands carry water-surface information invisible to RGB.
Cropland and Tree cover show no clear benefit, or slight regression.

**Caveats**:
- No fixed random seed — a second unweighted run showed mean IoU delta
  ranging 0.008-0.021 between runs, same direction (multispectral ahead)
  but different magnitude. Report the direction as the finding, not the
  exact delta value, unless multiple seeds are averaged.
- Class-weighted loss was tried to address Wetland's near-zero IoU
  (0.68% of pixels) but reduced major-class performance without fixing
  Wetland (capped at ~3% IoU even fully weighted) — concluded this is a
  data scarcity problem (too few Wetland pixels/tiles), not a loss-function
  problem. Noted as a limitation, not solved in this iteration.
- Rare/absent classes (Shrubland, Snow/ice, Mangroves, Moss/lichen) have
  near-zero or zero test pixels in this AOI and are excluded from
  meaningful interpretation — mean IoU is calculated only over classes
  with nonzero union, per `compute_iou_per_class`.