"""
train.py — Stage 4 for Project 1 (LULC Classification)

Run in Colab. Trains both the RGB baseline and multispectral model.

Checkpoint strategy: saves 'latest.pt' every epoch and 'best.pt' whenever
val loss improves, both to Drive, overwriting each time. This survives a
free-tier Colab disconnect without accumulating storage — only 2 files per
model, not one per epoch.

At the end, download best.pt + this notebook to your laptop for a
permanent local copy; Drive here is just a training-time safety net, not
long-term storage.

Requirements (Colab): pip install segmentation-models-pytorch torch
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# from google.colab import drive
# drive.mount('/content/drive')

CHECKPOINT_DIR = "/content/drive/MyDrive/patches/checkpoints"  # inside patches folder for easy download
PATCH_DIR = "/content/drive/MyDrive/patches"  # patches already uploaded to Drive directly
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = False  # avoids intermittent CUDNN_STATUS_INTERNAL_ERROR on Colab T4


class TileDataset(Dataset):
    def __init__(self, patch_dir, split, channels="all"):
        self.image_dir = os.path.join(patch_dir, split, "images")
        self.label_dir = os.path.join(patch_dir, split, "labels")
        self.files = sorted(os.listdir(self.image_dir))
        self.channels = channels  # "all" (13) or "rgb" (indices for B4,B3,B2)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        image = np.load(os.path.join(self.image_dir, fname))
        label = np.load(os.path.join(self.label_dir, fname))

        if self.channels == "rgb":
            # band order from Stage 1: B2,B3,B4,... -> RGB = indices 2,1,0
            image = image[[2, 1, 0], :, :]

        label = remap_labels(label)  # raw WorldCover codes (10-100) -> 0-10 class indices

        return torch.from_numpy(image).float(), torch.from_numpy(label).long()


def compute_class_weights(dataset, num_classes):
    """Inverse-frequency weights from the train split, capped to avoid
    instability from near-absent classes, zero for classes with no pixels
    at all (they can't be learned regardless of weight)."""
    counts = np.zeros(num_classes)
    for i in range(len(dataset)):
        _, label = dataset[i]
        for c in label.unique().tolist():
            counts[c] += (label == c).sum().item()

    total = counts.sum()
    weights = np.zeros(num_classes)
    nonzero = counts > 0
    weights[nonzero] = total / (num_classes * counts[nonzero])
    weights = np.clip(weights, 0, 10)  # cap extreme weights from rare classes
    return torch.tensor(weights, dtype=torch.float32)


def train_model(model, model_name, in_channels_mode, class_weights=None):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    train_ds = TileDataset(PATCH_DIR, "train", channels=in_channels_mode)
    val_ds = TileDataset(PATCH_DIR, "val", channels=in_channels_mode)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    else:
        criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"[{model_name}] Epoch {epoch+1}/{EPOCHS} — train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

        latest_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_latest.pt")
        torch.save(model.state_dict(), latest_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"  New best model saved (val_loss: {val_loss:.4f})")

    return model


if __name__ == "__main__":
    import sys
    sys.path.append("../../03_model_architecture/scripts")
    from model import build_rgb_baseline, build_multispectral_model, remap_labels, NUM_CLASSES

    print(f"Device: {DEVICE}")

    print("Computing class weights from train split...")
    weight_ds = TileDataset(PATCH_DIR, "train", channels="all")
    class_weights = compute_class_weights(weight_ds, NUM_CLASSES)
    print(f"Class weights: {class_weights.tolist()}")

    print("\n--- Training RGB baseline ---")
    rgb_model = build_rgb_baseline()
    train_model(rgb_model, "rgb_baseline", in_channels_mode="rgb", class_weights=class_weights)

    print("\n--- Training multispectral model ---")
    ms_model = build_multispectral_model()
    train_model(ms_model, "multispectral", in_channels_mode="all", class_weights=class_weights)