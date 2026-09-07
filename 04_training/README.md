# Stage 4: Training

## What this does
Trains both models from Stage 3 on the Stage 2 patches, on Colab (free tier).

## Decisions
| Decision | Choice | Why |
|---|---|---|
| Checkpoint location | `/content/drive/MyDrive/patches/checkpoints/`, 2 overwritten files per model | Inside `patches/` for one-folder download; overwriting avoids Drive bloat |
| Patch source | Drive `MyDrive/patches` (already uploaded) | No manual upload needed each session |
| Batch size | 8 | Reduced from 16 after hitting CUDNN_STATUS_INTERNAL_ERROR on free-tier T4 — likely GPU memory pressure |

## Bug encountered: label remapping never called
Training crashed with `CUDA error: device-side assert triggered`. Root cause:
raw WorldCover pixel values (10, 20, 30 ... 100) were fed straight into
`CrossEntropyLoss`, which expects class indices `0` to `num_classes-1` (0-10
here). A label value of `100` is a massive out-of-bounds index, and that's
exactly what a device-side assert catches. `remap_labels()` existed in
`model.py` but was never actually called in the dataset — fixed by calling
it inside `TileDataset.__getitem__` (Cell 4 below).

**Takeaway**: this class of error (device-side assert / CUDA assert) on a
segmentation or classification task is almost always an out-of-range label
index, not a hardware/environment problem — worth checking label ranges
before assuming anything about the GPU setup.
| Epochs | 30 (starting point) | Adjust based on val loss curve |

## Colab — run as 7 separate cells

**Cell 1 — mount Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Cell 2 — install**
```python
!pip install segmentation-models-pytorch -q
```

**Cell 3 — model definitions**
```python
import segmentation_models_pytorch as smp

WORLDCOVER_CLASSES = {10:0,20:1,30:2,40:3,50:4,60:5,70:6,80:7,90:8,95:9,100:10}
NUM_CLASSES = len(WORLDCOVER_CLASSES)

def remap_labels(label_array):
    remapped = label_array.copy()
    for raw_val, class_idx in WORLDCOVER_CLASSES.items():
        remapped[label_array == raw_val] = class_idx
    return remapped

def build_rgb_baseline(num_classes=NUM_CLASSES, encoder="resnet34"):
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet", in_channels=3, classes=num_classes)

def build_multispectral_model(num_classes=NUM_CLASSES, encoder="resnet34", in_channels=13):
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet", in_channels=in_channels, classes=num_classes)
```

**Cell 4 — config + dataset**
```python
import os, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

PATCH_DIR = "/content/drive/MyDrive/patches"
CHECKPOINT_DIR = "/content/drive/MyDrive/patches/checkpoints"
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = False  # avoids intermittent CUDNN_STATUS_INTERNAL_ERROR on Colab T4
print("Device:", DEVICE)

class TileDataset(Dataset):
    def __init__(self, patch_dir, split, channels="all"):
        self.image_dir = os.path.join(patch_dir, split, "images")
        self.label_dir = os.path.join(patch_dir, split, "labels")
        self.files = sorted(os.listdir(self.image_dir))
        self.channels = channels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        image = np.load(os.path.join(self.image_dir, fname))
        label = np.load(os.path.join(self.label_dir, fname))
        if self.channels == "rgb":
            image = image[[2, 1, 0], :, :]
        label = remap_labels(label)  # raw WorldCover codes (10-100) -> 0-10 class indices
        return torch.from_numpy(image).float(), torch.from_numpy(label).long()
```

**Cell 5 — training function**
```python
def train_model(model, model_name, in_channels_mode):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    train_ds = TileDataset(PATCH_DIR, "train", channels=in_channels_mode)
    val_ds = TileDataset(PATCH_DIR, "val", channels=in_channels_mode)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
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

        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"{model_name}_latest.pt"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pt"))
            print(f"  New best saved (val_loss: {val_loss:.4f})")
    return model
```

**Cell 6 — train RGB baseline**
```python
rgb_model = build_rgb_baseline()
rgb_model = train_model(rgb_model, "rgb_baseline", in_channels_mode="rgb")
```

**Cell 7 — train multispectral**
```python
ms_model = build_multispectral_model()
ms_model = train_model(ms_model, "multispectral", in_channels_mode="all")
```

## After training
Download the whole `patches/checkpoints/` folder + the notebook (`.ipynb`) — both live in one place (`MyDrive/patches/`) for convenience.

**Next**: `05_evaluation/` — compare RGB vs multispectral on the test split.