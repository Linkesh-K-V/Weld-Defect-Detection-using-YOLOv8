"""
dataset_utils.py

A classification dataset loader that walks a folder-per-class directory
using a FIXED, EXTERNALLY-SUPPLIED class order — unlike
torchvision.datasets.ImageFolder, which derives class order
independently from whatever subfolders it finds in that specific
directory.

Why this matters: with an imbalanced dataset (some classes have only a
handful of images), a random split can easily leave a rare class with
zero images in val/test. If you build train and val as two separate
ImageFolder calls, each one assigns class indices alphabetically from
ITS OWN folder listing — so if val is missing a class that train has,
every index after the missing one shifts by one between the two
datasets. The model's predicted index 4 might mean "wrong_size" in
train but "good" in val, and accuracy collapses to near-zero even
though the model is actually working. This class avoids that entirely
by using one shared class list everywhere.
"""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


class FixedClassImageFolder(Dataset):
    def __init__(self, root_dir, class_names, transform=None):
        self.root_dir = Path(root_dir)
        self.class_names = list(class_names)
        self.transform = transform
        self.samples = []  # list of (path, class_idx)
        self.missing_classes = []

        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                self.missing_classes.append(class_name)
                continue
            for path in class_dir.iterdir():
                if path.suffix.lower() in IMG_EXTS:
                    self.samples.append((path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, class_idx = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, class_idx
