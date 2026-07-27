"""
prepare_dataset.py

Converts the RAW Weld-Defect-Dataset (Linkesh-K-V/Weld-Defect-Dataset,
Pascal VOC format) into classification-ready folders.

Confirmed real repo layout (verified directly against the repo):

    Weld-Defect-Dataset/
        Annotations/     *.xml   (VOC XML, <object><name> = defect label)
        JPEGImages/       *.jpg  (all 2422 images)
        images/
            train/        *.jpg  (1695 images — the repo's own train split)
            val/          *.jpg  (242 images  — the repo's own val split)
            test/         *.jpg  (485 images  — the repo's own test split)
        labels/           (YOLO-format .txt, same split as images/ — not used here)
        txt/              (YOLO-format .txt, flat — not used here)

This script uses the REPO'S OWN train/val/test split (from the images/
folder) rather than re-shuffling, since that split was deliberately made
by whoever built the dataset.

Class labels come from Annotations/*.xml <object><name> tags. Multiple
distinct defects on one image are combined into one folder name, e.g.
"Surface_pores;wrong_size". The "Weld" box (just marks the weld seam
region, not a defect — present in ~21% of files inconsistently) is
excluded by default since it fragments otherwise-identical defect
combos into separate classes; use --keep_labels to include it.

Output:
    dataset/
        train/<class_or_combo>/*.jpg
        val/<class_or_combo>/*.jpg
        test/<class_or_combo>/*.jpg

USAGE
-----
python prepare_dataset.py --raw_dir "path/to/Weld-Defect-Dataset"
python prepare_dataset.py --raw_dir . --keep_labels Weld   # to keep the Weld box as a class
"""

import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATASET_DIR  # noqa: E402

IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".JPEG")
DEFAULT_EXCLUDE = {"Weld"}  # region marker, not a defect class


def find_dir(raw_dir: Path, *candidates) -> Optional[Path]:
    """Return the first existing candidate path (checked at raw_dir root,
    then under VOCdevkit/, then under VOCdevkit/VOC2007/ — for older clones)."""
    bases = [raw_dir, raw_dir / "VOCdevkit", raw_dir / "VOCdevkit" / "VOC2007"]
    for base in bases:
        for name in candidates:
            p = base / name
            if p.exists():
                return p
    return None


def parse_labels_from_xml(xml_path: Path, exclude: set) -> str:
    """Return a combined class-folder name, e.g. 'good' or
    'Not_fully_soldered;Surface_pores;wrong_size'."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    labels = set()
    for obj in root.findall("object"):
        name_tag = obj.find("name")
        if name_tag is not None and name_tag.text:
            name = name_tag.text.strip()
            if name not in exclude:
                labels.add(name)
    if not labels:
        return "good"
    return ";".join(sorted(labels))


def build_split_lookup(raw_dir: Path):
    """Use the repo's own images/train|val|test split if present.
    Returns {stem: 'train'|'val'|'test'} or None if that split doesn't exist."""
    images_root = find_dir(raw_dir, "images")
    if images_root is None:
        return None

    lookup = {}
    for split in ("train", "val", "test"):
        split_dir = images_root / split
        if not split_dir.exists():
            continue
        for p in split_dir.iterdir():
            if p.suffix.lower() in IMG_EXTS:
                lookup[p.stem] = split

    return lookup if lookup else None


def find_image(img_dirs, stem: str):
    for d in img_dirs:
        if d is None:
            continue
        for ext in IMG_EXTS:
            candidate = d / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="Path to the raw repo root")
    ap.add_argument("--out_dir", default=str(DATASET_DIR),
                     help=f"Where to write train/val/test (default: {DATASET_DIR})")
    ap.add_argument("--val_split", type=float, default=0.2,
                     help="Only used as a fallback if the repo has no images/train|val split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep_labels", nargs="*", default=[],
                     help="Object names to KEEP that are excluded by default (e.g. Weld)")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    exclude = DEFAULT_EXCLUDE - set(args.keep_labels)

    ann_dir = find_dir(raw_dir, "Annotations")
    img_dir = find_dir(raw_dir, "JPEGImages")

    if ann_dir is None:
        raise FileNotFoundError(f"Could not find Annotations/ under {raw_dir}.")
    if img_dir is None:
        raise FileNotFoundError(f"Could not find JPEGImages/ under {raw_dir}.")

    ann_files = sorted(ann_dir.glob("*.xml"))
    if not ann_files:
        raise FileNotFoundError(f"No .xml files found in {ann_dir}")
    print(f"Found {len(ann_files)} annotation files.")
    if exclude:
        print(f"Excluding these object names from class labels: {sorted(exclude)}")

    split_lookup = build_split_lookup(raw_dir)
    out_dir = Path(args.out_dir)

    if split_lookup:
        print(f"Using the repo's own images/train|val|test split "
              f"({len(split_lookup)} images covered).")
    else:
        print("No images/train|val split found in the raw repo — doing a random split instead.")
        random.seed(args.seed)
        all_ids = [x.stem for x in ann_files]
        random.shuffle(all_ids)
        n_val = int(len(all_ids) * args.val_split)
        val_ids = set(all_ids[:n_val])
        split_lookup = {stem: ("val" if stem in val_ids else "train") for stem in all_ids}

    class_counts = {}
    skipped = 0

    for ann_path in tqdm(ann_files, desc="Preparing dataset"):
        stem = ann_path.stem
        label = parse_labels_from_xml(ann_path, exclude)

        img_path = find_image([img_dir], stem)
        if img_path is None:
            skipped += 1
            continue

        split = split_lookup.get(stem, "train")  # default to train if not in any split list
        dest_dir = out_dir / split / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dest_dir / img_path.name)
        class_counts[label] = class_counts.get(label, 0) + 1

    print("\nDone. Class distribution (train + val + test combined):")
    for label, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:45s} : {count}")
    if skipped:
        print(f"\n[WARN] {skipped} annotation files had no matching image and were skipped.")

    print(f"\nOutput written to: {out_dir}")
    print("This folder is used directly by train_cnn_models.py (ResNet-18, MobileNetV3) "
          "and train_yolo.py (YOLOv8 classification).")


if __name__ == "__main__":
    main()
