"""
train_yolo.py

Trains a YOLOv8 classification model on the prepared weld defect
dataset and copies the best weights into models/.

Usage:
    python train_yolo.py
    python train_yolo.py --epochs 20 --batch 16
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATASET_DIR, MODELS_DIR, YOLO_BASE_MODEL, YOLO_EPOCHS, YOLO_IMG_SIZE, YOLO_BATCH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=YOLO_EPOCHS)
    ap.add_argument("--imgsz", type=int, default=YOLO_IMG_SIZE)
    ap.add_argument("--batch", type=int, default=YOLO_BATCH)
    ap.add_argument("--base_model", default=YOLO_BASE_MODEL)
    args = ap.parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Device: {'GPU (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")

    model = YOLO(args.base_model)
    model.train(
        data=str(DATASET_DIR),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project="runs/classify",
        name="weld_yolo_cls",
    )

    best_weights = Path("runs/classify/weld_yolo_cls/weights/best.pt")
    if best_weights.exists():
        dest = MODELS_DIR / "yolov8_best.pt"
        shutil.copy2(best_weights, dest)
        print(f"\nBest weights copied to: {dest}")
    else:
        print(f"\n[WARN] Expected weights at {best_weights} but they weren't found.")


if __name__ == "__main__":
    main()
