"""
evaluate_models.py

Evaluates one or all trained models (ResNet-18, MobileNetV3, YOLOv8):
accuracy, precision, recall, F1-score, confusion matrix (saved to
results/), and classification report (printed + saved to results/).

By default evaluates on dataset/val. Pass --split test to check against
the held-out test set instead (the more rigorous check, since it's
never touched during training).

Usage:
    python evaluate_models.py --model resnet18
    python evaluate_models.py --model all               # compares all three, saves a bar chart
    python evaluate_models.py --model all --split test   # check against the held-out test set
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from torchvision import models

# matplotlib/seaborn are only needed for the PNG plots. On some locked-down
# Windows machines (Application Control / Smart App Control policies) their
# compiled extensions get blocked at import time. Don't let that take down
# the whole script — the accuracy/precision/recall/F1 numbers and the text
# classification report are the important output and don't need plotting.
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError as e:
    PLOTTING_AVAILABLE = False
    print(f"[WARN] Plotting disabled — matplotlib/seaborn failed to import ({e}). "
          f"Numeric results and text reports still work; see the note at the end of this run "
          f"for how to fix plotting.")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import VAL_DIR, TEST_DIR, MODELS_DIR, RESULTS_DIR
from train_cnn_models import TRANSFORM, DEVICE

WEIGHTS_PATHS = {
    "resnet18": MODELS_DIR / "resnet18_best.pt",
    "mobilenetv3": MODELS_DIR / "mobilenetv3_best.pt",
    "yolov8": MODELS_DIR / "yolov8_best.pt",
}
CLASSES_PATH = MODELS_DIR / "classes.txt"

IMG_EXTS = (".jpg", ".jpeg", ".png")


def load_training_class_order():
    """The exact class list+order train_cnn_models.py used — evaluation
    MUST use this same order rather than re-deriving classes from
    whichever split folder it happens to scan, or prediction indices
    can silently line up with the wrong class names."""
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"{CLASSES_PATH} not found — run train_cnn_models.py first, it writes this file."
        )
    return CLASSES_PATH.read_text().strip().split("\n")


def load_torchvision_model(model_type, num_classes, weights_path):
    if model_type == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == "mobilenetv3":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(model_type)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    return model.to(DEVICE).eval()


def evaluate_torchvision(model_type, split_dir, class_names):
    """Walk split_dir class-by-class using the FIXED training class order
    (not an independently re-derived one), so indices always line up."""
    from PIL import Image

    model = load_torchvision_model(model_type, len(class_names), WEIGHTS_PATHS[model_type])

    y_true, y_pred = [], []
    missing_classes = []
    for class_idx, class_name in enumerate(class_names):
        class_dir = split_dir / class_name
        if not class_dir.exists():
            missing_classes.append(class_name)
            continue
        images = [p for p in class_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        for img_path in images:
            img = Image.open(img_path).convert("RGB")
            tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred = model(tensor).argmax(1).item()
            y_pred.append(pred)
            y_true.append(class_idx)

    if missing_classes:
        print(f"[NOTE] {len(missing_classes)} training classes have no examples in this split "
              f"(likely just very rare combos): {missing_classes}")

    return y_true, y_pred


def evaluate_yolov8(split_dir, class_names):
    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS_PATHS["yolov8"]))
    # Ultralytics assigns its own class order at train time (model.names) —
    # use THAT order here (it's already self-consistent), not the CNN
    # classes.txt, since the two models don't necessarily share ordering.
    yolo_class_names = list(model.names.values())

    y_true, y_pred = [], []
    for class_idx, class_name in model.names.items():
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue
        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            result = model.predict(str(img_path), verbose=False)[0]
            y_pred.append(int(result.probs.top1))
            y_true.append(class_idx)
    return y_true, y_pred, yolo_class_names


def report_and_confusion_matrix(y_true, y_pred, class_names, model_name, split_name):
    print(f"\n=== {model_name} — Classification Report ({split_name}) ===")
    labels_present = sorted(set(y_true) | set(y_pred))
    names_present = [class_names[i] for i in labels_present]
    report_text = classification_report(
        y_true, y_pred, labels=labels_present, target_names=names_present, zero_division=0
    )
    print(report_text)

    report_path = RESULTS_DIR / f"{model_name}_{split_name}_classification_report.txt"
    report_path.write_text(report_text)

    cm = confusion_matrix(y_true, y_pred, labels=labels_present)

    if PLOTTING_AVAILABLE:
        plt.figure(figsize=(max(6, len(names_present) * 0.6), max(5, len(names_present) * 0.5)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=names_present, yticklabels=names_present)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix — {model_name} ({split_name})")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out_path = RESULTS_DIR / f"{model_name}_{split_name}_confusion_matrix.png"
        plt.savefig(out_path)
        plt.close()
        print(f"Saved: {out_path}\nSaved: {report_path}")
    else:
        # Fall back to a plain-text confusion matrix so you still get the
        # per-class breakdown even without plotting working.
        cm_df = pd.DataFrame(cm, index=names_present, columns=names_present)
        cm_path = RESULTS_DIR / f"{model_name}_{split_name}_confusion_matrix.csv"
        cm_df.to_csv(cm_path)
        print(f"[No plot — matplotlib unavailable] Confusion matrix saved as text: {cm_path}")
        print(f"Saved: {report_path}")

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def run_one(model_type, split_dir, split_name):
    if not WEIGHTS_PATHS[model_type].exists():
        print(f"[SKIP] No weights found for {model_type} at {WEIGHTS_PATHS[model_type]} "
              f"— train it first.")
        return None

    if model_type == "yolov8":
        y_true, y_pred, class_names = evaluate_yolov8(split_dir, None)
    else:
        class_names = load_training_class_order()
        y_true, y_pred = evaluate_torchvision(model_type, split_dir, class_names)

    if not y_true:
        print(f"[SKIP] No images found for {model_type} under {split_dir}.")
        return None

    return report_and_confusion_matrix(y_true, y_pred, class_names, model_type, split_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["resnet18", "mobilenetv3", "yolov8", "all"], default="all")
    ap.add_argument("--split", choices=["val", "test"], default="val",
                     help="Which held-out split to evaluate against")
    args = ap.parse_args()

    split_dir = VAL_DIR if args.split == "val" else TEST_DIR

    targets = ["resnet18", "mobilenetv3", "yolov8"] if args.model == "all" else [args.model]
    summary = {}
    for model_type in targets:
        metrics = run_one(model_type, split_dir, args.split)
        if metrics is not None:
            summary[model_type] = metrics

    if len(summary) > 1:
        df = pd.DataFrame(summary).T
        print("\n=== Model Comparison ===")
        print(df)

        if PLOTTING_AVAILABLE:
            metrics_names = ["Accuracy", "Precision", "Recall", "F1"]
            x = np.arange(len(df.index))
            width = 0.2
            fig, ax = plt.subplots(figsize=(8, 6))
            for i, metric in enumerate(metrics_names):
                ax.bar(x + i * width, df[metric], width, label=metric)
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(df.index)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Score")
            ax.set_title(f"Algorithm Comparison ({args.split})")
            ax.legend()
            plt.tight_layout()
            out_path = RESULTS_DIR / f"model_comparison_{args.split}.png"
            plt.savefig(out_path)
            print(f"Saved: {out_path}")
        else:
            csv_path = RESULTS_DIR / f"model_comparison_{args.split}.csv"
            df.to_csv(csv_path)
            print(f"[No plot — matplotlib unavailable] Comparison table saved as: {csv_path}")

        best_model = df["Accuracy"].idxmax()
        print(f"\nBest model by accuracy: {best_model} ({df.loc[best_model, 'Accuracy']:.2%})")

    if not PLOTTING_AVAILABLE:
        print("\n[Plotting note] matplotlib/seaborn couldn't load — this is almost always a "
              "Windows Application Control / Smart App Control policy blocking the compiled "
              "extension DLL, not a bug in this script. See the chat for how to fix it "
              "(likely Smart App Control, especially on a freshly reset Windows install).")


if __name__ == "__main__":
    main()
