"""
train_cnn_models.py

Trains ResNet-18 and MobileNetV3-Large on the prepared weld defect
dataset (dataset/train, dataset/val), with early stopping, and saves
the best weights for each model to models/.

Usage:
    python train_cnn_models.py
    python train_cnn_models.py --epochs 20 --batch_size 16
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TRAIN_DIR, VAL_DIR, MODELS_DIR, IMG_SIZE, BATCH_SIZE, LEARNING_RATE, EPOCHS, EARLY_STOP_PATIENCE
from dataset_utils import FixedClassImageFolder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_dataloaders(batch_size):
    # train_ds (via ImageFolder) is the ONE place class order gets decided.
    train_ds = datasets.ImageFolder(root=str(TRAIN_DIR), transform=TRANSFORM)
    class_names = train_ds.classes

    # val_ds MUST reuse that exact same class order — see dataset_utils.py
    # for why building it as an independent ImageFolder(VAL_DIR) is a bug.
    val_ds = FixedClassImageFolder(VAL_DIR, class_names=class_names, transform=TRANSFORM)
    if val_ds.missing_classes:
        print(f"[NOTE] {len(val_ds.missing_classes)} classes have no examples in val "
              f"(expected for very rare combos): {val_ds.missing_classes}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, class_names


def train_and_evaluate(model, model_name, train_loader, val_loader, epochs, lr, patience):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    epochs_no_improve = 0
    weights_path = MODELS_DIR / f"{model_name}_best.pt"

    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        val_acc = evaluate_accuracy(model, val_loader)
        elapsed = time.time() - start

        print(f"[{model_name}] Epoch {epoch}/{epochs} | Loss: {running_loss/total:.4f} "
              f"| Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), weights_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[{model_name}] Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs).")
                break

    print(f"[{model_name}] Best Validation Accuracy: {best_val_acc:.2f}% -> saved to {weights_path}")
    return best_val_acc


def evaluate_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LEARNING_RATE)
    ap.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    args = ap.parse_args()

    print(f"Device: {DEVICE} "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'})")

    train_loader, val_loader, class_names = build_dataloaders(args.batch_size)
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # Save the exact class order used for training so evaluate_models.py
    # can map prediction indices back to the same class names — deriving
    # class order independently from val/test folders would silently
    # misalign indices if a class is missing or ordered differently there.
    classes_path = MODELS_DIR / "classes.txt"
    classes_path.write_text("\n".join(class_names))
    print(f"Saved class order to {classes_path}")

    print("\n=== Training ResNet-18 ===")
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    train_and_evaluate(resnet, "resnet18", train_loader, val_loader, args.epochs, args.lr, args.patience)

    print("\n=== Training MobileNetV3-Large ===")
    mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    mobilenet.classifier[3] = nn.Linear(mobilenet.classifier[3].in_features, num_classes)
    train_and_evaluate(mobilenet, "mobilenetv3", train_loader, val_loader, args.epochs, args.lr, args.patience)

    print("\nDone. Run evaluate_models.py for confusion matrices and classification reports.")


if __name__ == "__main__":
    main()
