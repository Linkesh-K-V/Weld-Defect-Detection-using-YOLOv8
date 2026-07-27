"""
yolo_inference.py

Runs the trained YOLOv8 classification model on a single image, a
folder of images, or a live webcam feed. Automatically handles both
binary (Normal/Defective) and multi-class models based on how many
classes the loaded weights actually have.

Usage:
    python yolo_inference.py --source path/to/image.jpg
    python yolo_inference.py --source path/to/folder
    python yolo_inference.py --camera
    python yolo_inference.py --source path/to/image.jpg --weights models/yolov8_best.pt --no_display
"""

import argparse
import sys
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_DIR

IMG_EXTS = (".jpg", ".jpeg", ".png")
DEFAULT_WEIGHTS = MODELS_DIR / "yolov8_best.pt"


def load_model(weights_path):
    if not Path(weights_path).exists():
        raise FileNotFoundError(
            f"No weights found at {weights_path}. Train the model first with train_yolo.py, "
            f"or pass --weights pointing at your best.pt."
        )
    model = YOLO(str(weights_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded on {device.upper()} | Classes: {model.names}")
    return model, device


def classify_image(model, device, image_path, display=True, top_k=5):
    """Run classification on one image. Prints prediction(s) and, unless
    --no_display was passed, shows the image with the top-1 label overlaid."""
    results = model.predict(str(image_path), device=device, verbose=False)
    probs = results[0].probs
    if probs is None:
        print(f"[WARN] No prediction generated for {image_path}")
        return

    num_classes = len(model.names)
    image_path = Path(image_path)

    if num_classes == 2:
        class_id = int(torch.argmax(probs.data))
        confidence = float(probs.data[class_id])
        predicted_class = model.names[class_id]
        print(f"{image_path.name:40s} -> {predicted_class:10s} | Confidence: {confidence:.2%}")
    else:
        k = min(top_k, num_classes)
        topk = torch.topk(probs.data, k)
        print(f"\n{image_path.name}")
        for rank in range(k):
            cid = int(topk.indices[rank])
            conf = float(topk.values[rank])
            print(f"  {rank+1}. {model.names[cid]:35s} {conf:.2%}")
        class_id = int(topk.indices[0])
        predicted_class = model.names[class_id]
        confidence = float(topk.values[0])

    if not display:
        return

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[WARN] Could not load image for display: {image_path}")
        return

    is_normal = predicted_class.strip().lower() == "normal"
    color = (0, 200, 0) if is_normal else (0, 0, 255)
    label = f"{predicted_class} ({confidence:.1%})"
    cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    img = cv2.resize(img, (800, 600))
    cv2.imshow("Weld Defect Classification", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_live_camera(model, device, camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {camera_index}")
        return

    print("Live weld inspection started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame, stopping.")
            break

        results = model.predict(frame, device=device, verbose=False)
        probs = results[0].probs

        if probs is not None:
            class_id = int(torch.argmax(probs.data))
            confidence = float(probs.data[class_id])
            predicted_class = model.names[class_id]
            is_normal = predicted_class.strip().lower() == "normal"
            color = (0, 200, 0) if is_normal else (0, 0, 255)
            label = f"{predicted_class} ({confidence:.1%})"
            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Weld Defect Classification - Live", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to trained YOLOv8-cls weights")
    ap.add_argument("--source", help="Image file or folder of images")
    ap.add_argument("--camera", action="store_true", help="Run live webcam inference instead of --source")
    ap.add_argument("--camera_index", type=int, default=0)
    ap.add_argument("--no_display", action="store_true", help="Skip the OpenCV popup window, print results only")
    args = ap.parse_args()

    model, device = load_model(args.weights)

    if args.camera:
        run_live_camera(model, device, args.camera_index)
        return

    if not args.source:
        print("Provide --source <image_or_folder> or use --camera for live mode.")
        return

    source = Path(args.source)
    if source.is_dir():
        images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMG_EXTS)
        if not images:
            print(f"No images found in {source}")
            return
        for img_path in images:
            classify_image(model, device, img_path, display=not args.no_display)
    elif source.is_file():
        classify_image(model, device, source, display=not args.no_display)
    else:
        print(f"Path not found: {source}")


if __name__ == "__main__":
    main()
