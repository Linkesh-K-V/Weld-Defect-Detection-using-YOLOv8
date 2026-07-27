"""
config.py — shared settings for the whole project.

All paths are relative to the repo root by default, so the project
runs the same way on any machine it's cloned onto. Override any of
these from the command line in the individual scripts where supported.
"""

from pathlib import Path

# ---- Directories (relative to repo root) ----
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ---- Training hyperparameters ----
IMG_SIZE = 128
BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 12
EARLY_STOP_PATIENCE = 3

# ---- YOLOv8 ----
YOLO_BASE_MODEL = "yolov8n-cls.pt"
YOLO_EPOCHS = 10
YOLO_IMG_SIZE = 128
YOLO_BATCH = 8
