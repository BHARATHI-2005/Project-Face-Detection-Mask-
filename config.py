"""Central configuration — edit paths / hyperparameters here or via CLI flags."""

from pathlib import Path

# --- Paths (all relative to project root) ---
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "train"
VAL_DIR = ROOT / "data" / "val"          # optional manual split; empty by default
MODEL_PATH = ROOT / "models" / "mask_detector_mobilenet.keras"
LABELS_PATH = ROOT / "models" / "class_indices.json"
OUTPUTS_DIR = ROOT / "outputs"
CAPTURES_DIR = OUTPUTS_DIR / "no_mask_captures"
LOG_PATH = OUTPUTS_DIR / "no_mask_log.txt"
REPORT_PATH = OUTPUTS_DIR / "session_report.txt"
VIDEO_PATH = OUTPUTS_DIR / "session_video.avi"
CURVES_PATH = OUTPUTS_DIR / "training_curves.png"

# --- Training hyperparameters ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SPLIT = 0.2
SEED = 42

# --- Detection ---
CAMERA_SOURCE = 0
FRAME_SIZE = (640, 480)
CAPTURE_COOLDOWN_SEC = 2.0   # min seconds between saved no-mask captures
HAAR_CASCADE = "haarcascade_frontalface_default.xml"
