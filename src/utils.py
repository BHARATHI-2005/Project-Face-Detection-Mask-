"""Shared helpers: preprocessing, label mapping, alert beep."""

import json
import time
from pathlib import Path

import cv2
import numpy as np


def preprocess_face(face_bgr: np.ndarray, img_size=(224, 224)) -> np.ndarray:
    """Convert an OpenCV BGR face crop to a normalised model-ready batch."""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, img_size)
    face_norm = face_resized.astype("float32") / 255.0
    return np.expand_dims(face_norm, axis=0)


def save_class_indices(class_indices: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(class_indices, f, indent=2)


def load_class_indices(path: Path) -> dict | None:
    path = Path(path)
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def prob_to_label(prob: float, class_indices: dict | None) -> tuple[str, tuple]:
    """Map sigmoid output -> (label, box colour).

    flow_from_directory with class_mode='binary' assigns prob=0 to the
    alphabetically-first class and prob=1 to the second. Using the saved
    class_indices avoids the original hard-coded `pred < 0.5 = Mask` bug.
    Falls back to Mask/No-Mask convention when mapping is unavailable.
    """
    GREEN, RED = (0, 255, 0), (0, 0, 255)
    if class_indices:
        inv = {v: k for k, v in class_indices.items()}
        pred_class = int(prob >= 0.5)
        raw = inv.get(pred_class, str(pred_class))
        key = raw.lower()
        if "without" in key or "no" in key or "nomask" in key.replace("_", "").replace("-", ""):
            return "No Mask", RED
        return "Mask", GREEN
    # Legacy fallback (matches original notebook behaviour).
    if prob < 0.5:
        return "Mask", GREEN
    return "No Mask", RED


def is_no_mask(label: str) -> bool:
    return label.lower() != "mask"


def beep_alert(enabled: bool = True) -> None:
    """Cross-platform no-dependency alert. Windows: Beep; else terminal bell."""
    if not enabled:
        return
    try:
        import winsound  # Windows only

        winsound.Beep(880, 400)
    except Exception:
        # Fallback on Linux/macOS: ASCII bell (may be silent in some terminals).
        print("\a", end="", flush=True)


class CaptureThrottle:
    """Allow at most one capture per `cooldown_sec` (prevents frame flooding)."""

    def __init__(self, cooldown_sec: float = 2.0):
        self.cooldown = cooldown_sec
        self._last = 0.0

    def ready(self) -> bool:
        now = time.monotonic()
        if now - self._last >= self.cooldown:
            self._last = now
            return True
        return False
