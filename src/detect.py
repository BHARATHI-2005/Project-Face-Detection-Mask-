"""Real-time face-mask detection from webcam or video file.

Usage:
    python -m src.detect                      # webcam
    python -m src.detect --source test.mp4    # video file
    python -m src.detect --no-alert           # silent mode
    python -m src.detect --source test.mp4 --no-display --max-frames 100  # headless test

Press 'q' or Esc to stop. Outputs: annotated video, no-mask captures,
log file and session report under outputs/.
"""

import argparse
import datetime
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg

from src.utils import (
    CaptureThrottle,
    beep_alert,
    is_no_mask,
    load_class_indices,
    preprocess_face,
    prob_to_label,
)


def parse_args():
    p = argparse.ArgumentParser(description="Real-time mask detection")
    p.add_argument("--model", default=str(cfg.MODEL_PATH))
    p.add_argument("--labels", default=str(cfg.LABELS_PATH))
    p.add_argument("--source", default=str(cfg.CAMERA_SOURCE),
                   help="Camera index (e.g. 0) or path to video file")
    p.add_argument("--video-out", default=str(cfg.VIDEO_PATH))
    p.add_argument("--captures-dir", default=str(cfg.CAPTURES_DIR))
    p.add_argument("--log", default=str(cfg.LOG_PATH))
    p.add_argument("--report", default=str(cfg.REPORT_PATH))
    p.add_argument("--cooldown", type=float, default=cfg.CAPTURE_COOLDOWN_SEC,
                   help="Min seconds between saved no-mask captures")
    p.add_argument("--no-alert", action="store_true", help="Disable beep alert")
    p.add_argument("--no-display", action="store_true",
                   help="Don't open a preview window (headless / testing)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Stop after N frames (0 = run until source ends / 'q'). Useful for tests.")
    p.add_argument("--width", type=int, default=cfg.FRAME_SIZE[0])
    p.add_argument("--height", type=int, default=cfg.FRAME_SIZE[1])
    return p.parse_args()


def resolve_source(value: str):
    """'0' -> int 0 (webcam), otherwise treat as file path."""
    try:
        return int(value)
    except ValueError:
        return value


def main():
    args = parse_args()  # parse first so --help works without TensorFlow
    model_path = Path(args.model)
    if not model_path.is_file():
        raise SystemExit(
            f"Model not found: {model_path}\nTrain first: python -m src.train")
    import tensorflow as tf  # local import: faster --help without TF

    class_indices = load_class_indices(args.labels)
    if class_indices:
        print(f"Class mapping: {class_indices}")
    else:
        print(f"Warning: {args.labels} not found, using default Mask/No-Mask rule.")

    captures_dir = Path(args.captures_dir)
    captures_dir.mkdir(parents=True, exist_ok=True)
    Path(args.video_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(model_path)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + cfg.HAAR_CASCADE)
    if face_cascade.empty():
        raise SystemExit("Could not load Haar cascade for face detection.")

    source = resolve_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video source: {args.source}")

    frame_w, frame_h = args.width, args.height
    writer = cv2.VideoWriter(
        args.video_out, cv2.VideoWriter_fourcc(*"XVID"),
        20.0, (frame_w, frame_h))
    if not writer.isOpened():  # XVID missing -> fall back to mp4v
        writer = cv2.VideoWriter(
            args.video_out, cv2.VideoWriter_fourcc(*"mp4v"),
            20.0, (frame_w, frame_h))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Cannot open video writer: {args.video_out}")

    mask_count = 0
    no_mask_count = 0
    timestamps = []
    throttle = CaptureThrottle(args.cooldown)
    start_time = datetime.datetime.now()
    print(f"Camera started (source={args.source}). Press 'q' or Esc to stop.")

    try:
        frames_done = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of stream / camera read failed.")
                break
            frame = cv2.resize(frame, (frame_w, frame_h))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                face_img = frame[y:y + h, x:x + w]
                if face_img.size == 0:
                    continue
                prob = float(model.predict(
                    preprocess_face(face_img), verbose=0)[0][0])
                label, color = prob_to_label(prob, class_indices)

                if is_no_mask(label):
                    no_mask_count += 1
                    if throttle.ready():  # avoid saving every single frame
                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        timestamps.append(ts)
                        cv2.imwrite(
                            str(captures_dir / f"no_mask_{ts}.jpg"), frame)
                        with open(args.log, "a", encoding="utf-8") as f:
                            f.write(f"No Mask Detected at {ts}\n")
                        beep_alert(enabled=not args.no_alert)
                else:
                    mask_count += 1

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {prob:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            total = mask_count + no_mask_count
            pct_mask = (mask_count / total * 100) if total else 0.0
            pct_no = (no_mask_count / total * 100) if total else 0.0
            cv2.putText(
                frame, f"Mask: {pct_mask:.1f}% No Mask: {pct_no:.1f}%",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if not args.no_display:
                cv2.imshow("Mask Detection", frame)
            writer.write(frame)

            if args.no_display:
                frames_done += 1
                if args.max_frames and frames_done >= args.max_frames:
                    print(f"Reached --max-frames={args.max_frames}, stopping.")
                    break
                continue  # no keyboard handling without a window

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # q or Esc
                break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        end_time = datetime.datetime.now()
        with open(args.report, "w", encoding="utf-8") as f:
            f.write("Face Mask Detection - Session Report\n")
            f.write("=" * 40 + "\n")
            f.write(f"Session started: {start_time}\n")
            f.write(f"Session ended:   {end_time}\n")
            f.write(f"Mask detections: {mask_count}\n")
            f.write(f"No-mask detections: {no_mask_count}\n")
            f.write("No-mask timestamps:\n")
            for ts in timestamps:
                f.write(f"  - {ts}\n")
        print(f"Session report: {args.report}")
        print(f"Session video:  {args.video_out}")
        print(f"Captures:       {captures_dir} ({len(timestamps)} saved)")


if __name__ == "__main__":
    main()
