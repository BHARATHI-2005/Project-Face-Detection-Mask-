<div align="center">

# 😷 Face Mask Detection

### Real-time face-mask classification with MobileNetV2 and OpenCV

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10 or later">
  <img src="https://img.shields.io/badge/TensorFlow-2.20%2B-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow 2.20 or later">
  <img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV 4">
  <img src="https://img.shields.io/badge/Model-MobileNetV2-00A98F" alt="MobileNetV2">
  <img src="https://img.shields.io/badge/Validation%20accuracy-99.27%25-2EA44F" alt="Reported validation accuracy 99.27 percent">
</p>

<p>
  A practical computer-vision project that detects faces in a webcam or video
  and classifies each face as <strong>Mask</strong> or <strong>No Mask</strong>.
  Train your own model, inspect predictions in real time, and save annotated
  sessions and no-mask event captures.
</p>

</div>

---

## ✨ Highlights

- **Transfer learning:** MobileNetV2 feature extraction with a lightweight binary-classification head.
- **Live detection:** OpenCV Haar cascade face localization paired with per-face mask predictions.
- **Train on your own data:** Directory-based classes, augmentation, and a configurable validation split.
- **Useful session artifacts:** Annotated video, throttled no-mask captures, event log, and session report.
- **Reproducible labels:** The class-index mapping is saved during training and used by the detector.
- **Flexible operation:** Webcam or video input, optional alert sound, and headless mode for automated runs.

## 🔎 How it works

```text
Webcam / video
      ↓
OpenCV face detection (Haar cascade)
      ↓
Face crop → resize to 224 × 224 → MobileNetV2 classifier
      ↓
Mask / No Mask label + annotated output
```

## 🚀 Get started

### 1. Set up the environment

Python 3.10 or later is recommended. TensorFlow 2.20 or later is required for Python 3.13.

```powershell
git clone https://github.com/BHARATHI-2005/Project-Face-Detection-Mask-.git
cd Project-Face-Detection-Mask-
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Prepare a dataset

Arrange training images in class-named folders:

```text
data/
├── train/
│   ├── with_mask/
│   └── without_mask/
└── val/                 # optional separate validation set
    ├── with_mask/
    └── without_mask/
```

Without a separate `data/val` set, training automatically reserves 20% of
`data/train` for validation. See [`data/README.md`](data/README.md) for more
details and the dataset source used during development.

### 3. Train the classifier

```powershell
python -m src.train --data-dir data/train --epochs 10 --batch-size 32
```

To use a separate validation set or customize the output model:

```powershell
python -m src.train `
  --data-dir data/train `
  --val-dir data/val `
  --epochs 15 `
  --output models/mask_detector_mobilenet.keras
```

Training writes the model and class mapping under `models/`, plus accuracy and
loss curves to `outputs/training_curves.png`.

### 4. Run detection

Start the webcam detector:

```powershell
python -m src.detect
```

Or process a video file without opening a preview window:

```powershell
python -m src.detect `
  --source path\to\video.mp4 `
  --no-display `
  --no-alert `
  --max-frames 300
```

Press **q** or **Esc** to stop an interactive session. Use
`python -m src.detect --help` to see all options.

## 📁 Project layout

```text
.
├── config.py                 # Paths and default settings
├── data/
│   └── README.md             # Dataset setup and source
├── models/                   # Trained model and class mapping
├── notebooks/                # Step-by-step notebook
├── outputs/                  # Generated reports, captures, and videos
├── src/
│   ├── detect.py             # Webcam and video inference
│   ├── train.py              # Model training and evaluation
│   └── utils.py              # Preprocessing, labels, and alerts
└── requirements.txt
```

## 📦 Generated files

| Location | Description |
|---|---|
| `models/mask_detector_mobilenet.keras` | Trained Keras model |
| `models/class_indices.json` | Class-to-index mapping used during training |
| `outputs/training_curves.png` | Training and validation accuracy/loss |
| `outputs/session_video.avi` | Annotated output video |
| `outputs/no_mask_captures/` | Saved frames for no-mask detections |
| `outputs/no_mask_log.txt` | Timestamped no-mask events |
| `outputs/session_report.txt` | Detection counts and session summary |

The pretrained model, its class mapping, and the training-curves plot are
included. Dataset images and runtime detection artifacts (recordings,
captures, logs, and reports) are not checked in.

## 📊 Results

The development run on the 1,376-image dataset described in
[`data/README.md`](data/README.md) reached **99.27% training accuracy** and
**99.27% validation accuracy** after 10 epochs. Results depend on the dataset,
split, and training environment; use the training curves and validation
metrics from your own run when evaluating a model.

## 📓 Notebook

For a guided walkthrough, open
[`notebooks/face_mask_detection.ipynb`](notebooks/face_mask_detection.ipynb).
The command-line detector is recommended for webcam sessions.

## ⚠️ Limitations

This is an educational computer-vision project, not a medical or safety
certification system. Detection quality depends on lighting, camera quality,
face visibility, and the data used to train the classifier.
