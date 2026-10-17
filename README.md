<div align="center">

# 😷 Face Mask Detection

### Real-time face-mask classification with MobileNetV2 and OpenCV

<p>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow and Keras">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV computer vision">
  <img src="https://img.shields.io/badge/Model-MobileNetV2-00A98F" alt="MobileNetV2">
  <img src="https://img.shields.io/badge/Interface-Jupyter%20Notebook-F37626?logo=jupyter&logoColor=white" alt="Jupyter Notebook">
</p>

<p>
  Train a binary face-mask classifier with transfer learning, then use your
  webcam to detect faces and classify them as <strong>Mask</strong> or
  <strong>No Mask</strong>. The notebook draws predictions on screen and saves
  no-mask captures and a session report.
</p>

</div>

---

## ✨ Features

- **MobileNetV2 transfer learning** with a compact binary-classification head.
- **Image augmentation** and a built-in 80/20 training-validation split.
- **Live webcam inference** using OpenCV's Haar cascade face detector.
- **Visual predictions** with colored bounding boxes and confidence scores.
- **Session outputs** including no-mask snapshots, an event log, annotated video, and detection report.
- **Alert sound support** when an audio file is configured.

## 🔄 Workflow

```text
Labeled face images
        ↓
Augmentation + 80/20 split
        ↓
MobileNetV2 transfer learning
        ↓
Webcam → face detection → Mask / No Mask prediction
        ↓
Annotated video + captures + session report
```

## 🚀 Quick start

### 1. Install dependencies

Use Python with Jupyter Notebook or JupyterLab installed.

```bash
python -m venv .venv
```

Activate the environment, then install the notebook dependencies:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

```bash
python -m pip install --upgrade pip
python -m pip install tensorflow opencv-python numpy matplotlib pygame jupyter
```

### 2. Prepare the dataset

Create a training directory with one subfolder per class:

```text
Train/
├── with_mask/
│   ├── image_001.jpg
│   └── ...
└── without_mask/
    ├── image_001.jpg
    └── ...
```

The notebook currently points to `C:/Users/hp5cd/Downloads/Train`. Before
running it, update the `flow_from_directory` paths in the notebook's code cell
to the location of your `Train` folder. The two class folders must be inside
that directory.

### 3. Run the notebook

Open [`project(facedetection).ipynb`](<./project(facedetection).ipynb>) in
Jupyter and run the training and detection cell. The notebook trains for 10
epochs, saves `mask_detector_mobilenet.keras`, then starts the webcam detector.
Press **q** or **Esc** in the camera window to end the session.

The webcam must be available to OpenCV. To use the sound alert, set `alert_path`
in the notebook to an audio file on your computer; the detector continues
without playing an alert if that file is missing.

## 📂 Project files

| File | Description |
|---|---|
| [`project(facedetection).ipynb`](<./project(facedetection).ipynb>) | Model training and webcam detection workflow |
| [`report(aiml).docx`](<./report(aiml).docx>) | Project report |

## 📦 Outputs

The notebook writes its generated files to the current working directory:

| Output | Description |
|---|---|
| `mask_detector_mobilenet.keras` | Trained Keras model |
| `no_mask_captures/` | Frames captured when a no-mask face is detected |
| `no_mask_log.txt` | Timestamped no-mask event log |
| `session_video.avi` | Annotated webcam recording |
| `session_report.txt` | Session times, detection counts, and event timestamps |

## ⚠️ Notes and limitations

- The notebook currently combines model training and webcam inference in one code cell.
- Dataset and alert-audio paths are machine-specific and need to be updated before running.
- The notebook assumes the first class returned by the directory iterator represents **Mask** and the second represents **No Mask**. Check the printed class order and label mapping for your dataset.
- This project is an educational demonstration, not a medical or safety certification system. Results depend on training data, lighting, camera quality, and face visibility.
