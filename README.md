Face Mask Detection System:-

Project Overview:
This project implements a real-time face mask detection system using deep learning and computer vision.  
It uses **MobileNetV2** (transfer learning) with **OpenCV** for webcam input, face detection, and mask classification.

Features:
- Real-time detection of masked / unmasked faces  
- Green bounding box = Mask, Red = No Mask  
- Automatic saving of no-mask snapshots  
- Generation of daily summary report of no-mask events  
- Privacy-preserving face blur for no-mask detections  
- Real-time compliance percentages (Mask / No Mask) overlay
  
Technologies Used:
- Python 3.x
- TensorFlow / Keras
- OpenCV
- NumPy

How to Run:
Install requirements

`bash
pip install tensorflow opencv-python numpy

Prepare model:
Ensure you have a trained model file, e.g. mask_detector_mobilenet.keras

Place it in your project directory (or update the path in the code)

Run the detector:
bash
Copy
Edit
python mask_detector.py
➡ The camera window will open.
➡ Press q or Esc to stop detection.

Output Files:
no_mask_captures/ → Saves snapshots of no-mask detections.

daily_no_mask_summary.txt → Appends no-mask events with timestamps.

(Optional enhancement) session_video.avi → Could be added to save entire session video.
