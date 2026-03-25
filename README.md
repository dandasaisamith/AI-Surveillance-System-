<<<<<<< HEAD
# AI-Surveillance-System-
=======
<<<<<<< HEAD
# AI Surveillance System

> **Developed by Sai Samith**

A professional, real-time AI Surveillance and Crowd Monitoring dashboard powered by Streamlit, YOLOv8, and DeepSORT. Designed to provide autonomous security monitoring, track suspicious behaviors, and automatically trigger alerts for overcrowding and dangerous objects.

## Features

- **Real-Time Object & Person Tracking**: Uses DeepSORT and YOLOv8 to seamlessly detect and assign unique IDs to individuals without relying on heavy facial recognition.
- **Suspicious Behavior Detection**: Analyzes person dwell time, loitering, and restricted zone proximity. Automatically flags instances of abnormal behavior.
- **Danger Detection Pipeline**: Instantly identifies dangerous objects (knives, firearms, etc.) and triggers high-priority alerts and voice announcements.
- **Crowd Monitoring System**: Keeps track of active objects and triggers automated Siren & TTS voice alerts when overcrowding thresholds are exceeded.
- **Insights Dashboard**: Dynamically surfaces recorded incidents, separating suspicious entities into cleanly formatted, screenshot-backed Insight logs without duplicate spam.
- **Persistent Data Logging**: Dumps historical system tracking parameters and security snapshots directly to disk for offline auditing.

## Tech Stack

- **Python 3**
- **Streamlit**: Web Dashboard & UI layer
- **Ultralytics YOLOv8**: Real-time Object Detection
- **DeepSORT**: Multi-Object Tracking
- **OpenCV**: Video capture and frame processing
- **PyTTSx3 / Winsound**: Intelligent voice alerts and sirens

## Repository Structure
- `app.py`: The main Streamlit dashboard application.
- `main.py`: The background tracking, detection, and logic processing engine.
- `/data_logs/`: Persistent database logs from tracking instances.
- `/suspicious_logs/`: Safely stored screenshots for identified suspicious and dangerous entities.

## How to Run

1. **Install Dependencies** (Make sure you have PyTorch, OpenCV, and Streamlit installed)
```bash
pip install -r requirements.txt
```

2. **Run the Streamlit Dashboard**
```bash
streamlit run app.py
```

3. **Navigate the Dashboard**
Use the sidebar controls to start the camera stream, adjust AI settings, and view different analytical tabs like **Insights** and **Crowd Monitoring**.

---
*Developed by Sai Samith*
=======
# AI-Surveillance-System-
>>>>>>> c597df76e47d7c9ebe284d30ea0eb64e330f6338
>>>>>>> 989dc93 (Initial clean commit (no model files))
