# AI Surveillance System - Setup Instructions

## Installation Steps

### 1. Install Dependencies
```bash
# Activate virtual environment
.venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA support (for RTX 4060)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

### 2. Download YOLO Model
The YOLOv8n model will be downloaded automatically on first run.

### 3. Run Application
```bash
streamlit run app.py
```

## Project Structure
```
ai_survilliance/
├── app.py              # Main Streamlit UI
├── pipeline.py         # Detection & tracking pipeline
├── tracking.py         # DeepSORT tracker
├── ai_analysis.py      # Gemini & DeepFace integration
├── utils.py            # Safe imports & helpers
├── requirements.txt    # Dependencies
└── suspicious_logs/    # Saved suspicious activity images
```

## Features
- ✅ Real-time person detection (YOLO)
- ✅ Multi-person tracking (DeepSORT)
- ✅ Duplicate prevention (track memory)
- ✅ Suspicious behavior detection
- ✅ Restricted zone monitoring
- ✅ AI-powered analysis (Gemini)
- ✅ Person attributes (DeepFace)
- ✅ GPU acceleration (CUDA)
- ✅ Optimized FPS (20-30 FPS)

## Troubleshooting

### If modules fail to import:
The system will continue to work with reduced functionality. Check System Status in sidebar.

### If camera doesn't start:
- Check camera permissions
- Try different camera index (modify safe_camera_init in utils.py)
- Ensure no other app is using the camera

### If FPS is low:
- Ensure GPU is being used (check System Status)
- Reduce frame resolution in pipeline.py
- Increase process_every_n in pipeline.py

### If Gemini API fails:
- Check API key in ai_analysis.py
- System will use rule-based fallback analysis

## Configuration

### Adjust Detection Sensitivity
In `pipeline.py`, modify:
```python
conf=0.4  # Lower = more detections, Higher = fewer false positives
```

### Adjust Idle Threshold
In `tracking.py`, modify:
```python
self.idle_threshold = 8.0  # seconds
self.movement_threshold = 20.0  # pixels
```

### Change Frame Processing Rate
In `pipeline.py`, modify:
```python
self.process_every_n = 2  # Process every Nth frame (higher = faster but less accurate)
```

## Notes
- System is designed to never crash even if modules fail
- All AI features have fallback mechanisms
- Logs are saved to `suspicious_logs/` directory
- Track IDs persist across frames to prevent duplicates
