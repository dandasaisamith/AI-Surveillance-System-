import time
import os
from pathlib import Path

# Safe imports with fallbacks
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    DEVICE = "cuda" if TORCH_AVAILABLE else "cpu"
except:
    torch = None
    TORCH_AVAILABLE = False
    DEVICE = "cpu"

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except:
    DeepSort = None
    DEEPSORT_AVAILABLE = False

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except:
    genai = None
    GENAI_AVAILABLE = False

try:
    from deepface import DeepFace
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    DEEPFACE_AVAILABLE = True
except:
    DeepFace = None
    DEEPFACE_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except:
    YOLO = None
    YOLO_AVAILABLE = False

def get_system_status():
    return {
        "torch": TORCH_AVAILABLE,
        "device": DEVICE,
        "deepsort": DEEPSORT_AVAILABLE,
        "genai": GENAI_AVAILABLE,
        "deepface": DEEPFACE_AVAILABLE,
        "yolo": YOLO_AVAILABLE
    }

def safe_camera_init(index=0):
    import cv2
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == 'nt' else [cv2.CAP_ANY]
    
    for backend in backends:
        try:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                return cap
            cap.release()
        except:
            pass
    return None

class TrackMemory:
    def __init__(self, cooldown=10.0):
        self.seen_ids = set()
        self.last_seen = {}
        self.cooldown = cooldown
    
    def is_new(self, track_id):
        now = time.time()
        if track_id not in self.seen_ids:
            self.seen_ids.add(track_id)
            self.last_seen[track_id] = now
            return True
        
        if now - self.last_seen.get(track_id, 0) > self.cooldown:
            self.last_seen[track_id] = now
            return True
        
        return False
    
    def update(self, track_id):
        self.last_seen[track_id] = time.time()
