# 🛡️ AI Surveillance System with Explainable Intelligence

## 🚀 Overview

This project is a real-time AI-powered surveillance system that detects suspicious human behavior using computer vision and machine learning.

Unlike traditional systems, it not only detects activity but also explains *why* a behavior is flagged as suspicious.

---

## 🎯 Key Features

* 🔍 Real-time object detection (YOLO)
* 🧠 Behavior analysis using AI logic
* ⚠️ Suspicious activity detection
* 📊 Explainable alerts (reason-based detection)
* 📁 Logging of suspicious events
* 🎥 Live camera monitoring

---

## 🧠 How It Works

The system tracks individuals and analyzes their behavior over time.

Examples of detection logic:

* Person idle for too long → flagged as suspicious
* Repeated entry/exit → abnormal behavior
* Sudden movement changes → alert triggered

---

## 🛠️ Tech Stack

* Python
* OpenCV
* YOLO (Ultralytics)
* DeepFace (optional face analysis)

---

## 📂 Project Structure

```
src/
 ├── main.py
 ├── pipeline.py
 ├── tracking.py
 ├── ai_analysis.py
 └── utils.py
```

---

## ⚙️ Setup Instructions

```bash
pip install -r requirements.txt
python src/main.py
```

---

## 🎥 Demo

(Add your demo video link here)

---

## 🔥 Future Improvements

* Web dashboard for monitoring
* Alert notifications (SMS/Email)
* Multi-camera support

---

## 👨‍💻 Author

Danda Sai Samith
