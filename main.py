import argparse
import csv
import json
import os
import queue
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
import sys
import threading
import time

PROJECT_ROOT = Path(__file__).resolve().parent

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("DEEPFACE_HOME", str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None

DEEPFACE_IMPORT_ERROR = None
try:
    from deepface import DeepFace
except Exception as exc:
    DeepFace = None
    DEEPFACE_IMPORT_ERROR = str(exc)

# Prefer the vendored Ultralytics checkout in this repository over any top-level
# namespace package with the same name.
VENDORED_ULTRALYTICS_ROOT = PROJECT_ROOT / "ultralytics"
YOLO_CONFIG_ROOT = PROJECT_ROOT / ".yolo"
if VENDORED_ULTRALYTICS_ROOT.is_dir():
    sys.path.insert(0, str(VENDORED_ULTRALYTICS_ROOT))
YOLO_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_ROOT))

from ultralytics import YOLO

DEFAULT_MODEL_PATH = PROJECT_ROOT / "yolov5su.pt"
LIGHTWEIGHT_MODEL_CANDIDATES = [
    PROJECT_ROOT / "yolov8n.pt",
    PROJECT_ROOT / "yolov5n.pt",
    PROJECT_ROOT / "yolov5su.pt",
]
DEFAULT_FALLBACK_IMAGE = VENDORED_ULTRALYTICS_ROOT / "ultralytics" / "assets" / "bus.jpg"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_logs"
DEFAULT_ALERT_DIR = DEFAULT_OUTPUT_DIR / "alerts"
LAST_SYSTEM_STATE = {}

COLOR_BG = (32, 18, 11)  # #0B1220 in BGR
COLOR_PANEL = (39, 24, 17)  # #111827 in BGR
COLOR_TEXT = (235, 231, 229)  # #E5E7EB in BGR
COLOR_TEAL = (238, 211, 34)  # #22D3EE in BGR
COLOR_VIOLET = (246, 92, 139)  # #8B5CF6 in BGR
COLOR_AMBER = (11, 158, 245)  # #F59E0B in BGR
COLOR_MUTED = (148, 163, 184)
COLOR_PANEL_ALT = (46, 32, 22)
COLOR_BORDER = (70, 82, 96)

UI_STATE = {
    "active_tab": 0,
    "tab_rects": [],
    "transition_from": None,
    "transition_started_at": 0.0,
    "transition_canvas": None,
    "last_dashboard": None,
    "analytics_alert_scroll": 0,
}

try:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(2)
except Exception:
    pass

try:
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def parse_source(value: str):
    """Treat integer-like sources as camera indices and anything else as a file/URL."""
    return int(value) if value.isdigit() else value


def choose_default_model_path() -> Path:
    for candidate in LIGHTWEIGHT_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_MODEL_PATH


def open_capture(source, width: int = 640, height: int = 480):
    """Prefer the DirectShow backend for Windows camera indices and keep the stream lightweight."""
    if isinstance(source, int) and os.name == "nt":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    return cap


def run_fallback_inference(model):
    """Run a single-image inference when no live camera source is available."""
    if not DEFAULT_FALLBACK_IMAGE.exists():
        print("Error: No webcam is available and no fallback image was found.")
        return

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DEFAULT_OUTPUT_DIR / "fallback_prediction.jpg"
    results = model(str(DEFAULT_FALLBACK_IMAGE), verbose=False)
    annotated_frame = results[0].plot()
    cv2.imwrite(str(output_path), annotated_frame)
    print(f"No webcam was available, so a fallback prediction was saved to {output_path}.")


def iterate_names(names):
    if isinstance(names, dict):
        return names.items()
    return enumerate(names)


def parse_label_filter(raw_value: str | None, names) -> set[int] | None:
    """Parse class filters from comma-separated ids or class names."""
    if raw_value is None:
        return None

    items = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not items:
        return None

    name_to_id = {str(name).lower(): int(idx) for idx, name in iterate_names(names)}
    resolved = set()
    for item in items:
        if item.isdigit():
            resolved.add(int(item))
            continue
        key = item.lower()
        if key not in name_to_id:
            valid = ", ".join(str(name) for _, name in iterate_names(names))
            raise ValueError(f"Unknown class filter {item!r}. Available classes: {valid}")
        resolved.add(name_to_id[key])
    return resolved


def get_class_name(names, cls_id) -> str:
    if cls_id is None:
        return "object"
    try:
        cls_id = int(cls_id)
    except (TypeError, ValueError):
        return str(cls_id)

    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def resolve_class_id(names, target_name: str) -> int | None:
    target_name = target_name.lower()
    for cls_id, class_name in iterate_names(names):
        if str(class_name).lower() == target_name:
            return int(cls_id)
    return None


def gpu_available() -> bool:
    return bool(torch.cuda.is_available())


def resolve_runtime() -> dict:
    use_gpu = gpu_available()
    return {
        "device": 0 if use_gpu else "cpu",
        "device_label": "cuda:0" if use_gpu else "cpu",
        "use_half": bool(use_gpu),
        "gpu_enabled": use_gpu,
        "gpu_name": torch.cuda.get_device_name(0) if use_gpu else "CPU",
    }


def warmup_model(model, runtime: dict, imgsz: int) -> None:
    warmup_frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    with torch.inference_mode():
        model(
            warmup_frame,
            verbose=False,
            imgsz=imgsz,
            device=runtime["device"],
            half=runtime["use_half"],
        )


def clip_box(box, frame_shape):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return x1, y1, x2, y2


def xyxy_to_ltwh(box):
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def build_detections(result):
    """Convert YOLO detections to DeepSORT's expected input format."""
    detections = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections

    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy().astype(int)
    for box, conf, cls_id in zip(xyxy, confs, clss):
        detections.append((xyxy_to_ltwh(box), float(conf), int(cls_id)))
    return detections


def build_tracker(args):
    return DeepSort(
        max_age=args.max_age,
        n_init=args.n_init,
        max_cosine_distance=args.max_cosine_distance,
        embedder="mobilenet",
        embedder_gpu=gpu_available(),
    )


class SpeechAnnouncer:
    """Speak short event messages without blocking the vision loop."""

    def __init__(self, enabled: bool, rate: int):
        self.enabled = enabled and pyttsx3 is not None
        self.rate = rate
        self.last_spoken_at: dict[str, float] = {}
        self.message_queue: queue.Queue[str] = queue.Queue(maxsize=32)
        self.stop_event = threading.Event()
        self.worker = None

        if self.enabled:
            self.worker = threading.Thread(target=self._run, name="tts-worker", daemon=True)
            self.worker.start()

    def _run(self) -> None:
        pythoncom = None
        engine = None
        try:
            try:
                import pythoncom as pythoncom_module

                pythoncom = pythoncom_module
                pythoncom.CoInitialize()
            except Exception:
                pythoncom = None

            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)

            while not self.stop_event.is_set():
                try:
                    message = self.message_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if not message:
                    continue
                engine.say(message)
                engine.runAndWait()
        except Exception as exc:
            self.enabled = False
            print(f"Text-to-speech unavailable: {exc}")
        finally:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass
            if pythoncom is not None:
                try:
                    pythoncom.CoUninitialize()
                except Exception:
                    pass

    def say(self, message: str, key: str | None = None, cooldown_seconds: float = 0.0) -> None:
        if not self.enabled:
            return

        now = time.time()
        cache_key = key or message
        if cooldown_seconds > 0.0 and now - self.last_spoken_at.get(cache_key, 0.0) < cooldown_seconds:
            return
        self.last_spoken_at[cache_key] = now

        try:
            self.message_queue.put_nowait(message)
        except queue.Full:
            pass

    def stop(self) -> None:
        self.stop_event.set()
        if self.worker is not None:
            self.worker.join(timeout=1.0)


class AudioAlertController:
    """Play lightweight siren cues and TTS without spamming the loop."""

    def __init__(self, speaker: SpeechAnnouncer, enabled: bool = True):
        self.speaker = speaker
        self.enabled = enabled
        self.muted = False
        self.last_trigger_at: dict[str, float] = {}
        self.queue: queue.Queue[dict] = queue.Queue(maxsize=16)
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._run, name="audio-alerts", daemon=True)
        self.worker.start()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                item = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item.get("siren"):
                self._play_siren()
            if item.get("speech") and item.get("message"):
                self.speaker.say(item["message"], key=item.get("speech_key"), cooldown_seconds=item.get("cooldown", 0.0))

    def _play_siren(self) -> None:
        if os.name == "nt":
            try:
                import winsound

                for freq, duration in ((1200, 160), (820, 160), (1200, 160), (820, 160)):
                    winsound.Beep(freq, duration)
            except Exception:
                pass
        else:
            print("\a", end="", flush=True)

    def trigger(self, message: str, alert_key: str, cooldown_seconds: float = 5.0, siren: bool = True, speech: bool = True) -> bool:
        if not self.enabled or self.muted:
            return False
        now = time.time()
        if now - self.last_trigger_at.get(alert_key, 0.0) < cooldown_seconds:
            return False
        self.last_trigger_at[alert_key] = now
        try:
            self.queue.put_nowait(
                {
                    "message": message,
                    "speech_key": f"audio:{alert_key}",
                    "cooldown": cooldown_seconds,
                    "siren": siren,
                    "speech": speech,
                }
            )
        except queue.Full:
            pass
        return True

    def toggle_mute(self) -> bool:
        self.muted = not self.muted
        return self.muted

    def stop(self) -> None:
        self.stop_event.set()
        self.worker.join(timeout=1.0)


class CommandTriggerSystem:
    """A simple in-process command bus for reliable demo control."""

    def __init__(self):
        self.command_queue: queue.Queue[str] = queue.Queue()
        self.status = "Command trigger ready"

    def trigger(self, command: str, source: str = "system") -> None:
        self.status = f"{source.title()} command: {command}"
        self.command_queue.put(command)

    def poll_commands(self) -> list[str]:
        commands = []
        while True:
            try:
                commands.append(self.command_queue.get_nowait())
            except queue.Empty:
                return commands


class VoiceCommandListener:
    """Listen for a few simple voice commands in the background."""

    COMMANDS = {
        "start detection": "start detection",
        "stop detection": "stop detection",
        "enable alerts": "enable alerts",
        "disable alerts": "disable alerts",
    }

    def __init__(self, enabled: bool):
        self.enabled = enabled and sr is not None
        self.offline = False
        self.command_queue: queue.Queue[str] = queue.Queue()
        self.status = "Voice off"
        self.stop_listening = None
        self.recognizer = None
        self.microphone = None
        self.last_error_at = 0.0
        self.last_heard_at = 0.0

    def start(self) -> None:
        if not self.enabled:
            return

        try:
            self.recognizer = sr.Recognizer()
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.5
            self.recognizer.non_speaking_duration = 0.25
            self.microphone = sr.Microphone()

            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.4)

            self.stop_listening = self.recognizer.listen_in_background(
                self.microphone,
                self._callback,
                phrase_time_limit=2.2,
            )
            self.status = "Voice listening"
        except Exception as exc:
            self.enabled = False
            self.status = "Voice unavailable"
            print(f"Voice control unavailable: {exc}")

    def _callback(self, recognizer, audio) -> None:
        try:
            transcript = recognizer.recognize_google(audio).lower().strip()
        except sr.UnknownValueError:
            return
        except sr.RequestError:
            now = time.time()
            if now - self.last_error_at > 8.0:
                self.status = "Voice recognition offline"
                self.last_error_at = now
            self.offline = True
            self.enabled = False
            if self.stop_listening is not None:
                try:
                    self.stop_listening(wait_for_stop=False)
                except Exception:
                    pass
                self.stop_listening = None
            return
        except Exception:
            return

        command = self.parse_command(transcript)
        if command is None:
            return

        self.last_heard_at = time.time()
        self.status = f"Voice heard: {command}"
        self.command_queue.put(command)

    def parse_command(self, transcript: str) -> str | None:
        for phrase, command in self.COMMANDS.items():
            if phrase in transcript:
                return command
        return None

    def poll_commands(self) -> list[str]:
        commands = []
        while True:
            try:
                commands.append(self.command_queue.get_nowait())
            except queue.Empty:
                return commands

    def stop(self) -> None:
        if self.stop_listening is not None:
            self.stop_listening(wait_for_stop=False)


class DetectionEventAnnouncer:
    """Speak when a new class appears after being absent."""

    def __init__(self, speaker: SpeechAnnouncer, audio_controller: AudioAlertController | None = None):
        self.speaker = speaker
        self.audio_controller = audio_controller
        self.present_classes: set[str] = set()

    def update(self, active_counts: Counter) -> list[str]:
        current = {class_name for class_name, count in active_counts.items() if count > 0}
        new_classes = sorted(current - self.present_classes)
        muted = self.audio_controller.muted if self.audio_controller is not None else False
        for class_name in new_classes[:2]:
            if muted:
                continue
            self.speaker.say(
                f"{class_name} detected",
                key=f"detected:{class_name}",
                cooldown_seconds=8.0,
            )
        self.present_classes = current
        return new_classes

    def reset(self) -> None:
        self.present_classes.clear()


class LineCounter:
    """Track unique stable IDs and keep active vs. unique counts separated."""

    def __init__(self, axis: str, position_ratio: float, direction: str):
        self.axis = axis
        self.position_ratio = min(max(position_ratio, 0.05), 0.95)
        self.direction = direction
        self.track_positions: dict[int, tuple[float, float]] = {}
        self.counted_track_ids: set[int] = set()
        self.unique_track_ids: set[int] = set()
        self.class_track_ids: dict[str, set[int]] = defaultdict(set)
        self.track_classes: dict[int, str] = {}
        self.active_track_ids: set[int] = set()
        self.active_class_counts = Counter()
        self.total_crossings = 0
        self.class_counts = Counter()

    def line_value(self, frame_shape) -> int:
        height, width = frame_shape[:2]
        if self.axis == "horizontal":
            return int(height * self.position_ratio)
        return int(width * self.position_ratio)

    def register(
        self,
        track_id: int,
        class_name: str,
        center: tuple[float, float],
        frame_shape,
        track_hits: int = 0,
    ) -> None:
        previous = self.track_positions.get(track_id)
        self.track_positions[track_id] = center
        if track_hits < 3:
            return

        if track_id not in self.counted_track_ids:
            self.counted_track_ids.add(track_id)
            self.unique_track_ids.add(track_id)
            self.track_classes[track_id] = class_name
            self.class_track_ids[class_name].add(track_id)
            self.class_counts[class_name] = len(self.class_track_ids[class_name])
            self.total_crossings = len(self.unique_track_ids)
            return

        stable_class = self.track_classes.get(track_id, class_name)
        if track_id not in self.class_track_ids[stable_class]:
            self.class_track_ids[stable_class].add(track_id)
            self.class_counts[stable_class] = len(self.class_track_ids[stable_class])

    def update_active_counts(self, active_tracks: list[dict]) -> None:
        self.active_track_ids = {track["track_id"] for track in active_tracks}
        self.active_class_counts = Counter(
            self.track_classes.get(track["track_id"], track["class_name"])
            for track in active_tracks
        )

    @property
    def unique_total(self) -> int:
        return len(self.unique_track_ids)

    @property
    def active_total(self) -> int:
        return len(self.active_track_ids)

    def crossed_line(self, previous, current, line_pos: int) -> bool:
        if self.axis == "horizontal":
            return (previous[1] < line_pos <= current[1]) or (previous[1] > line_pos >= current[1])
        return (previous[0] < line_pos <= current[0]) or (previous[0] > line_pos >= current[0])

    def direction_matches(self, previous, current) -> bool:
        if self.direction == "any":
            return True

        dx = current[0] - previous[0]
        dy = current[1] - previous[1]
        if self.axis == "horizontal":
            if self.direction == "down":
                return dy > 0
            if self.direction == "up":
                return dy < 0
        else:
            if self.direction == "right":
                return dx > 0
            if self.direction == "left":
                return dx < 0
        return False

    def draw(self, frame):
        return None


class SuspiciousBehaviorDetector:
    """Mark a person as suspicious when they remain idle for too long."""

    def __init__(self, person_class_id: int | None, idle_seconds: float, movement_threshold: float):
        self.person_class_id = person_class_id
        self.idle_seconds = idle_seconds
        self.movement_threshold = movement_threshold
        self.track_samples: dict[int, deque[tuple[float, tuple[float, float]]]] = defaultdict(deque)
        self.last_seen_at: dict[int, float] = {}
        self.current_suspicious_ids: set[int] = set()

    def update(self, active_tracks: list[dict], now: float):
        if self.person_class_id is None:
            self.current_suspicious_ids = set()
            return set(), [], {}

        new_suspicious_events = []
        current_suspicious_ids = set()
        track_analysis = {}

        for track in active_tracks:
            if track["class_id"] != self.person_class_id:
                continue

            track_id = track["track_id"]
            self.last_seen_at[track_id] = now
            samples = self.track_samples[track_id]
            samples.append((now, track["center"]))
            while len(samples) > 1 and now - samples[1][0] >= self.idle_seconds:
                samples.popleft()

            dwell_time = 0.0 if len(samples) < 2 else max(0.0, now - samples[0][0])
            if len(samples) < 2 or dwell_time < self.idle_seconds:
                track_analysis[track_id] = {
                    "track_id": track_id,
                    "dwell_time": dwell_time,
                    "suspicious": False,
                }
                continue

            reference = samples[0][1]
            max_distance = max(
                np.hypot(sample_center[0] - reference[0], sample_center[1] - reference[1])
                for _, sample_center in samples
            )
            is_suspicious = max_distance <= self.movement_threshold
            track_analysis[track_id] = {
                "track_id": track_id,
                "dwell_time": dwell_time,
                "movement": max_distance,
                "suspicious": is_suspicious,
            }
            if is_suspicious:
                current_suspicious_ids.add(track_id)
                if track_id not in self.current_suspicious_ids:
                    new_suspicious_events.append(track)

        stale_after = self.idle_seconds * 1.5
        for track_id in list(self.last_seen_at):
            if now - self.last_seen_at[track_id] > stale_after:
                self.last_seen_at.pop(track_id, None)
                self.track_samples.pop(track_id, None)

        self.current_suspicious_ids = current_suspicious_ids
        return current_suspicious_ids, new_suspicious_events, track_analysis


class IntentPredictor:
    """Rule-based intent scoring for lightweight, explainable behavior labels."""

    def __init__(self, person_class_id: int | None, history_seconds: float = 12.0):
        self.person_class_id = person_class_id
        self.history_seconds = history_seconds
        self.track_samples: dict[int, deque[tuple[float, tuple[float, float]]]] = defaultdict(deque)
        self.first_seen_at: dict[int, float] = {}
        self.last_seen_at: dict[int, float] = {}
        self.last_update_at: dict[int, float] = {}
        self.time_in_zone: dict[int, float] = defaultdict(float)
        self.last_intent: dict[int, str] = {}

    def restricted_zone(self, frame_shape) -> tuple[int, int, int, int]:
        height, width = frame_shape[:2]
        x1 = int(width * 0.72)
        y1 = int(height * 0.18)
        x2 = int(width * 0.96)
        y2 = int(height * 0.82)
        return x1, y1, x2, y2

    def _distance_to_rect(self, center: tuple[float, float], rect: tuple[int, int, int, int]) -> float:
        x, y = center
        x1, y1, x2, y2 = rect
        dx = max(x1 - x, 0.0, x - x2)
        dy = max(y1 - y, 0.0, y - y2)
        return float(np.hypot(dx, dy))

    def _zone_proximity(self, center: tuple[float, float], rect: tuple[int, int, int, int], frame_shape) -> float:
        diagonal = max(1.0, float(np.hypot(frame_shape[1], frame_shape[0])))
        distance = self._distance_to_rect(center, rect)
        if distance <= 0.0:
            return 1.0
        return max(0.0, 1.0 - min(distance / (diagonal * 0.45), 1.0))

    def update(self, active_tracks: list[dict], now: float, frame_shape) -> tuple[dict[int, dict], list[dict]]:
        zone_rect = self.restricted_zone(frame_shape)
        analysis: dict[int, dict] = {}
        new_events: list[dict] = []

        for track in active_tracks:
            track_id = track["track_id"]
            self.last_seen_at[track_id] = now
            self.first_seen_at.setdefault(track_id, now)
            delta = max(0.0, now - self.last_update_at.get(track_id, now))
            self.last_update_at[track_id] = now

            if track["class_id"] != self.person_class_id:
                analysis[track_id] = {
                    "intent": "normal movement",
                    "speed": 0.0,
                    "direction_consistency": 0.0,
                    "time_in_zone": 0.0,
                    "time_in_frame": max(0.0, now - self.first_seen_at[track_id]),
                    "zone_proximity": 0.0,
                    "risk_score": min(0.35, 0.12 + track.get("area", 0) / 300000.0),
                    "suspicious_intent": False,
                }
                continue

            samples = self.track_samples[track_id]
            samples.append((now, track["center"]))
            while len(samples) > 1 and now - samples[0][0] > self.history_seconds:
                samples.popleft()

            if self._distance_to_rect(track["center"], zone_rect) <= 0.0:
                self.time_in_zone[track_id] += delta

            time_in_frame = max(0.0, now - self.first_seen_at[track_id])
            time_in_zone = self.time_in_zone[track_id]
            speed = 0.0
            direction_consistency = 0.0
            zone_delta = 0.0

            if len(samples) >= 2:
                duration = max(1e-6, samples[-1][0] - samples[0][0])
                path_length = sum(
                    np.hypot(curr[1][0] - prev[1][0], curr[1][1] - prev[1][1])
                    for prev, curr in zip(samples, list(samples)[1:])
                )
                displacement = np.hypot(
                    samples[-1][1][0] - samples[0][1][0],
                    samples[-1][1][1] - samples[0][1][1],
                )
                speed = path_length / duration
                direction_consistency = 0.0 if path_length <= 1e-6 else min(1.0, displacement / path_length)
                zone_delta = self._distance_to_rect(samples[0][1], zone_rect) - self._distance_to_rect(samples[-1][1], zone_rect)

            zone_proximity = self._zone_proximity(track["center"], zone_rect, frame_shape)
            loitering = speed < 14.0 and time_in_frame >= 6.0 and (time_in_zone >= 1.5 or zone_proximity >= 0.55)
            approaching = speed >= 8.0 and direction_consistency >= 0.58 and zone_delta > 28.0 and zone_proximity >= 0.4

            if loitering:
                intent = "loitering"
            elif approaching:
                intent = "approaching restricted zone"
            else:
                intent = "normal movement"

            risk_score = 0.12
            risk_score += min(0.22, time_in_frame / 30.0)
            risk_score += min(0.22, time_in_zone / 8.0)
            risk_score += zone_proximity * 0.18
            risk_score += min(0.12, max(0.0, 20.0 - speed) / 80.0)
            if loitering:
                risk_score += 0.28
            if approaching:
                risk_score += 0.34
            if direction_consistency > 0.72:
                risk_score += 0.08
            risk_score = min(0.99, risk_score)

            suspicious_intent = intent != "normal movement"
            previous_intent = self.last_intent.get(track_id)
            if suspicious_intent and previous_intent != intent:
                new_events.append(track)
            self.last_intent[track_id] = intent

            analysis[track_id] = {
                "intent": intent,
                "speed": round(float(speed), 2),
                "direction_consistency": round(float(direction_consistency), 2),
                "time_in_zone": round(float(time_in_zone), 1),
                "time_in_frame": round(float(time_in_frame), 1),
                "zone_proximity": round(float(zone_proximity), 2),
                "risk_score": round(float(risk_score), 2),
                "suspicious_intent": suspicious_intent,
            }

        stale_after = self.history_seconds * 1.5
        for track_id in list(self.last_seen_at):
            if now - self.last_seen_at[track_id] <= stale_after:
                continue
            self.last_seen_at.pop(track_id, None)
            self.first_seen_at.pop(track_id, None)
            self.last_update_at.pop(track_id, None)
            self.track_samples.pop(track_id, None)
            self.time_in_zone.pop(track_id, None)
            self.last_intent.pop(track_id, None)

        return analysis, new_events


class PersonIntelligenceManager:
    """Cache lightweight person analytics and run DeepFace asynchronously when available."""

    def __init__(
        self,
        person_class_id: int | None,
        analyze_every_frames: int = 12,
        stale_seconds: float = 1.5,
        analyze_cooldown_seconds: float = 10.0,
    ):
        self.person_class_id = person_class_id
        self.analyze_every_frames = max(10, analyze_every_frames)
        self.stale_seconds = stale_seconds
        self.analyze_cooldown_seconds = analyze_cooldown_seconds
        self.enabled = DeepFace is not None and person_class_id is not None
        self.records: dict[int, dict] = {}
        self.lock = threading.Lock()
        self.pending_track_ids: set[int] = set()
        self.job_queue: queue.Queue[dict] = queue.Queue(maxsize=12)
        self.stop_event = threading.Event()
        self.worker = None

        if self.enabled:
            self.worker = threading.Thread(target=self._run, name="person-intelligence", daemon=True)
            self.worker.start()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                job = self.job_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            track_id = job["track_id"]
            try:
                result = self._analyze_crop(job["crop"])
                if result is None:
                    with self.lock:
                        record = self.records.get(track_id)
                        if record is not None:
                            record["last_analyzed_epoch"] = job["now"]
                    continue
                with self.lock:
                    record = self.records.get(track_id)
                    if record is None:
                        continue
                    record["age"] = result["age"]
                    record["gender"] = result["gender"]
                    record["emotion"] = result["emotion"]
                    record["last_analyzed_frame"] = job["frame_index"]
                    record["last_analyzed_epoch"] = job["now"]
                    record["analysis_ready"] = True
                    record["analysis_error"] = None
            except Exception as exc:
                with self.lock:
                    record = self.records.get(track_id)
                    if record is not None:
                        record["analysis_error"] = str(exc)
                        record["last_analyzed_epoch"] = job["now"]
                        if not record.get("analysis_ready"):
                            record["age"] = "Pending"
                            record["gender"] = "Pending"
                            record["emotion"] = "Pending"
            finally:
                with self.lock:
                    self.pending_track_ids.discard(track_id)

    def _analyze_crop(self, crop) -> dict | None:
        if DeepFace is None or crop is None or crop.size == 0:
            return None

        height, width = crop.shape[:2]
        if height < 24 or width < 24:
            return None

        max_side = max(height, width)
        if max_side > 224:
            scale = 224.0 / max_side
            crop = cv2.resize(crop, (max(24, int(width * scale)), max(24, int(height * scale))))

        analysis = DeepFace.analyze(
            img_path=crop,
            actions=("age", "gender", "emotion"),
            enforce_detection=True,
            detector_backend="opencv",
            silent=True,
        )
        if isinstance(analysis, list):
            analysis = analysis[0] if analysis else {}
        if not isinstance(analysis, dict):
            return None

        age = analysis.get("age")
        dominant_gender = analysis.get("dominant_gender")
        if dominant_gender is None and isinstance(analysis.get("gender"), dict):
            dominant_gender = max(analysis["gender"], key=analysis["gender"].get)
        dominant_emotion = analysis.get("dominant_emotion")
        if dominant_emotion is None and isinstance(analysis.get("emotion"), dict):
            dominant_emotion = max(analysis["emotion"], key=analysis["emotion"].get)

        return {
            "age": None if age is None else int(round(float(age))),
            "gender": dominant_gender or "Unknown",
            "emotion": dominant_emotion or "neutral",
        }

    def _ensure_record(self, track: dict, now: float) -> dict:
        track_id = track["track_id"]
        timestamp = datetime.fromtimestamp(now).strftime("%H:%M:%S")
        record = self.records.get(track_id)
        if record is None:
            record = {
                "track_id": track_id,
                "age": "Analyzing" if self.enabled else "Unavailable",
                "gender": "Analyzing" if self.enabled else "Unavailable",
                "emotion": "Analyzing" if self.enabled else "Unavailable",
                "first_seen": timestamp,
                "first_seen_epoch": now,
                "last_seen": timestamp,
                "last_seen_epoch": now,
                "entry_time": timestamp,
                "exit_time": None,
                "total_duration": 0.0,
                "movement_state": "moving",
                "active": True,
                "last_analyzed_frame": -self.analyze_every_frames,
                "last_analyzed_epoch": 0.0,
                "analysis_attempts": 0,
                "analysis_ready": False,
                "analysis_error": None,
            }
            self.records[track_id] = record
        return record

    def _queue_analysis(self, track_id: int, crop, frame_index: int, now: float) -> bool:
        if not self.enabled or track_id in self.pending_track_ids:
            return False
        self.pending_track_ids.add(track_id)
        try:
            self.job_queue.put_nowait(
                {
                    "track_id": track_id,
                    "crop": crop.copy(),
                    "frame_index": frame_index,
                    "now": now,
                }
            )
            return True
        except queue.Full:
            self.pending_track_ids.discard(track_id)
            return False

    def update(self, active_tracks: list[dict], frame, now: float, frame_index: int) -> dict:
        timestamp = datetime.fromtimestamp(now).strftime("%H:%M:%S")
        active_ids = set()

        with self.lock:
            for track in active_tracks:
                if track["class_id"] != self.person_class_id:
                    continue
                track_id = track["track_id"]
                active_ids.add(track_id)
                record = self._ensure_record(track, now)
                record["last_seen"] = timestamp
                record["last_seen_epoch"] = now
                record["exit_time"] = None
                record["active"] = True
                record["movement_state"] = "idle" if track.get("movement_speed", 0.0) < 8.0 or track.get("suspicious") else "moving"
                record["total_duration"] = round(max(0.0, now - float(record.get("first_seen_epoch", now))), 1)

                time_since_last_analyze = now - float(record.get("last_analyzed_epoch", 0.0))
                should_analyze = (
                    self.enabled
                    and track.get("hits", 0) >= 5
                    and track.get("area", 0) >= 2500
                    and time_since_last_analyze >= 8.0
                )
                if should_analyze:
                    x1, y1, x2, y2 = track["box"]
                    pad_x = max(8, int((x2 - x1) * 0.08))
                    pad_y = max(8, int((y2 - y1) * 0.08))
                    crop = frame[max(0, y1 - pad_y):min(frame.shape[0], y2 + pad_y), max(0, x1 - pad_x):min(frame.shape[1], x2 + pad_x)]
                    if self._queue_analysis(track_id, crop, frame_index, now):
                        record["last_analyzed_frame"] = frame_index
                        record["last_analyzed_epoch"] = now
                        record["analysis_attempts"] = int(record.get("analysis_attempts", 0)) + 1

            for track_id, record in self.records.items():
                if track_id in active_ids:
                    continue
                last_seen_epoch = float(record.get("last_seen_epoch", now))
                if now - last_seen_epoch >= self.stale_seconds:
                    record["active"] = False
                    if record["exit_time"] is None:
                        record["exit_time"] = record["last_seen"]

            return self.snapshot_locked()

    def snapshot_locked(self) -> dict:
        ordered = sorted(
            self.records.values(),
            key=lambda item: (not item.get("active", False), -float(item.get("total_duration", 0.0)), item.get("track_id", 0)),
        )
        return {
            "enabled": self.enabled,
            "available": DeepFace is not None,
            "error": DEEPFACE_IMPORT_ERROR,
            "records": [dict(item) for item in ordered],
        }

    def snapshot(self) -> dict:
        with self.lock:
            return self.snapshot_locked()

    def stop(self) -> None:
        self.stop_event.set()
        if self.worker is not None:
            self.worker.join(timeout=1.0)


class AlertManager:
    """Trigger visual overlays and optional saved-frame snapshots for alerts."""

    def __init__(
        self,
        threshold: int,
        cooldown_seconds: float,
        class_filter: set[int] | None,
        save_frames: bool,
        beep: bool,
        output_dir: Path,
        speaker: SpeechAnnouncer,
    ):
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self.class_filter = class_filter
        self.save_frames = save_frames
        self.beep = beep
        self.output_dir = output_dir
        self.speaker = speaker
        self.banner_until = 0.0
        self.last_message = ""
        self.last_trigger_at: dict[str, float] = {}

    def trigger(
        self,
        frame,
        message: str,
        trigger_key: str,
        cooldown_seconds: float | None = None,
    ) -> bool:
        cooldown = self.cooldown_seconds if cooldown_seconds is None else cooldown_seconds
        now = time.time()
        if now - self.last_trigger_at.get(trigger_key, 0.0) < cooldown:
            self.banner_until = max(self.banner_until, now + 2.0)
            return False

        self.last_trigger_at[trigger_key] = now
        self.last_message = message
        self.banner_until = now + 3.0
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

        if self.save_frames:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = trigger_key.replace(":", "_").replace(" ", "_")
            cv2.imwrite(str(self.output_dir / f"{safe_name}_{timestamp}.jpg"), frame)

        if self.beep:
            self._beep()

        return True

    def evaluate(self, frame, active_tracks: list[dict], enabled: bool) -> None:
        if not enabled or self.threshold <= 0:
            return

        monitored_tracks = [
            track
            for track in active_tracks
            if self.class_filter is None or track["class_id"] in self.class_filter
        ]
        monitored_count = len(monitored_tracks)
        if monitored_count < self.threshold:
            return

        label_counts = Counter(track["class_name"] for track in monitored_tracks)
        summary = ", ".join(f"{name}: {count}" for name, count in label_counts.most_common(3))
        message = f"Alert {monitored_count} monitored objects detected"
        if summary:
            message += f" ({summary})"
        self.trigger(frame, message, trigger_key="threshold-alert")

    def clear(self) -> None:
        self.banner_until = 0.0
        self.last_message = ""

    def _beep(self) -> None:
        try:
            if os.name == "nt":
                import winsound

                winsound.Beep(2200, 180)
            else:
                print("\a", end="", flush=True)
        except Exception:
            pass

    def draw(self, frame) -> None:
        if time.time() > self.banner_until or not self.last_message:
            return

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 52), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        cv2.putText(frame, self.last_message, (16, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.73, (255, 255, 255), 2)


class PersistentDataLogger:
    """Save track-based intelligence and suspicious event snapshots securely."""

    def __init__(self, log_dir: str = "suspicious_logs"):
        self.log_dir = Path(log_dir)
        self.snapshots_dir = self.log_dir / "snapshots"
        self.persons_file = self.log_dir / "persons.json"
        self.events_file = self.log_dir / "suspicious_events.csv"
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        
        self.persons_db = {}
        if self.persons_file.exists():
            try:
                with open(self.persons_file, "r") as f:
                    self.persons_db = json.load(f)
            except Exception:
                pass
                
        if not self.events_file.exists():
            try:
                with open(self.events_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Timestamp", "Track ID", "Reason", "Snapshot File"])
            except Exception:
                pass

    def update_persons(self, records: list[dict], enabled: bool = True):
        if not enabled or not records:
            return
        updated = False
        for record in records:
            tid = str(record["track_id"])
            if tid not in self.persons_db or self.persons_db[tid] != record:
                self.persons_db[tid] = record
                updated = True
        if updated:
            try:
                with open(self.persons_file, "w") as f:
                    json.dump(self.persons_db, f, indent=2)
            except Exception:
                pass

    def log_suspicious_event(self, frame, track: dict, reason: str, enabled: bool = True):
        if not enabled:
            return None
            
        track_id = track["track_id"]
        safe_reason = reason.replace(" ", "_").replace(":", "").lower()
        safe_reason = "".join(x for x in safe_reason if x.isalnum() or x == "_")
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"suspicious_track_{track_id}_{safe_reason}_{timestamp_str}.jpg"
        filepath = self.snapshots_dir / filename
        
        x1, y1, x2, y2 = track["box"]
        pad_x = max(8, int((x2 - x1) * 0.12))
        pad_y = max(8, int((y2 - y1) * 0.12))
        crop = frame[max(0, y1 - pad_y):min(frame.shape[0], y2 + pad_y), max(0, x1 - pad_x):min(frame.shape[1], x2 + pad_x)]
        
        filename_saved = "No Image"
        if crop.size > 0:
            try:
                cv2.imwrite(str(filepath), crop)
                filename_saved = filename
            except Exception:
                pass

        try:
            with open(self.events_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), track_id, reason, filename_saved])
        except Exception:
            pass

        return str(filepath) if filename_saved != "No Image" else None


class EventLogger:
    """Keep demo-friendly structured logs in memory for future UI use."""

    def __init__(self, max_entries: int = 200):
        self.entries = deque(maxlen=max_entries)

    def add(self, event: str, track_id: int | None, details: str):
        entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "event": event,
            "track_id": track_id,
            "details": details,
        }
        self.entries.append(entry)
        print(f"[LOG {entry['time']}] {event} track={track_id}: {details}")
        return entry

    def as_list(self, limit: int | None = None) -> list[dict]:
        items = list(self.entries)
        return items if limit is None else items[-limit:]


class DemoModeManager:
    """Keep the demo visually active even when the scene is quiet."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.last_simulated_at = 0.0
        self.last_overlay_message = "Demo mode active"

    def maybe_simulate_alert(self, now: float, active_tracks: list[dict]):
        if not self.enabled or active_tracks:
            return None
        if now - self.last_simulated_at < 12.0:
            return None

        self.last_simulated_at = now
        self.last_overlay_message = "Demo mode simulated alert"
        return {
            "event": "demo_alert",
            "track_id": None,
            "reason": "Demo mode simulated alert due to quiet scene",
            "confidence": 0.66,
            "priority": "MEDIUM",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "dwell_time": 0.0,
        }


def extract_detection_summary(result, names) -> list[dict]:
    summaries = []
    if result.boxes is None or len(result.boxes) == 0:
        return summaries

    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy().astype(int)
    for box, conf, cls_id in zip(xyxy, confs, clss):
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        summaries.append(
            {
                "class_id": int(cls_id),
                "class_name": get_class_name(names, cls_id),
                "confidence": round(float(conf), 3),
                "box": [x1, y1, x2, y2],
            }
        )
    return summaries


def score_dwell_time(dwell_time: float) -> tuple[float, str]:
    if dwell_time >= 15.0:
        return min(0.98, 0.65 + min(dwell_time, 30.0) / 60.0), "HIGH"
    if dwell_time >= 10.0:
        return min(0.9, 0.45 + min(dwell_time - 10.0, 10.0) / 22.0), "MEDIUM"
    return min(0.74, 0.25 + min(dwell_time, 10.0) / 20.0), "LOW"


def build_reasoning(event: str, track_id: int | None, reason: str, confidence: float, priority: str, dwell_time: float = 0.0):
    return {
        "event": event,
        "track_id": track_id,
        "reason": reason,
        "confidence": round(float(confidence), 2),
        "priority": priority,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "dwell_time": round(float(dwell_time), 1),
    }


def build_suspicious_reasoning(track: dict, track_info: dict):
    confidence, priority = score_dwell_time(track_info.get("dwell_time", 0.0))
    dwell_time = track_info.get("dwell_time", 0.0)
    reason = f"Person idle for {dwell_time:.1f} seconds"
    return build_reasoning("suspicious_activity", track["track_id"], reason, confidence, priority, dwell_time=dwell_time)


def build_intent_reasoning(track: dict, intent_info: dict):
    intent = intent_info.get("intent", "normal movement")
    if intent == "loitering":
        reason = (
            f"Track {track['track_id']} loitering for {intent_info.get('time_in_frame', 0.0):.1f}s "
            f"near zone ({intent_info.get('time_in_zone', 0.0):.1f}s in zone)"
        )
        priority = "MEDIUM"
    else:
        reason = (
            f"Track {track['track_id']} approaching restricted zone "
            f"with consistency {intent_info.get('direction_consistency', 0.0):.2f}"
        )
        priority = "HIGH"
    confidence = max(0.5, min(0.96, intent_info.get("risk_score", 0.5)))
    return build_reasoning("intent_alert", track["track_id"], reason, confidence, priority, dwell_time=intent_info.get("time_in_frame", 0.0))


def build_alert_reasoning(active_tracks: list[dict], monitored_class_ids: set[int] | None):
    monitored_tracks = [
        track
        for track in active_tracks
        if monitored_class_ids is None or track["class_id"] in monitored_class_ids
    ]
    if not monitored_tracks:
        return None

    priority = "HIGH" if len(monitored_tracks) >= 4 else "MEDIUM"
    confidence = min(0.96, 0.45 + len(monitored_tracks) * 0.12)
    reason = f"{len(monitored_tracks)} monitored objects detected"
    return build_reasoning("alert", monitored_tracks[0]["track_id"], reason, confidence, priority)


def prioritize_target(active_tracks: list[dict], track_analysis: dict[int, dict]):
    if not active_tracks:
        return None

    def ranking(track: dict):
        analysis = track_analysis.get(track["track_id"], {})
        suspicious_rank = 1 if track.get("suspicious") else 0
        intent_rank = 2 if track.get("intent") == "approaching restricted zone" else 1 if track.get("intent") == "loitering" else 0
        risk_rank = analysis.get("risk_score", track.get("risk_score", 0.0))
        dwell_rank = analysis.get("dwell_time", track.get("dwell_time", 0.0))
        return suspicious_rank, intent_rank, risk_rank, dwell_rank, track["area"]

    return max(active_tracks, key=ranking)


def remember_reasoning(reasoning_store: deque, reasoning: dict | None):
    if reasoning is None:
        return None
    reasoning_store.append(reasoning)
    return reasoning


def draw_reasoning_panel(frame, reasoning: dict | None, primary_target_id: int | None) -> None:
    if reasoning is None:
        return

    overlay = frame.copy()
    panel_y1 = max(250, frame.shape[0] - 150)
    panel_y2 = min(frame.shape[0] - 10, panel_y1 + 140)
    cv2.rectangle(overlay, (12, panel_y1), (430, panel_y2), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.42, frame, 0.58, 0, frame)

    lines = [
        "AI Reason",
        f"Track #{reasoning['track_id']} | {reasoning['event']}",
        f"Dwell {reasoning.get('dwell_time', 0.0):.1f}s | Priority {reasoning['priority']}",
        f"Confidence {reasoning['confidence']:.2f} | Time {reasoning['timestamp']}",
    ]
    if primary_target_id is not None:
        lines.append(f"Primary Target ID: {primary_target_id}")

    y = panel_y1 + 24
    for index, line in enumerate(lines):
        scale = 0.64 if index == 0 else 0.52
        color = (0, 220, 255) if index == 0 else (255, 255, 255)
        cv2.putText(frame, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
        y += 20

    cv2.putText(
        frame,
        reasoning["reason"],
        (24, min(panel_y2 - 12, y + 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )


def build_system_state(
    detections: list[dict],
    tracks: list[dict],
    alerts: list[dict],
    primary_target: dict | None,
    reasoning: dict | None,
    logs: list[dict],
    status: dict | None = None,
    person_intelligence: dict | None = None,
):
    normalized_tracks = []
    for track in tracks:
        normalized = dict(track)
        normalized["box"] = list(track["box"])
        normalized["center"] = [round(track["center"][0], 1), round(track["center"][1], 1)]
        normalized_tracks.append(normalized)

    normalized_people = {"enabled": False, "available": False, "records": []}
    if person_intelligence is not None:
        normalized_people = {
            "enabled": bool(person_intelligence.get("enabled")),
            "available": bool(person_intelligence.get("available")),
            "error": person_intelligence.get("error"),
            "records": [],
        }
        for record in person_intelligence.get("records", []):
            cleaned = {
                key: value
                for key, value in record.items()
                if key not in {"first_seen_epoch", "last_seen_epoch", "last_analyzed_frame", "analysis_error"}
            }
            normalized_people["records"].append(cleaned)

    return {
        "detections": detections,
        "tracks": normalized_tracks,
        "alerts": alerts,
        "primary_target": None if primary_target is None else primary_target["track_id"],
        "reasoning": reasoning,
        "logs": logs,
        "status": status or {},
        "person_intelligence": normalized_people,
    }


def get_system_state() -> dict:
    return LAST_SYSTEM_STATE.copy()


def build_dashboard_analytics(analytics_store: dict, counter: LineCounter, system_state: dict) -> dict:
    now = time.time()
    analytics_store["active_timeline"] = deque(
        [(ts, count) for ts, count in analytics_store["active_timeline"] if now - ts <= 60.0],
        maxlen=240,
    )
    average_active = 0.0
    peak_active = 0
    if analytics_store["active_timeline"]:
        counts = [count for _, count in analytics_store["active_timeline"]]
        average_active = sum(counts) / len(counts)
        peak_active = max(counts)

    class_bars = counter.class_counts.most_common(6)
    tracking_bars = [
        ("Active", counter.active_total),
        ("Unique", counter.unique_total),
        ("Alerts", analytics_store["alerts_triggered"]),
    ]
    intent_totals = Counter(track.get("intent", "normal movement") for track in system_state["tracks"])
    return {
        "class_bars": class_bars if class_bars else [("none", 0)],
        "tracking_bars": tracking_bars,
        "alerts_triggered": analytics_store["alerts_triggered"],
        "average_active": average_active,
        "peak_active": peak_active,
        "active_now": counter.active_total,
        "unique_tracks": counter.unique_total,
        "top_class": class_bars[0][0] if class_bars else "none",
        "intent_bars": intent_totals.most_common(3) if intent_totals else [("normal", 0)],
        "intelligence_count": len(system_state.get("person_intelligence", {}).get("records", [])),
    }


class SmartFocusManager:
    """Highlight the most important target without cluttering the live feed."""

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def choose_target(
        self,
        active_tracks: list[dict],
        suspicious_ids: set[int],
        monitored_class_ids: set[int] | None,
    ) -> dict | None:
        if not self.enabled or not active_tracks:
            return None

        suspicious_tracks = [
            track
            for track in active_tracks
            if track["track_id"] in suspicious_ids or track.get("intent") in {"loitering", "approaching restricted zone"}
        ]
        if suspicious_tracks:
            return max(suspicious_tracks, key=lambda track: (track.get("risk_score", 0.0), track["area"]))

        if monitored_class_ids is not None:
            monitored_tracks = [track for track in active_tracks if track["class_id"] in monitored_class_ids]
            if monitored_tracks:
                return max(monitored_tracks, key=lambda track: (track.get("risk_score", 0.0), track["area"]))

        return max(active_tracks, key=lambda track: (track.get("risk_score", 0.0), track["area"]))

    def draw(self, frame, target: dict | None, suspicious_ids: set[int]) -> None:
        if not self.enabled or target is None:
            return

        x1, y1, x2, y2 = target["box"]
        border_color = COLOR_TEAL if target["track_id"] not in suspicious_ids else COLOR_AMBER
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 4)


def draw_label(frame, text: str, top_left: tuple[int, int], color: tuple[int, int, int]) -> None:
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    x, y = top_left
    cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width + 10, y), color, -1)
    cv2.putText(frame, text, (x + 5, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def draw_rounded_card(frame, rect, fill_color, border_color=None, radius: int = 18, shadow: bool = True) -> None:
    x1, y1, x2, y2 = rect
    if shadow:
        shadow_color = (12, 16, 22)
        cv2.rectangle(frame, (x1 + 6, y1 + 6), (x2 + 6, y2 + 6), shadow_color, -1)
    cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), fill_color, -1)
    cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), fill_color, -1)
    cv2.circle(frame, (x1 + radius, y1 + radius), radius, fill_color, -1)
    cv2.circle(frame, (x2 - radius, y1 + radius), radius, fill_color, -1)
    cv2.circle(frame, (x1 + radius, y2 - radius), radius, fill_color, -1)
    cv2.circle(frame, (x2 - radius, y2 - radius), radius, fill_color, -1)
    if border_color is not None:
        cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), border_color, 1)
        cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), border_color, 1)
        cv2.circle(frame, (x1 + radius, y1 + radius), radius, border_color, 1)
        cv2.circle(frame, (x2 - radius, y1 + radius), radius, border_color, 1)
        cv2.circle(frame, (x1 + radius, y2 - radius), radius, border_color, 1)
        cv2.circle(frame, (x2 - radius, y2 - radius), radius, border_color, 1)


def fit_frame_to_rect(frame, rect):
    x1, y1, x2, y2 = rect
    target_w = max(1, x2 - x1)
    target_h = max(1, y2 - y1)
    src_h, src_w = frame.shape[:2]
    scale = min(target_w / src_w, target_h / src_h)
    resized_w = max(1, int(src_w * scale))
    resized_h = max(1, int(src_h * scale))
    resized = cv2.resize(frame, (resized_w, resized_h))
    canvas = np.full((target_h, target_w, 3), COLOR_BG, dtype=np.uint8)
    offset_x = (target_w - resized_w) // 2
    offset_y = (target_h - resized_h) // 2
    canvas[offset_y:offset_y + resized_h, offset_x:offset_x + resized_w] = resized
    return canvas


def clip_text_to_width(text: str, max_width: int, scale: float, thickness: int) -> str:
    if max_width <= 12:
        return ""
    if cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] <= max_width:
        return text

    ellipsis = "..."
    clipped = text
    while clipped:
        clipped = clipped[:-1]
        candidate = f"{clipped}{ellipsis}"
        if cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] <= max_width:
            return candidate
    return ellipsis


def draw_metric_card(frame, rect, title: str, value: str, accent, subtitle: str | None = None) -> None:
    draw_rounded_card(frame, rect, COLOR_PANEL, border_color=COLOR_BORDER)
    x1, y1, x2, y2 = rect
    text_width = max(32, x2 - x1 - 46)
    cv2.rectangle(frame, (x1 + 16, y1 + 18), (x1 + 24, y1 + 60), accent, -1)
    cv2.putText(frame, clip_text_to_width(title, text_width, 0.5, 1), (x1 + 38, y1 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_MUTED, 1)
    cv2.putText(frame, clip_text_to_width(value, text_width, 0.86, 2), (x1 + 38, y1 + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.86, COLOR_TEXT, 2)
    if subtitle:
        cv2.putText(frame, clip_text_to_width(subtitle, text_width, 0.44, 1), (x1 + 38, y1 + 92), cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_MUTED, 1)


def draw_panel_header(frame, rect, title: str, subtitle: str | None = None, accent=COLOR_TEAL) -> None:
    x1, y1, x2, _ = rect
    cv2.putText(frame, title, (x1 + 18, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, COLOR_TEXT, 2)
    cv2.rectangle(frame, (x1 + 18, y1 + 38), (x1 + 78, y1 + 42), accent, -1)
    if subtitle:
        width = max(40, x2 - x1 - 36)
        cv2.putText(frame, clip_text_to_width(subtitle, width, 0.46, 1), (x1 + 18, y1 + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_MUTED, 1)


def draw_glow_indicator(frame, center: tuple[int, int], color, label: str) -> None:
    for radius, alpha in ((18, 0.12), (12, 0.2)):
        overlay = frame.copy()
        cv2.circle(overlay, center, radius, color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    cv2.circle(frame, center, 6, color, -1)
    cv2.putText(frame, label, (center[0] + 16, center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_TEXT, 1)


def draw_key_value_rows(frame, rect, rows: list[tuple[str, str]], accent=COLOR_VIOLET) -> None:
    draw_rounded_card(frame, rect, COLOR_PANEL, border_color=COLOR_BORDER)
    x1, y1, x2, _ = rect
    draw_panel_header(frame, rect, "System Status", None, accent=accent)
    y = y1 + 86
    row_width = max(40, x2 - x1 - 36)
    for label, value in rows:
        cv2.putText(frame, clip_text_to_width(label, row_width // 2, 0.46, 1), (x1 + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_MUTED, 1)
        value_text = clip_text_to_width(value, row_width // 2, 0.48, 1)
        text_size = cv2.getTextSize(value_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
        cv2.putText(frame, value_text, (x2 - 18 - text_size, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_TEXT, 1)
        y += 28


def draw_log_panel(frame, rect, title: str, items: list[str], offset: int = 0, accent=COLOR_AMBER) -> None:
    draw_rounded_card(frame, rect, COLOR_PANEL, border_color=COLOR_BORDER)
    draw_panel_header(frame, rect, title, "Latest events with lightweight scroll", accent=accent)
    x1, y1, x2, y2 = rect
    start_y = y1 + 90
    row_height = 28
    visible_rows = max(1, (y2 - start_y - 12) // row_height)
    max_offset = max(0, len(items) - visible_rows)
    offset = max(0, min(offset, max_offset))
    window = items[offset:offset + visible_rows]
    if not window:
        cv2.putText(frame, "No alert activity in the current session.", (x1 + 18, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_MUTED, 1)
        return

    width = max(40, x2 - x1 - 36)
    y = start_y
    for item in window:
        cv2.putText(frame, clip_text_to_width(item, width, 0.46, 1), (x1 + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_TEXT, 1)
        y += row_height

    if max_offset > 0:
        footer = f"Scroll {offset + 1}-{offset + len(window)} of {len(items)}"
        cv2.putText(frame, footer, (x1 + 18, y2 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_MUTED, 1)


def draw_tab_bar(frame, active_tab: int) -> list[tuple[int, int, int, int]]:
    tab_titles = ["Live Monitoring", "Analytics", "Person Intelligence", "Use Cases"]
    tab_rects = []
    x = 24
    y = 18
    tab_widths = [208, 144, 214, 148]
    for index, (title, width) in enumerate(zip(tab_titles, tab_widths)):
        rect = (x, y, x + width, y + 44)
        active = index == active_tab
        fill = COLOR_PANEL if active else COLOR_PANEL_ALT
        accent = COLOR_TEAL if index == 0 else COLOR_VIOLET if index == 1 else COLOR_AMBER if index == 2 else COLOR_VIOLET
        draw_rounded_card(frame, rect, fill, border_color=COLOR_BORDER, radius=14, shadow=False)
        if active:
            cv2.rectangle(frame, (rect[0] + 12, rect[3] - 8), (rect[2] - 12, rect[3] - 4), accent, -1)
        cv2.putText(frame, title, (rect[0] + 18, rect[1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, COLOR_TEXT, 2)
        tab_rects.append(rect)
        x += width + 12
    return tab_rects


def render_live_tab(frame, video_frame, system_state: dict, fps: float, runtime: dict, detection_enabled: bool, alerts_enabled: bool, latest_reasoning: dict | None) -> None:
    h, w = frame.shape[:2]
    video_rect = (24, 98, int(w * 0.69), h - 24)
    side_rect = (int(w * 0.69) + 16, 98, w - 24, h - 24)
    draw_rounded_card(frame, video_rect, COLOR_PANEL, border_color=COLOR_BORDER)
    draw_rounded_card(frame, side_rect, COLOR_PANEL, border_color=COLOR_BORDER)

    x1, y1, x2, y2 = video_rect
    draw_panel_header(frame, video_rect, "Live Monitoring", "Clean video with only track visuals and focus highlight", accent=COLOR_TEAL)
    video_inner = (x1 + 16, y1 + 76, x2 - 16, y2 - 16)
    fitted = fit_frame_to_rect(video_frame, video_inner)
    frame[video_inner[1]:video_inner[1] + fitted.shape[0], video_inner[0]:video_inner[0] + fitted.shape[1]] = fitted

    status = system_state.get("status", {})
    primary_track = None
    if system_state["primary_target"] is not None:
        primary_track = next((track for track in system_state["tracks"] if track["track_id"] == system_state["primary_target"]), None)
    primary_intent = primary_track.get("intent", "normal movement") if primary_track else "none"
    audio_value = "MUTED" if status.get("audio_muted") else "ON"
    unique_total = status.get("unique_total", len(system_state["tracks"]))
    active_total = status.get("active_total", len(system_state["tracks"]))

    stats = [
        ("FPS", f"{fps:.1f}", COLOR_TEAL, "Real-time performance"),
        ("GPU", "Enabled" if runtime["gpu_enabled"] else "CPU", COLOR_VIOLET, runtime["gpu_name"] if runtime["gpu_enabled"] else "Fallback mode"),
        ("Alerts", "Enabled" if alerts_enabled else "Muted", COLOR_AMBER, "Alert pipeline"),
        ("Active Tracks", str(active_total), COLOR_TEAL, "Objects currently visible"),
    ]

    sx1, sy1, sx2, sy2 = side_rect
    draw_panel_header(frame, side_rect, "Operational Snapshot", "All monitoring metrics stay off the video feed", accent=COLOR_VIOLET)
    inner_w = sx2 - sx1 - 32
    gap = 12
    card_w = (inner_w - gap) // 2
    card_h = 92
    card_positions = []
    cy = sy1 + 76
    for row in range(2):
        left_rect = (sx1 + 16, cy, sx1 + 16 + card_w, cy + card_h)
        right_rect = (sx1 + 16 + card_w + gap, cy, sx1 + 16 + card_w + gap + card_w, cy + card_h)
        card_positions.extend([left_rect, right_rect])
        cy += card_h + gap

    for (title, value, accent, subtitle), rect in zip(stats, card_positions):
        draw_metric_card(frame, rect, title, value, accent, subtitle)

    status_rect = (sx1 + 16, cy, sx2 - 16, cy + 172)
    status_rows = [
        ("System", "Active" if detection_enabled else "Paused"),
        ("Unique Objects", str(unique_total)),
        ("Focus Target", "None" if primary_track is None else f"#{primary_track['track_id']}"),
        ("Intent", primary_intent.title()),
        ("Audio", audio_value),
    ]
    draw_key_value_rows(frame, status_rect, status_rows, accent=COLOR_TEAL)
    draw_glow_indicator(
        frame,
        (status_rect[0] + 26, status_rect[1] + 28),
        COLOR_TEAL if detection_enabled else COLOR_VIOLET,
        "SYSTEM STATUS",
    )

    alert_rect = (sx1 + 16, status_rect[3] + 12, sx2 - 16, sy2 - 16)
    alert_items = []
    for alert in list(reversed(system_state["alerts"][-6:])):
        alert_items.append(f"{alert['timestamp']}  {alert['priority']}  {alert['reason']}")
    if latest_reasoning is not None and not alert_items:
        alert_items.append(f"{latest_reasoning['timestamp']}  {latest_reasoning['reason']}")
    draw_log_panel(frame, alert_rect, "Alert Center", alert_items, 0, accent=COLOR_AMBER)


def draw_bar_chart(frame, rect, data: list[tuple[str, int]], accent) -> None:
    draw_rounded_card(frame, rect, COLOR_PANEL, border_color=COLOR_BORDER)
    x1, y1, x2, y2 = rect
    max_value = max((value for _, value in data), default=1)
    inner_w = max(40, x2 - x1 - 48)
    bar_area_h = max(40, y2 - y1 - 86)
    bar_w = max(20, inner_w // max(1, len(data)))
    for index, (label, value) in enumerate(data):
        bx1 = x1 + 24 + index * bar_w
        bx2 = min(x2 - 24, bx1 + max(12, bar_w - 12))
        bh = int((value / max_value) * max(12, bar_area_h - 10)) if max_value > 0 else 0
        by2 = y2 - 28
        by1 = by2 - bh
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), accent, -1)
        label_text = clip_text_to_width(label, max(12, bx2 - bx1), 0.38, 1)
        cv2.putText(frame, label_text, (bx1, y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_MUTED, 1)
        value_text = clip_text_to_width(str(value), max(18, bx2 - bx1 + 18), 0.4, 1)
        cv2.putText(frame, value_text, (bx1, max(y1 + 52, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT, 1)


def render_analytics_tab(frame, analytics: dict, system_state: dict) -> None:
    h, w = frame.shape[:2]
    card_top = 98
    metric_cards = [
        ("Tracked IDs", str(analytics["unique_tracks"]), COLOR_TEAL),
        ("Active Now", str(analytics["active_now"]), COLOR_VIOLET),
        ("Peak Active", str(analytics["peak_active"]), COLOR_AMBER),
        ("Top Class", analytics["top_class"], COLOR_TEAL),
    ]
    x = 24
    for title, value, accent in metric_cards:
        draw_metric_card(frame, (x, card_top, x + 310, card_top + 92), title, value, accent)
        x += 326

    secondary_cards = [
        ("Alerts Triggered", str(analytics["alerts_triggered"]), COLOR_AMBER),
        ("Avg Active", f"{analytics['average_active']:.1f}", COLOR_VIOLET),
    ]
    x = 24
    for title, value, accent in secondary_cards:
        draw_metric_card(frame, (x, card_top + 106, x + 310, card_top + 198), title, value, accent)
        x += 326

    class_chart_rect = (24, 318, int(w * 0.43), h - 24)
    track_chart_rect = (int(w * 0.43) + 16, 318, int(w * 0.72), h - 24)
    alerts_rect = (int(w * 0.72) + 16, 318, w - 24, h - 24)

    draw_bar_chart(frame, class_chart_rect, analytics["class_bars"], COLOR_TEAL)
    draw_panel_header(frame, class_chart_rect, "Object Distribution", "Unique counts by stable class ID", accent=COLOR_TEAL)

    draw_bar_chart(frame, track_chart_rect, analytics["tracking_bars"], COLOR_VIOLET)
    draw_panel_header(frame, track_chart_rect, "Tracking Stats", "Active vs unique tracks stay stable over time", accent=COLOR_VIOLET)

    alert_logs = []
    for log in reversed(system_state.get("logs", [])):
        if log.get("event") not in {"alert", "intent_alert", "suspicious_activity"}:
            continue
        alert_logs.append(f"{log['time']}  {log['event']}  {log['details']}")
    draw_log_panel(frame, alerts_rect, "Alert Logs", alert_logs, UI_STATE["analytics_alert_scroll"], accent=COLOR_AMBER)


def render_person_intelligence_tab(frame, system_state: dict) -> None:
    h, w = frame.shape[:2]
    summary_rect = (24, 98, w - 24, 216)
    table_rect = (24, 332, w - 24, h - 24)
    draw_rounded_card(frame, summary_rect, COLOR_PANEL, border_color=COLOR_BORDER)
    draw_rounded_card(frame, table_rect, COLOR_PANEL, border_color=COLOR_BORDER)

    intelligence = system_state.get("person_intelligence", {})
    records = intelligence.get("records", [])
    active_records = [record for record in records if record.get("active")]
    idle_records = [record for record in active_records if record.get("movement_state") == "idle"]

    cards = [
        ("Tracked Persons", str(len(records)), COLOR_TEAL, "Current session person records"),
        ("Active Persons", str(len(active_records)), COLOR_VIOLET, "Currently visible tracked people"),
        ("Idle Persons", str(len(idle_records)), COLOR_AMBER, "People currently marked idle"),
        ("DeepFace", "Ready" if intelligence.get("enabled") else "Offline", COLOR_VIOLET, "Cached asynchronous enrichment"),
    ]
    draw_panel_header(frame, summary_rect, "Person Intelligence", "Human-centric attributes stay cached per stable person track", accent=COLOR_VIOLET)
    x = 42
    for title, value, accent, subtitle in cards:
        draw_metric_card(frame, (x, 190, x + 300, 282), title, value, accent, subtitle)
        x += 320

    headers = ["ID", "Age", "Gender", "Emotion", "Duration", "Status", "Entry Time", "Exit Time"]
    column_x = [44, 118, 210, 344, 520, 658, 824, 1012]
    column_widths = [56, 70, 110, 150, 112, 130, 150, 150]
    header_y = 372
    draw_panel_header(frame, table_rect, "Tracked Persons Table", "Each row stays linked to a single DeepSORT person ID", accent=COLOR_AMBER)
    for header, x in zip(headers, column_x):
        cv2.putText(frame, header, (x, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.54, COLOR_MUTED, 1)

    if not records:
        message = "DeepFace backend unavailable." if not intelligence.get("available") else "No person tracks captured yet."
        cv2.putText(frame, message, (42, 418), cv2.FONT_HERSHEY_SIMPLEX, 0.62, COLOR_TEXT, 2)
        error_text = intelligence.get("error")
        if error_text:
            cv2.putText(frame, clip_text_to_width(error_text, w - 84, 0.42, 1), (42, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_MUTED, 1)
        return

    start_y = 406
    row_h = 46
    max_rows = max(1, (table_rect[3] - start_y - 18) // row_h)
    for index, record in enumerate(records[:max_rows]):
        y1 = start_y + index * row_h
        y2 = y1 + 34
        row_rect = (36, y1 - 24, w - 36, y2)
        fill = COLOR_PANEL_ALT if record.get("active") else (30, 26, 24)
        draw_rounded_card(frame, row_rect, fill, border_color=COLOR_BORDER, radius=12, shadow=False)
        values = [
            f"#{record['track_id']}",
            str(record.get("age", "-")),
            str(record.get("gender", "-"))[:10],
            str(record.get("emotion", "-"))[:12],
            f"{record.get('total_duration', 0.0):.1f}s",
            str(record.get("movement_state", "-")).title(),
            str(record.get("entry_time", "-")),
            str(record.get("exit_time") or "Active"),
        ]
        for value, x, width in zip(values, column_x, column_widths):
            cv2.putText(frame, clip_text_to_width(value, width, 0.5, 1), (x, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)


def draw_wrapped_lines(frame, text: str, origin: tuple[int, int], max_width: int, color, scale: float = 0.5, thickness: int = 1) -> int:
    words = text.split()
    x, y = origin
    line = ""
    line_height = 22
    for word in words:
        test_line = f"{line} {word}".strip()
        (w, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if w > max_width and line:
            cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
            y += line_height
            line = word
        else:
            line = test_line
    if line:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
        y += line_height
    return y


def render_use_cases_tab(frame) -> None:
    h, w = frame.shape[:2]
    hero_rect = (24, 98, w - 24, 208)
    draw_rounded_card(frame, hero_rect, COLOR_PANEL, border_color=COLOR_BORDER)
    draw_panel_header(frame, hero_rect, "Use Cases", "Modern surveillance workflows powered by detection, tracking, and person intelligence", accent=COLOR_VIOLET)
    description = "The dashboard is designed for live monitoring environments where stable object counts, explainable alerts, and clean operator visibility matter more than noisy overlays."
    draw_wrapped_lines(frame, description, (42, 170), w - 84, COLOR_MUTED, scale=0.54, thickness=1)

    cards = [
        ("Smart Surveillance", "Monitor live spaces with stable track-based counting, clear alerting, and operator-friendly summaries.", COLOR_TEAL),
        ("Behavior Monitoring", "Surface loitering, idle behavior, and suspicious movement patterns without interrupting the live feed.", COLOR_VIOLET),
        ("Restricted Zones", "Track who approaches sensitive areas and keep focus on the most relevant target at any moment.", COLOR_AMBER),
        ("Anomaly Detection", "Combine alerts, person intelligence, and tracking history to highlight unusual scene activity in real time.", COLOR_TEAL),
    ]
    card_w = (w - 72) // 2
    card_h = 220
    positions = [
        (24, 330, 24 + card_w, 330 + card_h),
        (48 + card_w, 330, w - 24, 330 + card_h),
        (24, 570, 24 + card_w, 570 + card_h),
        (48 + card_w, 570, w - 24, 570 + card_h),
    ]
    for (title, body, accent), rect in zip(cards, positions):
        draw_rounded_card(frame, rect, COLOR_PANEL, border_color=COLOR_BORDER)
        draw_panel_header(frame, rect, title, None, accent=accent)
        draw_wrapped_lines(frame, body, (rect[0] + 18, rect[1] + 80), rect[2] - rect[0] - 36, COLOR_MUTED, scale=0.52, thickness=1)


def apply_tab_transition(current_canvas):
    if UI_STATE["transition_canvas"] is None:
        return current_canvas
    elapsed = time.time() - UI_STATE["transition_started_at"]
    duration = 0.18
    if elapsed >= duration:
        UI_STATE["transition_canvas"] = None
        return current_canvas
    alpha = elapsed / duration
    return cv2.addWeighted(current_canvas, alpha, UI_STATE["transition_canvas"], 1.0 - alpha, 0)


def handle_dashboard_mouse(event, x, y, flags, param) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        for index, rect in enumerate(UI_STATE["tab_rects"]):
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 <= y <= y2 and index != UI_STATE["active_tab"]:
                UI_STATE["transition_from"] = UI_STATE["active_tab"]
                UI_STATE["transition_started_at"] = time.time()
                if UI_STATE["last_dashboard"] is not None:
                    UI_STATE["transition_canvas"] = UI_STATE["last_dashboard"].copy()
                UI_STATE["active_tab"] = index
                return
    if event == cv2.EVENT_MOUSEWHEEL and UI_STATE["active_tab"] == 1:
        delta = -1 if flags < 0 else 1
        UI_STATE["analytics_alert_scroll"] = max(0, UI_STATE["analytics_alert_scroll"] - delta)


def render_dashboard(video_frame, system_state: dict, analytics: dict, fps: float, runtime: dict, detection_enabled: bool, alerts_enabled: bool, latest_reasoning: dict | None):
    dashboard = np.full((820, 1360, 3), COLOR_BG, dtype=np.uint8)
    cv2.putText(dashboard, "AI Surveillance Console", (24, 74), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_TEXT, 2)
    cv2.putText(dashboard, "Professional monitoring surface with clean operator context", (24, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLOR_MUTED, 1)
    UI_STATE["tab_rects"] = draw_tab_bar(dashboard, UI_STATE["active_tab"])
    draw_glow_indicator(
        dashboard,
        (1136, 40),
        COLOR_TEAL if detection_enabled else COLOR_VIOLET,
        "SYSTEM STATUS",
    )
    state_text = "Monitoring" if detection_enabled else "Paused"
    cv2.putText(dashboard, state_text, (1248, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.54, COLOR_TEXT, 1)

    if UI_STATE["active_tab"] == 0:
        render_live_tab(dashboard, video_frame, system_state, fps, runtime, detection_enabled, alerts_enabled, latest_reasoning)
    elif UI_STATE["active_tab"] == 1:
        render_analytics_tab(dashboard, analytics, system_state)
    elif UI_STATE["active_tab"] == 2:
        render_person_intelligence_tab(dashboard, system_state)
    else:
        render_use_cases_tab(dashboard)
    return apply_tab_transition(dashboard)


def draw_sidebar(
    frame,
    fps: float,
    active_counts: Counter,
    counter: LineCounter,
    detection_enabled: bool,
    alerts_enabled: bool,
    voice_status: str,
    suspicious_count: int,
    focus_enabled: bool,
    primary_target_id: int | None,
    demo_mode: bool,
    runtime: dict,
    process_every_n: int,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 58), (360, 298), (18, 22, 28), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    lines = [
        f"FPS: {fps:.1f}",
        f"Detection: {'ON' if detection_enabled else 'OFF'}",
        f"Alerts: {'ON' if alerts_enabled else 'OFF'}",
        f"GPU: {'ON' if runtime['gpu_enabled'] else 'OFF'}",
        f"FP16: {'ON' if runtime['use_half'] else 'OFF'}",
        f"Smart focus: {'ON' if focus_enabled else 'OFF'}",
        f"Active tracks: {sum(active_counts.values())}",
        f"Unique tracked: {len(counter.unique_track_ids)}",
        f"Process stride: {process_every_n}",
        f"Line count: {counter.total_crossings}",
        f"Idle persons: {suspicious_count}",
        f"Primary target: {primary_target_id}",
        f"Demo mode: {'ON' if demo_mode else 'OFF'}",
        voice_status,
    ]

    if active_counts:
        top_active = ", ".join(f"{name}:{count}" for name, count in active_counts.most_common(3))
        lines.append(f"Active by class: {top_active}")

    y = 84
    for line in lines:
        cv2.putText(frame, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2)
        y += 22


def build_color(track_id: int) -> tuple[int, int, int]:
    return (
        (37 * track_id) % 255,
        (17 * track_id + 90) % 255,
        (29 * track_id + 180) % 255,
    )


def render_tracks(frame, tracks, names, counter: LineCounter, track_history, trail_length: int):
    active_tracks = []
    active_counts = Counter()

    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue

        box = track.to_ltrb(orig=True)
        if box is None:
            box = track.to_ltrb()
        if box is None:
            continue

        x1, y1, x2, y2 = clip_box(box, frame.shape)
        if x2 <= x1 or y2 <= y1:
            continue

        track_id = int(track.track_id)
        track_hits = int(getattr(track, "hits", 0))
        class_id = track.get_det_class()
        class_name = get_class_name(names, class_id)
        confidence = track.get_det_conf()
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        color = (88, 98, 112)
        area = max(1, (x2 - x1) * (y2 - y1))

        track_history[track_id].append((int(center[0]), int(center[1])))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        counter.register(track_id, class_name, center, frame.shape, track_hits=track_hits)
        stable_class_name = counter.track_classes.get(track_id, class_name)
        label = f"#{track_id} {stable_class_name}"
        if confidence is not None:
            label += f" {confidence:.2f}"
        draw_label(frame, label, (x1, max(20, y1)), color)
        active_counts[stable_class_name] += 1
        active_tracks.append(
            {
                "track_id": track_id,
                "class_id": int(class_id) if class_id is not None else None,
                "class_name": stable_class_name,
                "box": (x1, y1, x2, y2),
                "center": center,
                "area": area,
                "hits": track_hits,
                "confirmed": bool(track.is_confirmed()),
                "detection_confidence": None if confidence is None else round(float(confidence), 3),
            }
        )

    counter.update_active_counts(active_tracks)
    return active_tracks, active_counts


def highlight_special_tracks(frame, active_tracks: list[dict], suspicious_ids: set[int], primary_target_id: int | None) -> None:
    for track in active_tracks:
        x1, y1, x2, y2 = track["box"]
        if track["track_id"] == primary_target_id:
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_TEAL, 3)
            continue
        if track["track_id"] in suspicious_ids or track.get("intent") in {"loitering", "approaching restricted zone"}:
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_AMBER, 3)


def handle_control_commands(
    commands: list[str],
    detection_enabled: bool,
    alerts_enabled: bool,
    speaker: SpeechAnnouncer,
    alert_manager: AlertManager,
    audio_controller: AudioAlertController,
):
    for command in commands:
        if command == "start detection" and not detection_enabled:
            detection_enabled = True
            print("Control command: detection started.")
            speaker.say("Detection started", key="command:start", cooldown_seconds=1.0)
        elif command == "stop detection" and detection_enabled:
            detection_enabled = False
            print("Control command: detection stopped.")
            speaker.say("Detection stopped", key="command:stop", cooldown_seconds=1.0)
        elif command in {"enable alerts", "toggle alerts"} and not alerts_enabled:
            alerts_enabled = True
            print("Control command: alerts enabled.")
            speaker.say("Alerts enabled", key="command:alerts-on", cooldown_seconds=1.0)
        elif command in {"disable alerts", "toggle alerts"} and alerts_enabled:
            alerts_enabled = False
            alert_manager.clear()
            print("Control command: alerts disabled.")
            speaker.say("Alerts disabled", key="command:alerts-off", cooldown_seconds=1.0)
        elif command == "toggle mute":
            muted = audio_controller.toggle_mute()
            state = "muted" if muted else "enabled"
            print(f"Control command: audio alerts {state}.")

    return detection_enabled, alerts_enabled


def command_from_key(key_code: int) -> str | None:
    if key_code in (ord("s"), ord("S")):
        return "start detection"
    if key_code in (ord("x"), ord("X")):
        return "stop detection"
    if key_code in (ord("a"), ord("A")):
        return "toggle alerts"
    if key_code in (ord("m"), ord("M")):
        return "toggle mute"
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time YOLO + DeepSORT detection with command triggers, audio alerts, intent analysis, smart focus, and person intelligence."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index, video file path, or stream URL. Defaults to webcam index 0.",
    )
    parser.add_argument(
        "--model",
        default=str(choose_default_model_path()),
        help="Path to YOLO model weights.",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="Detection NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--camera-width", type=int, default=640, help="Camera width for webcam capture.")
    parser.add_argument("--camera-height", type=int, default=480, help="Camera height for webcam capture.")
    parser.add_argument("--target-fps", type=float, default=30.0, help="Soft loop FPS cap to reduce CPU load.")
    parser.add_argument(
        "--process-every-n",
        type=int,
        default=1,
        help="Run the full detection pipeline every N frames. Use 1 for max responsiveness.",
    )
    parser.add_argument(
        "--classes",
        default=None,
        help="Optional comma-separated class ids or names to track, e.g. 'person,car' or '0,2'.",
    )
    parser.add_argument("--max-age", type=int, default=30, help="DeepSORT max age before dropping a track.")
    parser.add_argument("--n-init", type=int, default=2, help="DeepSORT hits required before confirming a track.")
    parser.add_argument(
        "--max-cosine-distance",
        type=float,
        default=0.3,
        help="DeepSORT appearance matching threshold.",
    )
    parser.add_argument(
        "--count-axis",
        choices=("horizontal", "vertical"),
        default="horizontal",
        help="Orientation of the virtual counting line.",
    )
    parser.add_argument(
        "--count-position",
        type=float,
        default=0.55,
        help="Relative line position from 0.0 to 1.0 along the chosen axis.",
    )
    parser.add_argument(
        "--count-direction",
        choices=("any", "down", "up", "left", "right"),
        default="any",
        help="Only count tracks moving in this direction through the line.",
    )
    parser.add_argument(
        "--alert-threshold",
        type=int,
        default=3,
        help="Trigger an alert when this many monitored objects are simultaneously visible. Use 0 to disable.",
    )
    parser.add_argument(
        "--alert-classes",
        default=None,
        help="Optional comma-separated class ids or names to monitor for alerts. Defaults to tracked classes.",
    )
    parser.add_argument(
        "--alert-cooldown",
        type=float,
        default=10.0,
        help="Minimum seconds between alert triggers.",
    )
    parser.add_argument(
        "--save-alert-frames",
        action="store_true",
        help="Save an annotated frame to runs/alerts whenever an alert triggers.",
    )
    parser.add_argument(
        "--beep-alert",
        action="store_true",
        help="Play a short local beep when an alert triggers.",
    )
    parser.add_argument("--trail-length", type=int, default=20, help="Tracked centroid trail length.")
    parser.add_argument(
        "--idle-seconds",
        type=float,
        default=8.0,
        help="Mark a person as suspicious after this many seconds of minimal movement.",
    )
    parser.add_argument(
        "--idle-movement-threshold",
        type=float,
        default=24.0,
        help="Maximum centroid movement in pixels allowed during idle-person detection.",
    )
    parser.add_argument(
        "--disable-focus",
        action="store_true",
        help="Disable the smart focus inset.",
    )
    parser.add_argument(
        "--disable-voice",
        action="store_true",
        help="Reserved compatibility flag. Command control now uses lightweight local triggers.",
    )
    parser.add_argument(
        "--disable-tts",
        action="store_true",
        help="Disable spoken announcements.",
    )
    parser.add_argument(
        "--demo-mode",
        action="store_true",
        help="Expo-friendly mode with faster suspicious triggers and simulated quiet-scene activity.",
    )
    parser.add_argument(
        "--tts-rate",
        type=int,
        default=185,
        help="Text-to-speech speech rate.",
    )
    args = parser.parse_args()

    if args.count_axis == "horizontal" and args.count_direction in {"left", "right"}:
        parser.error("--count-direction left/right only works with --count-axis vertical.")
    if args.count_axis == "vertical" and args.count_direction in {"up", "down"}:
        parser.error("--count-direction up/down only works with --count-axis horizontal.")
    if args.trail_length < 2:
        parser.error("--trail-length must be at least 2.")
    if args.idle_seconds <= 0:
        parser.error("--idle-seconds must be greater than 0.")
    if args.idle_movement_threshold <= 0:
        parser.error("--idle-movement-threshold must be greater than 0.")
    if args.camera_width <= 0 or args.camera_height <= 0:
        parser.error("--camera-width and --camera-height must be greater than 0.")
    if args.target_fps <= 0:
        parser.error("--target-fps must be greater than 0.")
    if args.process_every_n < 1:
        parser.error("--process-every-n must be at least 1.")

    return args


def main():
    global LAST_SYSTEM_STATE
    args = parse_args()
    runtime = resolve_runtime()

    model = YOLO(args.model)
    try:
        model.to(runtime["device_label"])
        model.fuse()
    except Exception:
        pass
    warmup_model(model, runtime, args.imgsz)
    names = model.names
    person_class_id = resolve_class_id(names, "person")

    try:
        tracked_class_ids = parse_label_filter(args.classes, names)
        alert_class_ids = parse_label_filter(args.alert_classes, names)
    except ValueError as exc:
        print(f"Error: {exc}")
        return
    if alert_class_ids is None:
        alert_class_ids = tracked_class_ids

    source = parse_source(args.source)
    cap = open_capture(source, width=args.camera_width, height=args.camera_height)

    if not cap.isOpened():
        if args.source == "0":
            run_fallback_inference(model)
            return
        print(f"Error: Could not open source {args.source!r}.")
        print("Try a different camera index or pass --source path/to/video.mp4.")
        return

    effective_idle_seconds = min(args.idle_seconds, 4.0) if args.demo_mode else args.idle_seconds
    effective_alert_threshold = min(args.alert_threshold, 1) if args.demo_mode else args.alert_threshold
    target_frame_interval = 1.0 / args.target_fps

    speaker = SpeechAnnouncer(enabled=not args.disable_tts, rate=args.tts_rate)
    audio_controller = AudioAlertController(speaker, enabled=not args.disable_tts)
    command_triggers = CommandTriggerSystem()
    detection_announcer = DetectionEventAnnouncer(speaker, audio_controller=audio_controller)
    event_logger = EventLogger()
    demo_manager = DemoModeManager(enabled=args.demo_mode)
    reasoning_store = deque(maxlen=20)
    recent_alerts = deque(maxlen=20)
    counter = LineCounter(
        axis=args.count_axis,
        position_ratio=args.count_position,
        direction=args.count_direction,
    )
    suspicious_detector = SuspiciousBehaviorDetector(
        person_class_id=person_class_id,
        idle_seconds=effective_idle_seconds,
        movement_threshold=args.idle_movement_threshold,
    )
    intent_predictor = IntentPredictor(person_class_id=person_class_id)
    person_intelligence = PersonIntelligenceManager(person_class_id=person_class_id)
    alert_manager = AlertManager(
        threshold=effective_alert_threshold,
        cooldown_seconds=args.alert_cooldown,
        class_filter=alert_class_ids,
        save_frames=args.save_alert_frames,
        beep=args.beep_alert,
        output_dir=DEFAULT_ALERT_DIR,
        speaker=speaker,
    )
    focus_manager = SmartFocusManager(enabled=not args.disable_focus)
    tracker = build_tracker(args)
    trail_length = max(2, args.trail_length)
    track_history = defaultdict(lambda: deque(maxlen=trail_length))
    detection_enabled = True
    alerts_enabled = True
    smoothed_fps = 0.0
    person_intelligence_state = person_intelligence.snapshot()
    system_state = build_system_state([], [], [], None, None, [], person_intelligence=person_intelligence_state)
    frame_index = 0
    analytics_store = {
        "active_timeline": deque(maxlen=240),
        "alerts_triggered": 0,
    }
    window_name = "Expo AI Surveillance Dashboard"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, handle_dashboard_mouse)

    speaker.say("Surveillance system ready", key="system-ready", cooldown_seconds=1.0)
    event_logger.add(
        "system_ready",
        None,
        f"Demo mode={'ON' if args.demo_mode else 'OFF'} | GPU={'ON' if runtime['gpu_enabled'] else 'OFF'} | Model={Path(args.model).name} | Audio={'OFF' if audio_controller.muted else 'ON'}",
    )
    print("Starting YOLO + DeepSORT... Hotkeys: S=start, X=stop, A=alerts, M=mute, 1/2/3/4=tabs, Q=quit.")

    try:
        while True:
            started_at = time.perf_counter()
            frame_index += 1
            control_commands = command_triggers.poll_commands()
            for command in control_commands:
                event_logger.add("control_command", None, command)
            detection_enabled, alerts_enabled = handle_control_commands(
                control_commands,
                detection_enabled,
                alerts_enabled,
                speaker,
                alert_manager,
                audio_controller,
            )

            success, frame = cap.read()
            if not success:
                print("Video stream ended or failed to grab a frame.")
                break

            video_frame = frame.copy()
            now = time.time()
            detection_summaries = []
            active_tracks = []
            active_counts = Counter()
            suspicious_ids = set()
            track_analysis = {}
            intent_analysis = {}
            person_intelligence_state = person_intelligence.snapshot()
            primary_target = None
            latest_reasoning = reasoning_store[-1] if reasoning_store else None
            should_process_frame = (frame_index % args.process_every_n) == 0

            if detection_enabled and should_process_frame:
                with torch.inference_mode():
                    results = model(
                        frame,
                        verbose=False,
                        conf=args.conf,
                        iou=args.iou,
                        imgsz=args.imgsz,
                        classes=sorted(tracked_class_ids) if tracked_class_ids is not None else None,
                        device=runtime["device"],
                        half=runtime["use_half"],
                    )
                detection_summaries = extract_detection_summary(results[0], names)
                detections = build_detections(results[0])
                tracks = tracker.update_tracks(detections, frame=frame)
                active_tracks, active_counts = render_tracks(
                    video_frame,
                    tracks,
                    names,
                    counter,
                    track_history,
                    trail_length=trail_length,
                )
                suspicious_ids, new_suspicious_tracks, track_analysis = suspicious_detector.update(active_tracks, now)
                intent_analysis, new_intent_tracks = intent_predictor.update(active_tracks, now, frame.shape)
                for track in active_tracks:
                    suspicious_analysis = track_analysis.get(track["track_id"], {})
                    intent_info = intent_analysis.get(track["track_id"], {})
                    dwell_time = suspicious_analysis.get("dwell_time", 0.0)
                    confidence, priority = score_dwell_time(dwell_time)
                    track["dwell_time"] = round(dwell_time, 1)
                    track["priority"] = priority if suspicious_analysis.get("suspicious") else "NORMAL"
                    track["confidence"] = (
                        round(confidence, 2)
                        if suspicious_analysis.get("suspicious")
                        else track.get("detection_confidence", 0.0)
                    )
                    track["suspicious"] = bool(suspicious_analysis.get("suspicious"))
                    track["intent"] = intent_info.get("intent", "normal movement")
                    track["movement_speed"] = intent_info.get("speed", 0.0)
                    track["direction_consistency"] = intent_info.get("direction_consistency", 0.0)
                    track["time_in_zone"] = intent_info.get("time_in_zone", 0.0)
                    track["time_in_frame"] = intent_info.get("time_in_frame", 0.0)
                    track["zone_proximity"] = intent_info.get("zone_proximity", 0.0)
                    track["risk_score"] = intent_info.get("risk_score", 0.0)
                    track["suspicious_intent"] = intent_info.get("suspicious_intent", False)

                person_intelligence_state = person_intelligence.update(active_tracks, frame, now, frame_index)
                intelligence_lookup = {
                    record["track_id"]: record
                    for record in person_intelligence_state.get("records", [])
                }
                for track in active_tracks:
                    record = intelligence_lookup.get(track["track_id"])
                    if record is None:
                        continue
                    track["person_age"] = record.get("age")
                    track["person_gender"] = record.get("gender")
                    track["person_emotion"] = record.get("emotion")
                    track["entry_time"] = record.get("entry_time")
                    track["exit_time"] = record.get("exit_time")
                    track["movement_state"] = record.get("movement_state", "moving")
                    track["person_duration"] = record.get("total_duration", 0.0)

                primary_target = prioritize_target(active_tracks, {**track_analysis, **intent_analysis})
                highlight_special_tracks(
                    video_frame,
                    active_tracks,
                    suspicious_ids,
                    None if primary_target is None else primary_target["track_id"],
                )

                if alerts_enabled:
                    for track in new_suspicious_tracks[:2]:
                        reasoning = build_suspicious_reasoning(track, track_analysis.get(track["track_id"], {}))
                        if alert_manager.trigger(
                            video_frame,
                            reasoning["reason"],
                            trigger_key=f"idle-person-{track['track_id']}",
                            cooldown_seconds=max(effective_idle_seconds, 6.0),
                        ):
                            latest_reasoning = remember_reasoning(reasoning_store, reasoning)
                            recent_alerts.append(reasoning)
                            event_logger.add("suspicious_activity", track["track_id"], reasoning["reason"])
                            audio_controller.trigger(
                                "Warning: suspicious activity detected",
                                alert_key=f"suspicious-audio-{track['track_id']}",
                                cooldown_seconds=max(5.0, effective_idle_seconds),
                                siren=True,
                                speech=True,
                            )

                    for track in new_intent_tracks[:2]:
                        intent_info = intent_analysis.get(track["track_id"], {})
                        if not intent_info.get("suspicious_intent"):
                            continue
                        reasoning = build_intent_reasoning(track, intent_info)
                        if alert_manager.trigger(
                            video_frame,
                            reasoning["reason"],
                            trigger_key=f"intent-{track['track_id']}-{intent_info.get('intent', 'normal')}",
                            cooldown_seconds=6.0,
                        ):
                            latest_reasoning = remember_reasoning(reasoning_store, reasoning)
                            recent_alerts.append(reasoning)
                            event_logger.add("intent_alert", track["track_id"], reasoning["reason"])
                            audio_controller.trigger(
                                "Warning: suspicious activity detected",
                                alert_key=f"intent-audio-{track['track_id']}-{intent_info.get('intent', 'normal')}",
                                cooldown_seconds=6.0,
                                siren=True,
                                speech=True,
                            )

                new_classes = detection_announcer.update(active_counts)
                for class_name in new_classes:
                    event_logger.add("person_detected" if class_name == "person" else "object_detected", None, f"{class_name} detected")

                counter.draw(video_frame)
                monitored_tracks = [
                    track for track in active_tracks if alert_class_ids is None or track["class_id"] in alert_class_ids
                ]
                if alerts_enabled and len(monitored_tracks) >= alert_manager.threshold > 0:
                    reasoning = build_alert_reasoning(active_tracks, alert_class_ids)
                    if reasoning and alert_manager.trigger(
                        video_frame,
                        reasoning["reason"],
                        trigger_key="threshold-alert",
                    ):
                        latest_reasoning = remember_reasoning(reasoning_store, reasoning)
                        recent_alerts.append(reasoning)
                        event_logger.add("alert", reasoning["track_id"], reasoning["reason"])
                        audio_controller.trigger(
                            "Warning: multiple monitored objects detected",
                            alert_key="threshold-audio",
                            cooldown_seconds=5.0,
                            siren=True,
                            speech=True,
                        )

                demo_reasoning = demo_manager.maybe_simulate_alert(now, active_tracks)
                if alerts_enabled and demo_reasoning is not None:
                    if alert_manager.trigger(
                        video_frame,
                        demo_reasoning["reason"],
                        trigger_key="demo-alert",
                        cooldown_seconds=12.0,
                    ):
                        latest_reasoning = remember_reasoning(reasoning_store, demo_reasoning)
                        recent_alerts.append(demo_reasoning)
                        event_logger.add("alert", None, demo_reasoning["reason"])
                        audio_controller.trigger(
                            "Demo alert triggered",
                            alert_key="demo-audio",
                            cooldown_seconds=12.0,
                            siren=False,
                            speech=True,
                        )

                focus_manager.draw(video_frame, primary_target, suspicious_ids)
            elif detection_enabled:
                tracks = tracker.update_tracks([], frame=frame)
                active_tracks, active_counts = render_tracks(
                    video_frame,
                    tracks,
                    names,
                    counter,
                    track_history,
                    trail_length=trail_length,
                )
                intent_analysis, _ = intent_predictor.update(active_tracks, now, frame.shape)
                for track in active_tracks:
                    intent_info = intent_analysis.get(track["track_id"], {})
                    track["intent"] = intent_info.get("intent", "normal movement")
                    track["risk_score"] = intent_info.get("risk_score", 0.0)
                person_intelligence_state = person_intelligence.update(active_tracks, frame, now, frame_index)
                intelligence_lookup = {
                    record["track_id"]: record
                    for record in person_intelligence_state.get("records", [])
                }
                for track in active_tracks:
                    record = intelligence_lookup.get(track["track_id"])
                    if record is not None:
                        track["movement_state"] = record.get("movement_state", "moving")
                        track["person_duration"] = record.get("total_duration", 0.0)
                primary_target = prioritize_target(active_tracks, intent_analysis)
                highlight_special_tracks(
                    video_frame,
                    active_tracks,
                    suspicious_ids,
                    None if primary_target is None else primary_target["track_id"],
                )
                focus_manager.draw(video_frame, primary_target, suspicious_ids)
            else:
                tracker.update_tracks([], frame=frame)
                suspicious_detector.update([], now)
                detection_announcer.reset()
                person_intelligence_state = person_intelligence.update([], frame, now, frame_index)

            elapsed = max(time.perf_counter() - started_at, 1e-6)
            instant_fps = 1.0 / elapsed
            smoothed_fps = instant_fps if smoothed_fps == 0.0 else (smoothed_fps * 0.9 + instant_fps * 0.1)
            analytics_store["active_timeline"].append((now, counter.active_total))
            analytics_store["alerts_triggered"] = sum(1 for log in event_logger.entries if log["event"] == "alert")
            system_state = build_system_state(
                detections=detection_summaries,
                tracks=active_tracks,
                alerts=list(recent_alerts),
                primary_target=primary_target,
                reasoning=latest_reasoning,
                logs=event_logger.as_list(limit=25),
                status={
                    "detection_enabled": detection_enabled,
                    "alerts_enabled": alerts_enabled,
                    "audio_muted": audio_controller.muted,
                    "gpu_enabled": runtime["gpu_enabled"],
                    "command_status": command_triggers.status,
                    "deepface_enabled": person_intelligence.enabled,
                    "active_total": counter.active_total,
                    "unique_total": counter.unique_total,
                    "class_unique_counts": dict(counter.class_counts),
                    "class_active_counts": dict(counter.active_class_counts),
                },
                person_intelligence=person_intelligence_state,
            )
            LAST_SYSTEM_STATE = system_state
            analytics_view = build_dashboard_analytics(analytics_store, counter, system_state)
            dashboard = render_dashboard(
                video_frame,
                system_state,
                analytics_view,
                smoothed_fps,
                runtime,
                detection_enabled,
                alerts_enabled,
                latest_reasoning,
            )
            UI_STATE["last_dashboard"] = dashboard.copy()

            cv2.imshow(window_name, dashboard)
            key_code = cv2.waitKey(1) & 0xFF
            command = command_from_key(key_code)
            if command is not None:
                command_triggers.trigger(command, source="keyboard")
                event_logger.add("keyboard_command", None, command)
            elif key_code in (ord("1"), ord("2"), ord("3"), ord("4")):
                UI_STATE["transition_canvas"] = dashboard.copy()
                UI_STATE["transition_started_at"] = time.time()
                UI_STATE["active_tab"] = int(chr(key_code)) - 1

            if key_code == ord("q"):
                break

            remaining = target_frame_interval - (time.perf_counter() - started_at)
            if remaining > 0:
                time.sleep(min(remaining, 0.01))
    finally:
        person_intelligence.stop()
        audio_controller.stop()
        speaker.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
