import time
from collections import deque, Counter, defaultdict
from pathlib import Path
import cv2
import torch
import numpy as np
import streamlit as st
import pandas as pd
import atexit

def cleanup():
    if "cap" in st.session_state and st.session_state.cap:
        st.session_state.cap.release()

atexit.register(cleanup)

from main import (
    resolve_runtime, warmup_model, resolve_class_id, parse_label_filter,
    build_detections, render_tracks, highlight_special_tracks,
    build_system_state, score_dwell_time, build_suspicious_reasoning,
    build_intent_reasoning, build_alert_reasoning, remember_reasoning,
    build_dashboard_analytics,
    SpeechAnnouncer, AudioAlertController, DetectionEventAnnouncer,
    EventLogger, DemoModeManager, LineCounter, SuspiciousBehaviorDetector,
    IntentPredictor, PersonIntelligenceManager, AlertManager, SmartFocusManager,
    build_tracker, DEFAULT_ALERT_DIR, extract_detection_summary, prioritize_target,
    choose_default_model_path, PersistentDataLogger
)

from ultralytics import YOLO

class Args:
    source = "0"
    model = str(choose_default_model_path())
    conf = 0.35
    iou = 0.45
    imgsz = 640
    camera_width = 640
    camera_height = 480
    target_fps = 30.0
    process_every_n = 1
    classes = None
    max_age = 30
    n_init = 2
    max_cosine_distance = 0.3
    count_axis = "horizontal"
    count_position = 0.55
    count_direction = "any"
    alert_threshold = 3
    alert_classes = None
    alert_cooldown = 10.0
    save_alert_frames = False
    beep_alert = False
    trail_length = 20
    idle_seconds = 10.0
    idle_movement_threshold = 15.0
    disable_focus = False
    disable_voice = True
    disable_tts = False
    demo_mode = False
    tts_rate = 185

def init_system(args, model, runtime):
    names = model.names
    person_class_id = resolve_class_id(names, "person")
    
    tracked_class_ids = parse_label_filter(args.classes, names)
    alert_class_ids = parse_label_filter(args.alert_classes, names)
    if alert_class_ids is None:
        alert_class_ids = tracked_class_ids
        
    speaker = SpeechAnnouncer(enabled=not args.disable_tts, rate=args.tts_rate)
    audio_controller = AudioAlertController(speaker, enabled=not args.disable_tts)
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
        idle_seconds=args.idle_seconds,
        movement_threshold=args.idle_movement_threshold,
    )
    intent_predictor = IntentPredictor(person_class_id=person_class_id)
    person_intelligence = PersonIntelligenceManager(person_class_id=person_class_id)
    alert_manager = AlertManager(
        threshold=args.alert_threshold,
        cooldown_seconds=args.alert_cooldown,
        class_filter=alert_class_ids,
        save_frames=args.save_alert_frames,
        beep=args.beep_alert,
        output_dir=DEFAULT_ALERT_DIR,
        speaker=speaker,
    )
    focus_manager = SmartFocusManager(enabled=not args.disable_focus)
    tracker = build_tracker(args)
    track_history = defaultdict(lambda: deque(maxlen=max(2, args.trail_length)))
    data_logger = PersistentDataLogger()
    
    return {
        "model": model,
        "names": names,
        "runtime": runtime,
        "tracked_class_ids": tracked_class_ids,
        "alert_class_ids": alert_class_ids,
        "speaker": speaker,
        "audio_controller": audio_controller,
        "detection_announcer": detection_announcer,
        "event_logger": event_logger,
        "demo_manager": demo_manager,
        "reasoning_store": reasoning_store,
        "recent_alerts": recent_alerts,
        "counter": counter,
        "suspicious_detector": suspicious_detector,
        "intent_predictor": intent_predictor,
        "person_intelligence": person_intelligence,
        "alert_manager": alert_manager,
        "focus_manager": focus_manager,
        "tracker": tracker,
        "track_history": track_history,
        "data_logger": data_logger,
        "analytics_store": {"active_timeline": deque(maxlen=240), "alerts_triggered": 0},
        "frame_index": 0,
        "unique_persons": set(),
        "suspicious_logs": deque(maxlen=50),
        "saved_suspicious_ids": set(),
        "crowd_alert_last_time": 0.0,
        "danger_alert_last_time": 0.0,
        "alert_cooldowns": defaultdict(float)
    }

def main():
    st.set_page_config(page_title="AI Surveillance Console", layout="wide")
    st.title("AI Surveillance Console")
    st.markdown("Professional monitoring dashboard powered by Streamlit.")
    
    if "args" not in st.session_state:
        st.session_state.args = Args()

    args = st.session_state.args

    if "runtime" not in st.session_state:
        st.session_state.runtime = resolve_runtime()

    if "model" not in st.session_state:
        st.session_state.model = YOLO("yolov8n.pt")
        try:
            st.session_state.model.to(st.session_state.runtime["device_label"])
            st.session_state.model.fuse()
        except Exception:
            pass
        warmup_model(st.session_state.model, st.session_state.runtime, args.imgsz)

    if "core" not in st.session_state:
        st.session_state.core = init_system(args, st.session_state.model, st.session_state.runtime)
        st.session_state.smoothed_fps = 0.0

    if "cap" not in st.session_state:
        st.session_state.cap = None

    if "running" not in st.session_state:
        st.session_state.running = False
        
    if "muted" not in st.session_state:
        st.session_state.muted = False

    core = st.session_state.core
    args = st.session_state.args

    with st.sidebar:
        st.header("Controls")
        
        if st.button("Start Stream"):
            if st.session_state.cap is None:
                new_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not new_cap.isOpened():
                    new_cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
                if not new_cap.isOpened():
                    st.error("Camera cannot be opened. Close other apps using camera.")
                    st.stop()
                st.session_state.cap = new_cap
                st.session_state.running = True

        if st.button("Stop Stream"):
            st.session_state.running = False
            if st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None

        st.write("Camera status:", "Opened" if (st.session_state.cap and st.session_state.cap.isOpened()) else "Not initialized")
                
        st.markdown("---")
        
        detection_enabled = st.checkbox("Detection Enabled", value=True)
        alerts_enabled = st.checkbox("Alerts Enabled", value=True)
        demo_mode = st.checkbox("Demo Mode", value=args.demo_mode)
        st.checkbox("Mute Audio", key="muted")
        save_data_enabled = st.checkbox("Save Logs to Disk", value=True)
        
        args.demo_mode = demo_mode
        core["demo_manager"].enabled = demo_mode
        core["audio_controller"].muted = st.session_state.muted

        st.markdown("---")
        st.write(f"GPU: {'ON' if core['runtime']['gpu_enabled'] else 'OFF'}")
        st.write(f"Tracked Classes: {args.classes or 'All'}")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Live Monitoring", "Analytics", "Person Intelligence", "Use Cases", "Insights", "Crowd Monitoring"])
    
    with tab1:
        col1, col2 = st.columns([7, 3])
        with col1:
            frame_placeholder = st.empty()
        with col2:
            st.markdown("### Operational Snapshot")
            if detection_enabled:
                st.markdown("🟢 **Monitoring Active**")
            else:
                st.markdown("🔴 **Monitoring Off**")
            metric_cols = st.columns(2)
            fps_metric = metric_cols[0].empty()
            gpu_metric = metric_cols[1].empty()
            tracks_metric = st.columns(2)[0].empty()
            alerts_metric = st.columns(2)[1].empty()
            
            st.markdown("### Status")
            status_placeholder = st.empty()
            
            st.markdown("### Alert Center")
            alert_logs_placeholder = st.empty()

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            u_tracks_metric = st.empty()
            a_now_metric = st.empty()
            st.markdown("### Object Distribution")
            class_chart = st.empty()
        with col2:
            p_active_metric = st.empty()
            alerts_tr_metric = st.empty()
            st.markdown("### Tracking Stats")
            track_chart = st.empty()

    with tab3:
        i_col1, i_col2, i_col3, i_col4 = st.columns(4)
        pi_tracked = i_col1.empty()
        pi_active = i_col2.empty()
        pi_idle = i_col3.empty()
        pi_df = i_col4.empty()
        st.markdown("### Tracked Persons Table")
        pi_table = st.empty()

    with tab4:
        st.markdown("### 🛡️ Suspicious Behavior Parameters")
        st.markdown(
            "The system automatically classifies tracks as **Suspicious** when specific behavioral or spatial "
            "thresholds are breached. Here are the active conditions:"
        )

        colA, colB, colC = st.columns(3)
        with colA:
            st.info("**⏱️ Idle Loitering**\n\nIdle time > configuration threshold (e.g. 10 seconds).")
            st.info("**🧭 Restricted Zones**\n\nEntry into an off-limits or high-security zone.")
        with colB:
            st.info("**🚶‍♂️ Micro-Movement**\n\nMovement speed consistently below normal activity threshold.")
            st.info("**📊 Tracking Risks**\n\nHigh computational risk score generated from DeepSORT intent analysis.")
        with colC:
            st.info("**🔁 Repeated Presence**\n\nRe-occurring presence in the camera frame over a short duration.")
            
        st.markdown("---")
        st.markdown("### 🏢 Core Use Cases")
        st.write("- **Smart Surveillance**: Monitor live spaces efficiently with non-intrusive bounding boxes.")
        st.write("- **Behavior Monitoring**: Surface idle and loitering behaviors automatically with snapshots.")
        st.write("- **Restricted Access**: Keep intelligent focus on priority intruders using tracking.")
        st.write("- **Anomaly Detection**: Combine multi-class alerts and facial intelligence locally.")

    with tab5:
        st.markdown("### 🚨 Suspicious Persons")
        insights_placeholder = st.empty()

    with tab6:
        st.markdown("### 👥 Crowd Monitoring")
        crowd_placeholder = st.empty()

    if not st.session_state.running or st.session_state.cap is None:
        st.info("Stream is currently stopped. Click 'Start Stream' to begin.")
        return

    cap = st.session_state.cap

    # Pipeline loop inside Streamlit to deliver live frames
    while st.session_state.running:
        started_at = time.perf_counter()
        core["frame_index"] += 1
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read frame")
            st.session_state.running = False
            break

        annotated_frame = frame.copy()
        now = time.time()
        
        detection_summaries = []
        active_tracks = []
        active_counts = Counter()
        suspicious_ids = set()
        track_analysis = {}
        intent_analysis = {}
        person_intelligence_state = core["person_intelligence"].snapshot()
        primary_target = None
        latest_reasoning = core["reasoning_store"][-1] if core["reasoning_store"] else None
        
        if detection_enabled:
            with torch.inference_mode():
                inference_device = 0 if core["runtime"]["gpu_enabled"] else "cpu"
                results = st.session_state.model(
                    frame,
                    verbose=False,
                    device=inference_device,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    classes=sorted(core["tracked_class_ids"]) if core["tracked_class_ids"] is not None else None,
                    half=core["runtime"]["use_half"],
                )
            annotated_frame = results[0].plot()
            detection_summaries = extract_detection_summary(results[0], core["names"])
            detections = build_detections(results[0])
            tracks = core["tracker"].update_tracks(detections, frame=frame)
            active_tracks, active_counts = render_tracks(
                annotated_frame, tracks, core["names"], core["counter"], core["track_history"], trail_length=max(2, args.trail_length)
            )
            
            for track in active_tracks:
                if track.get("hits", 0) >= 3 and str(track.get("class_name", "")).lower() == "person":
                    core["unique_persons"].add(track["track_id"])
                    
            # Crowd overcrowding detection
            person_count = active_counts.get("person", 0)
            if alerts_enabled and person_count > 4:
                if now - core["crowd_alert_last_time"] > 10.0:
                    core["crowd_alert_last_time"] = now
                    core["audio_controller"].trigger("Overcrowded area detected", alert_key="crowd-alert", cooldown_seconds=10.0, siren=True, speech=True)

            # Danger detection
            danger_detected_this_frame = False
            danger_track = None
            for track in active_tracks:
                class_name = str(track.get("class_name", "")).lower()
                if class_name in ["knife", "weapon", "blood", "gun", "firearm", "sword", "baseball bat", "scissors", "bottle", "rifle"]:
                    danger_detected_this_frame = True
                    danger_track = track
                    break
                    
            if danger_detected_this_frame and alerts_enabled:
                if now - core["danger_alert_last_time"] > 10.0:
                    core["danger_alert_last_time"] = now
                    core["audio_controller"].trigger("Danger detected", alert_key="danger-alert", cooldown_seconds=10.0, siren=True, speech=True)
                    if danger_track and danger_track.get("track_id") not in core["saved_suspicious_ids"]:
                        filepath = core["data_logger"].log_suspicious_event(annotated_frame, danger_track, "danger object", enabled=save_data_enabled)
                        if filepath:
                            core["saved_suspicious_ids"].add(danger_track["track_id"])
                            core["suspicious_logs"].append({
                                "track_id": danger_track["track_id"],
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "reason": "danger object",
                                "image_path": filepath
                            })

            suspicious_ids, new_suspicious_tracks, track_analysis = core["suspicious_detector"].update(active_tracks, now)
            intent_analysis, new_intent_tracks = core["intent_predictor"].update(active_tracks, now, frame.shape)
            
            for track in active_tracks:
                suspicious_analysis = track_analysis.get(track["track_id"], {})
                intent_info = intent_analysis.get(track["track_id"], {})
                dwell_time = suspicious_analysis.get("dwell_time", 0.0)
                confidence, priority = score_dwell_time(dwell_time)
                track["dwell_time"] = round(dwell_time, 1)
                track["priority"] = priority if suspicious_analysis.get("suspicious") else "NORMAL"
                track["confidence"] = round(confidence, 2) if suspicious_analysis.get("suspicious") else track.get("detection_confidence", 0.0)
                track["suspicious"] = bool(suspicious_analysis.get("suspicious"))
                track["intent"] = intent_info.get("intent", "normal movement")
                track["movement_speed"] = intent_info.get("speed", 0.0)
                track["direction_consistency"] = intent_info.get("direction_consistency", 0.0)
                track["time_in_zone"] = intent_info.get("time_in_zone", 0.0)
                track["time_in_frame"] = intent_info.get("time_in_frame", 0.0)
                track["zone_proximity"] = intent_info.get("zone_proximity", 0.0)
                track["risk_score"] = intent_info.get("risk_score", 0.0)
                track["suspicious_intent"] = intent_info.get("suspicious_intent", False)

            person_intelligence_state = core["person_intelligence"].update(active_tracks, frame, now, core["frame_index"])
            intelligence_lookup = {record["track_id"]: record for record in person_intelligence_state.get("records", [])}
            
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
            
            core["data_logger"].update_persons(person_intelligence_state.get("records", []), enabled=save_data_enabled)

            primary_target = prioritize_target(active_tracks, {**track_analysis, **intent_analysis})
            highlight_special_tracks(annotated_frame, active_tracks, suspicious_ids, None if primary_target is None else primary_target["track_id"])

            if alerts_enabled:
                for t in new_suspicious_tracks[:2]:
                    reasoning = build_suspicious_reasoning(t, track_analysis.get(t["track_id"], {}))
                    if now - core["alert_cooldowns"][f"suspicious-{t['track_id']}"] > 10.0:
                        if core["alert_manager"].trigger(annotated_frame, reasoning["reason"], trigger_key=f"idle-person-{t['track_id']}", cooldown_seconds=10.0):
                            core["alert_cooldowns"][f"suspicious-{t['track_id']}"] = now
                            latest_reasoning = remember_reasoning(core["reasoning_store"], reasoning)
                            core["recent_alerts"].append(reasoning)
                            core["event_logger"].add("suspicious_activity", t["track_id"], reasoning["reason"])
                            if t["track_id"] not in core["saved_suspicious_ids"]:
                                filepath = core["data_logger"].log_suspicious_event(annotated_frame, t, reasoning["reason"], enabled=save_data_enabled)
                                if filepath:
                                    core["saved_suspicious_ids"].add(t["track_id"])
                                    core["suspicious_logs"].append({
                                        "track_id": t["track_id"],
                                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "reason": reasoning["reason"],
                                        "image_path": filepath
                                    })

                for t in new_intent_tracks[:2]:
                    i_info = intent_analysis.get(t["track_id"], {})
                    if not i_info.get("suspicious_intent"): continue
                    reasoning = build_intent_reasoning(t, i_info)
                    if now - core["alert_cooldowns"][f"intent-{t['track_id']}"] > 10.0:
                        if core["alert_manager"].trigger(annotated_frame, reasoning["reason"], trigger_key=f"intent-{t['track_id']}", cooldown_seconds=10.0):
                            core["alert_cooldowns"][f"intent-{t['track_id']}"] = now
                            latest_reasoning = remember_reasoning(core["reasoning_store"], reasoning)
                            core["recent_alerts"].append(reasoning)
                            core["event_logger"].add("intent_alert", t["track_id"], reasoning["reason"])
                            if t["track_id"] not in core["saved_suspicious_ids"]:
                                filepath = core["data_logger"].log_suspicious_event(annotated_frame, t, reasoning["reason"], enabled=save_data_enabled)
                                if filepath:
                                    core["saved_suspicious_ids"].add(t["track_id"])
                                    core["suspicious_logs"].append({
                                        "track_id": t["track_id"],
                                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                        "reason": reasoning["reason"],
                                        "image_path": filepath
                                    })

            new_classes = core["detection_announcer"].update(active_counts)
            for c_name in new_classes:
                core["event_logger"].add("person_detected" if c_name == "person" else "object_detected", None, f"{c_name} detected")

            monitored_tracks = [t for t in active_tracks if core["alert_class_ids"] is None or t["class_id"] in core["alert_class_ids"]]
            if alerts_enabled and len(monitored_tracks) >= core["alert_manager"].threshold > 0:
                reasoning = build_alert_reasoning(active_tracks, core["alert_class_ids"])
                if reasoning and core["alert_manager"].trigger(annotated_frame, reasoning["reason"], trigger_key="threshold-alert"):
                    latest_reasoning = remember_reasoning(core["reasoning_store"], reasoning)
                    core["recent_alerts"].append(reasoning)
                    core["event_logger"].add("alert", reasoning["track_id"], reasoning["reason"])

            demo_reasoning = core["demo_manager"].maybe_simulate_alert(now, active_tracks)
            if alerts_enabled and demo_reasoning is not None:
                if core["alert_manager"].trigger(annotated_frame, demo_reasoning["reason"], trigger_key="demo-alert", cooldown_seconds=12.0):
                    latest_reasoning = remember_reasoning(core["reasoning_store"], demo_reasoning)
                    core["recent_alerts"].append(demo_reasoning)
                    core["event_logger"].add("alert", None, demo_reasoning["reason"])

            core["focus_manager"].draw(annotated_frame, primary_target, suspicious_ids)
            
        else:
            core["tracker"].update_tracks([], frame=frame)
            core["suspicious_detector"].update([], now)
            core["detection_announcer"].reset()
            person_intelligence_state = core["person_intelligence"].update([], frame, now, core["frame_index"])
            core["data_logger"].update_persons(person_intelligence_state.get("records", []), enabled=save_data_enabled)

        elapsed = max(time.perf_counter() - started_at, 1e-6)
        instant_fps = 1.0 / elapsed
        st.session_state.smoothed_fps = instant_fps if st.session_state.smoothed_fps == 0.0 else (st.session_state.smoothed_fps * 0.9 + instant_fps * 0.1)
        
        core["analytics_store"]["active_timeline"].append((now, core["counter"].active_total))
        core["analytics_store"]["alerts_triggered"] = sum(1 for log in core["event_logger"].entries if log["event"] == "alert")
        
        system_state = build_system_state(
            detections=detection_summaries,
            tracks=active_tracks,
            alerts=list(core["recent_alerts"]),
            primary_target=primary_target,
            reasoning=latest_reasoning,
            logs=core["event_logger"].as_list(limit=25),
            status={
                "detection_enabled": detection_enabled,
                "alerts_enabled": alerts_enabled,
                "audio_muted": st.session_state.muted,
                "active_total": core["counter"].active_total,
                "unique_total": core["counter"].unique_total,
            },
            person_intelligence=person_intelligence_state,
        )
        analytics_view = build_dashboard_analytics(core["analytics_store"], core["counter"], system_state)

        # Update Live Dashboard
        frame_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
        fps_metric.metric("FPS", f"{st.session_state.smoothed_fps:.1f}")
        gpu_metric.metric("Device", core["runtime"]["gpu_name"] if core["runtime"]["gpu_enabled"] else "CPU")
        tracks_metric.metric("Active Tracks", str(core["counter"].active_total))
        alerts_metric.metric("Alert Status", "Enabled" if alerts_enabled else "Muted")

        status_placeholder.markdown(f"""
        - **System**: {'Active' if detection_enabled else 'Paused'}
        - **Unique Tracked Persons**: {len(core["unique_persons"])}
        - **Total Unique Objects**: {core["counter"].unique_total}
        - **Focus Target**: {'None' if primary_target is None else f"#{primary_target['track_id']}"}
        - **Intent**: {getattr(primary_target, 'intent', 'None').title() if primary_target else 'None'}
        """)

        alerts_html = "<ul>"
        for alert in list(reversed(system_state["alerts"][-6:])):
            alerts_html += f"<li><b>{alert['timestamp']}</b> [{alert['priority']}] {alert['reason']}</li>"
        alerts_html += "</ul>"
        alert_logs_placeholder.markdown(alerts_html, unsafe_allow_html=True)
        
        # Update Analytics Dashboard
        with u_tracks_metric.container():
            st.metric("Unique Tracked Persons", len(core["unique_persons"]))
            st.caption("Counts each individual tracked once using persistent track IDs.")
            
        with a_now_metric.container():
            st.metric("Active Persons Now", str(active_counts.get("person", 0)))
            st.caption("Number of people currently visible in frame.")
            
        with p_active_metric.container():
            st.metric("Peak Active", analytics_view["peak_active"])
            st.caption("Highest number of concurrent persons recorded.")
            
        with alerts_tr_metric.container():
            st.metric("Alerts Triggered", analytics_view["alerts_triggered"])
            st.caption("Triggered based on behavior analysis and danger detection.")
        
        if analytics_view["class_bars"]:
            cb_df = pd.DataFrame(analytics_view["class_bars"], columns=["Class", "Count"]).set_index("Class")
            class_chart.bar_chart(cb_df)
            
        tb_df = pd.DataFrame(analytics_view["tracking_bars"], columns=["Metric", "Value"]).set_index("Metric")
        track_chart.bar_chart(tb_df)

        # Update Person Intelligence Dashboard
        pi_records = system_state.get("person_intelligence", {}).get("records", [])
        active_records = [r for r in pi_records if r.get("active")]
        idle_records = [r for r in active_records if r.get("movement_state") == "idle"]
        
        pi_tracked.metric("Tracked Persons", str(len(pi_records)))
        pi_active.metric("Active Persons", str(len(active_records)))
        pi_idle.metric("Idle Persons", str(len(idle_records)))
        pi_df.metric("DeepFace Backend", "Ready" if system_state.get("person_intelligence", {}).get("enabled") else "Offline")
        
        if pi_records:
            pi_df_gui = pd.DataFrame(pi_records)
            pi_table.dataframe(
                pi_df_gui[["track_id", "age", "gender", "emotion", "total_duration", "movement_state", "entry_time", "exit_time"]],
                use_container_width=True,
                hide_index=True
            )
        else:
            pi_table.info("No person tracks captured yet.")
            
        # Update Insights Dashboard
        logs = list(core["suspicious_logs"])
        with insights_placeholder.container():
            if not logs:
                st.info("No suspicious persons logged yet.")
            else:
                cols = st.columns(3)
                for i, log in enumerate(reversed(logs)):
                    with cols[i % 3]:
                        with st.container(border=True):
                            if log.get("image_path"):
                                st.image(log["image_path"], use_container_width=True)
                            st.markdown(f"#### Track #{log['track_id']}")
                            st.caption(f"🕒 {log['timestamp']}")
                            st.error(f"🚨 {log['reason']}")

        # Update Crowd Dashboard
        with crowd_placeholder.container():
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Current Active Persons", str(active_counts.get("person", 0)))
            with col2:
                if active_counts.get("person", 0) > 4:
                    st.error("⚠️ Overcrowding Warning: Active persons threshold exceeded!")
                else:
                    st.success("✅ Crowd levels normal.")

        # Rate limiting to hit target FPS roughly without freezing
        time.sleep(0.03)

if __name__ == "__main__":
    main()
