import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import time
import threading
import psutil
from collections import deque
from pathlib import Path

from .classifier_runtime import (
    ClassifierRuntime,
    OnnxClassifierRuntime,
    TorchClassifierRuntime,
    load_runtime_metadata,
)
from .feature_contract import FEATURE_CONTRACT
from .picam_feature_extractor import FeatureExtractor, draw_landmarks


def _iter_runtime_roots():
    module_path = Path(__file__).resolve()
    yield module_path.parents[3]

    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            yield Path(meipass)
        exe_dir = Path(sys.executable).resolve().parent
        yield exe_dir
        yield exe_dir / "_internal"


def _resolve_train_dir() -> Path:
    seen = set()
    first_root = None
    for root in _iter_runtime_roots():
        root = root.resolve()
        if first_root is None:
            first_root = root
        if root in seen:
            continue
        seen.add(root)
        candidate = root / "Train"
        if candidate.exists():
            return candidate
    return (first_root or Path(__file__).resolve().parents[3]) / "Train"


PROJECT_ROOT = next(iter(_iter_runtime_roots())).resolve()
TRAIN_DIR = _resolve_train_dir()

CONF_THRESHOLD = 0.4
MIN_HOLD_TIME = 0.6
MIN_SHOW_TIME = 0.5

hCam = 1280
wCam = 720
sequence_length = FEATURE_CONTRACT.sequence_length
save_video = False
latest_web_frame = None
latest_gesture_probs = {"ts": 0, "probs": {}}
gesture_history = deque(maxlen=20)

mp_hands = None
hands = None

CAPTURE_FPS = 60
CAPTURE_INTERVAL = 1.0 / CAPTURE_FPS
DISPLAY_FPS = 60
DISPLAY_INTERVAL = 1.0 / DISPLAY_FPS

fps = 0
stopThreads = False
out = None
cap = None
labels = None
num_classes = None
real_input_size = None
feature_buffer = None
latest_frame = None
processed_frame = None
lock = None
new_frame_ready = None
threads = None
recording_mode = False
recording_buffer = []
feature_extractor = None
classifier_runtime: ClassifierRuntime | None = None


def print_error(message: str):
    print(f"\033[31m[ERROR] {message}\033[0m")
    time.sleep(2)


def _build_classifier_runtime() -> ClassifierRuntime:
    backend = os.getenv("GESTURE_RUNTIME_BACKEND", "onnx").strip().lower()
    onnx_path = TRAIN_DIR / "model.onnx"
    metadata_path = TRAIN_DIR / "model.runtime.json"
    checkpoint_path = TRAIN_DIR / "model.pth"

    if backend == "torch":
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print("[INFO] Runtime backend: torch")
        return TorchClassifierRuntime(checkpoint_path)

    if not onnx_path.exists() or not metadata_path.exists():
        if checkpoint_path.exists():
            backend = "torch"
            print("[WARN] ONNX artifacts were not found, falling back to torch backend.")
            print("[WARN] Export model first with: python Train/export_to_onnx.py")
            return TorchClassifierRuntime(checkpoint_path)
        raise FileNotFoundError(
            f"Missing ONNX runtime artifacts: {onnx_path} and/or {metadata_path}"
        )

    metadata = load_runtime_metadata(metadata_path)
    if metadata.input_size != FEATURE_CONTRACT.feature_size:
        raise ValueError(
            f"Metadata input_size={metadata.input_size}, expected {FEATURE_CONTRACT.feature_size}"
        )
    if metadata.sequence_length != sequence_length:
        raise ValueError(
            f"Metadata sequence_length={metadata.sequence_length}, expected {sequence_length}"
        )
    if metadata.feature_contract_version != FEATURE_CONTRACT.version:
        raise ValueError(
            f"Feature contract mismatch: {metadata.feature_contract_version} != {FEATURE_CONTRACT.version}"
        )

    print("[INFO] Runtime backend: onnx")
    return OnnxClassifierRuntime(onnx_path, metadata)


def capture_thread():
    global latest_frame
    next_capture_at = time.perf_counter()
    while not stopThreads:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            with lock:
                latest_frame = frame.copy()
        next_capture_at += CAPTURE_INTERVAL
        sleep_for = next_capture_at - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_capture_at = time.perf_counter()


def get_text(history):
    return " ".join(g.capitalize() for g in history if g is not None)


def process_thread():
    global latest_frame, processed_frame, feature_extractor

    feature_extractor.reset()
    gesture_name = ""
    confidence = 0.0
    all_probabilities = []

    frame_id = 0
    last_results = None
    while not stopThreads:
        with lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.005)
            continue

        start_time = time.time()
        display_frame = frame.copy()

        if frame_id % 2 == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_results = hands.process(rgb_frame)
        frame_id += 1
        results = last_results

        if not results or not results.multi_hand_landmarks:
            feature_buffer.clear()
            feature_extractor.reset()
            gesture_name = ""
            confidence = 0.0
            all_probabilities = []
        else:
            feat_vector, hand_coords = feature_extractor.create_feature_vector(
                results.multi_hand_landmarks, results.multi_handedness
            )
            feature_buffer.append(feat_vector)
            draw_landmarks(
                display_frame, hand_coords, mp_hands.HAND_CONNECTIONS
            )

            if len(feature_buffer) >= sequence_length:
                seq_feats = np.stack(feature_buffer, axis=0).astype(np.float32, copy=False)

                motion_energy = seq_feats[:, FEATURE_CONTRACT.motion_energy_index : FEATURE_CONTRACT.motion_energy_index + 1]
                mean = np.mean(motion_energy)
                std = np.std(motion_energy) + 1e-6
                seq_feats[:, FEATURE_CONTRACT.motion_energy_index : FEATURE_CONTRACT.motion_energy_index + 1] = (
                    motion_energy - mean
                ) / std

                probs_np = classifier_runtime.predict_proba(np.expand_dims(seq_feats, axis=0))
                best_idx = int(np.argmax(probs_np))
                best_conf = float(probs_np[best_idx])

                if best_conf >= CONF_THRESHOLD:
                    gesture_name = labels[best_idx]
                    confidence = best_conf
                else:
                    gesture_name = ""
                    confidence = best_conf

                all_probabilities = probs_np.tolist()
            else:
                gesture_name = ""
                confidence = 0.0
                all_probabilities = []

        latency = (time.time() - start_time) * 1000
        with lock:
            processed_frame = (
                display_frame,
                latency,
                gesture_name,
                confidence,
                all_probabilities,
            )
        new_frame_ready.set()


def display_thread():
    global processed_frame, out, latest_web_frame, latest_gesture_probs, fps
    global gesture_history, recording_mode, recording_buffer

    frame_count = 0
    last_print = time.time()
    next_display_at = time.perf_counter()
    state = "NO_HANDS"
    gesture_start_time = 0.0
    last_added_gesture = None
    current_gesture = ""

    while not stopThreads:
        if not new_frame_ready.wait(timeout=DISPLAY_INTERVAL):
            continue
        new_frame_ready.clear()

        with lock:
            if processed_frame is None:
                continue
            frame, latency, gesture, confidence, probabilities = processed_frame
            if probabilities is not None and len(probabilities) == len(labels):
                latest_gesture_probs = {
                    "ts": int(time.time() * 1000),
                    "probs": {
                        labels[i].capitalize(): round(
                            float(round(probabilities[i], 4)) * 100
                        )
                        for i in range(len(labels))
                    },
                }

        now = time.time()
        if gesture == "":
            if state == "TRACKING":
                duration = now - gesture_start_time
                if duration >= MIN_SHOW_TIME and last_added_gesture != current_gesture:
                    gesture_history.append(current_gesture)
                    last_added_gesture = current_gesture
                state = "NO_HANDS"
        else:
            if state == "NO_HANDS":
                current_gesture = gesture
                gesture_start_time = now
                state = "TRACKING"
            elif state == "TRACKING":
                if gesture != current_gesture:
                    current_gesture = gesture
                    gesture_start_time = now
                elif now - gesture_start_time >= MIN_HOLD_TIME:
                    if last_added_gesture != current_gesture:
                        duration = now - gesture_start_time
                        gesture_history.append(current_gesture)
                        if recording_mode:
                            recording_buffer.append(
                                {
                                    "gesture": current_gesture,
                                    "duration": round(duration, 2),
                                    "confidence": int(confidence * 100),
                                }
                            )
                        last_added_gesture = current_gesture
                        state = "FIXED"
            elif state == "FIXED" and gesture != current_gesture:
                current_gesture = gesture
                gesture_start_time = now
                state = "TRACKING"

        current_tick = time.perf_counter()
        if current_tick >= next_display_at:
            latest_web_frame = frame.copy()
            if save_video and out is not None:
                out.write(frame)
            frame_count += 1
            next_display_at = current_tick + DISPLAY_INTERVAL

        now = time.time()
        if now - last_print >= 2.0:
            text = get_text(gesture_history)
            fps = frame_count / (now - last_print)
            cpu_usage = psutil.cpu_percent(interval=None)
            print(f"[INFO] FPS: {fps:.1f}, Latency: {latency:.1f}ms, CPU: {cpu_usage:.1f}%")
            print(f"       Current: {gesture} (conf: {confidence:.2f})")
            print(f"       Sentence: {text}")
            frame_count = 0
            last_print = now


async def initialization():
    global mp_hands, hands, cap, out, labels, num_classes, real_input_size
    global feature_buffer, lock, new_frame_ready, threads, latest_gesture_probs
    global feature_extractor, classifier_runtime, stopThreads

    stopThreads = False
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.2,
        static_image_mode=False,
        min_tracking_confidence=0.1,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print_error("Камера не подключена")
        os._exit(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
    cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if save_video:
        output_dir = PROJECT_ROOT / "Output"
        output_dir.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(
            str(output_dir / "output.avi"), fourcc, float(DISPLAY_FPS), (hCam, wCam)
        )
        if not out.isOpened():
            print_error("Не удалось открыть файл для записи видео")

    try:
        classifier_runtime = _build_classifier_runtime()
    except Exception as exc:
        print_error(f"Не удалось инициализировать runtime: {exc}")
        os._exit(0)

    labels = classifier_runtime.labels
    num_classes = classifier_runtime.metadata.num_classes
    real_input_size = classifier_runtime.metadata.input_size

    print(f"[INFO] input_size: {real_input_size}")
    print(f"[INFO] classes: {labels}")

    feature_buffer = deque(maxlen=sequence_length)
    feature_extractor = FeatureExtractor()
    lock = threading.Lock()
    new_frame_ready = threading.Event()

    threads = [
        threading.Thread(target=capture_thread, daemon=True),
        threading.Thread(target=process_thread, daemon=True),
        threading.Thread(target=display_thread, daemon=True),
    ]
    threads[0].start()
    latest_gesture_probs = {
        "ts": int(time.time() * 1000),
        "probs": {labels[i].capitalize(): 0 for i in range(len(labels))},
    }


isAlreadyStart = False


def start():
    global threads, stopThreads, isAlreadyStart
    if threads is not None:
        stopThreads = False
        threads[1].start()
        threads[2].start()
        isAlreadyStart = True


def stop():
    global stopThreads, isAlreadyStart
    stopThreads = True
    isAlreadyStart = False
