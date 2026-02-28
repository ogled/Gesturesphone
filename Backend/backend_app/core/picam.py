import cv2
import mediapipe as mp
import numpy as np
import os
import time
import threading
import psutil
import torch
import torch.nn as nn
from collections import deque

from pathlib import Path

from .picam_feature_extractor import FeatureExtractor, draw_smoothed_landmarks

BASE_DIR = Path(__file__).resolve().parents[2]

torch.set_num_threads(max(1, os.cpu_count() // 2))

CONF_THRESHOLD = 0.4
MIN_HOLD_TIME = 0.6
MIN_SHOW_TIME = 0.5

hCam = 640
wCam = 480
sequence_length = 14
save_video = False
last_gesture = None
latest_web_frame = None
latest_gesture_probs = {
    "ts": 0,
    "probs": {}
}
gesture_history = deque(maxlen=20)

mp_hands = None
mp_drawing = None
hands = None

CAPTURE_FPS = 30
DISPLAY_FPS = 24 
DISPLAY_INTERVAL = 1.0 / DISPLAY_FPS
MIN_CONFIDENNCE = 90
LANDMARK_SMOOTHING_ALPHA = 0.35

fps = 0

stopThreads = False
out = None
cap = None
device = None
model_path = None
state = None
labels = None
num_classes = None
real_input_size = None
model = None
feature_buffer = None
latest_frame = None
processed_frame = None
lock = None
new_frame_ready = None
threads = None
recording_mode = False
recording_buffer = []
feature_extractor = None

def print_error(str):
    print(f"\033[31m[ERROR] {str}\033[0m")
    time.sleep(2)

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=34.0, m=0.37):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, x, label=None):
        cosine = torch.nn.functional.linear(
            torch.nn.functional.normalize(x),
            torch.nn.functional.normalize(self.weight)
        )
        if label is None:
            return cosine * self.s
        theta = torch.acos(torch.clamp(cosine, -1 + 1e-6, 1 - 1e-6))
        target = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        logits = cosine * (1 - one_hot) + target * one_hot
        return logits * self.s


class Model(nn.Module):
    def __init__(self, input_size, num_classes, emb_dim=512):
        super().__init__()

        self.local_branch = nn.Sequential(
            nn.Conv1d(input_size, 128, 3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(128, 256, 3, padding=2, dilation=2),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3)
        )

        self.mid_branch = nn.Sequential(
            nn.Conv1d(input_size, 128, 5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(128, 256, 5, padding=4, dilation=2),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3)
        )

        self.global_branch = nn.Sequential(
            nn.Conv1d(input_size, 128, 7, padding=3),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(128, 256, 7, padding=6, dilation=2),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3)
        )

        self.fusion = nn.Sequential(
            nn.Conv1d(256 * 3, emb_dim, 1),
            nn.BatchNorm1d(emb_dim), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.arcface = ArcMarginProduct(emb_dim, num_classes)

    def forward(self, x, lengths=None, labels=None, return_embedding=False):
        x = x.permute(0, 2, 1)  # (batch, features, time)
        
        local_feat = self.local_branch(x)
        mid_feat = self.mid_branch(x)
        global_feat = self.global_branch(x)
        
        combined = torch.cat([local_feat, mid_feat, global_feat], dim=1)
        
        x = self.fusion(combined).squeeze(-1)
        
        if return_embedding:
            return x
        
        return self.arcface(x, labels)

def capture_thread():
    global latest_frame
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            with lock:
                latest_frame = frame.copy()
        time.sleep(1.0 / CAPTURE_FPS)

def get_text(gesture_history):
    return " ".join(g.capitalize() for g in gesture_history if g != None)

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
            feat_vector, smoothed_coords = feature_extractor.create_feature_vector(
                results.multi_hand_landmarks,
                results.multi_handedness,
            )
            feature_buffer.append(feat_vector)
            draw_smoothed_landmarks(display_frame, smoothed_coords, mp_hands.HAND_CONNECTIONS)

            if len(feature_buffer) >= sequence_length:
                seq_feats = np.stack(feature_buffer, axis=0)

                raw_me = seq_feats[:, 379:380]
                mean = np.mean(raw_me)
                std = np.std(raw_me) + 1e-6
                seq_feats[:, 379:380] = (raw_me - mean) / std

                sequence_tensor = torch.from_numpy(seq_feats).unsqueeze(0).to(device)

                with torch.no_grad():
                    emb = model(sequence_tensor, return_embedding=True)
                    logits = model.arcface(emb)
                    probs = torch.softmax(logits, dim=1)

                    probs_np = probs.detach().cpu().numpy()[0]
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
            processed_frame = (display_frame, latency, gesture_name, confidence, all_probabilities)
        new_frame_ready.set()

def display_thread():
    global processed_frame, out, latest_web_frame, last_gesture, latest_gesture_probs, fps, gesture_history, recording_mode, recording_buffer
    frame_count = 0
    last_print = time.time()
    state = "NO_HANDS"
    gesture_start_time = 0.0
    last_added_gesture = None
    while not stopThreads:
        if not new_frame_ready.wait(timeout=1.0):
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
                        labels[i].capitalize(): round(float(round(probabilities[i], 4)) * 100)
                        for i in range(len(labels))
                    }
                }

        if save_video and out is not None:
            out.write(frame)

        now = time.time()

        if gesture == "":
            if state == "TRACKING":
                duration = now - gesture_start_time
                if duration >= MIN_SHOW_TIME:
                    if last_added_gesture != current_gesture:
                        gesture_history.append(current_gesture)
                        last_added_gesture = current_gesture
                state = "NO_HANDS"

        elif gesture != "":
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
                            recording_buffer.append({
                                "gesture": current_gesture,
                                "duration": round(duration, 2),
                                "confidence": int(confidence * 100)
                            })

                        last_added_gesture = current_gesture
                        state = "FIXED"


            elif state == "FIXED":
                if gesture != current_gesture:
                    current_gesture = gesture
                    gesture_start_time = now
                    state = "TRACKING"


        latest_web_frame = frame.copy()
        frame_count += 1
        now = time.time()
        if now - last_print >= 2.0:
            text = get_text(gesture_history)
            fps = frame_count / (now - last_print)
            cpu_usage = psutil.cpu_percent(interval=None)
            most_common = max(set(gesture_history), key=gesture_history.count) if gesture_history else "None"
            stability = gesture_history.count(most_common)/len(gesture_history) if gesture_history else 0.0
            print(f"[INFO] FPS: {fps:.1f}, Latency: {latency:.1f}ms, CPU: {cpu_usage:.1f}%")
            print(f"       Current: {gesture} (conf: {confidence:.2f})")
            print(f"       Sentence: {text}")
            frame_count = 0
            last_print = now

def initialization():
    global mp_hands, hands, model
    global cap, out, device, model_path, state, labels, num_classes, real_input_size
    global feature_buffer, lock, new_frame_ready, threads, latest_gesture_probs, feature_extractor

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2, 
        model_complexity=1,
        min_detection_confidence=0.2,
        static_image_mode=False,
        min_tracking_confidence=0.1
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print_error("Камера не подключена")
        os._exit(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if save_video:
        output_dir = BASE_DIR / 'Output'
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(output_dir / 'output.avi'), fourcc, 15.0, (hCam, wCam))
        if not out.isOpened():
            print_error("Не удалось открыть файл для записи видео")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model_path = str(BASE_DIR / ".." / "Train" / "model.pth")
    if not os.path.exists(model_path):
        print_error("Файл ML не найден")
        os._exit(0)

    state = torch.load(model_path, map_location=device)

    labels = state["labels"]
    num_classes = state["num_classes"]
    real_input_size = state["input_size"]
    
    print(f"[INFO] input_size: {real_input_size}")
    print(f"[INFO] classes: {labels}")
    
    model = Model(real_input_size, num_classes).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    feature_buffer = deque(maxlen=sequence_length)
    feature_extractor = FeatureExtractor(smoothing_alpha=LANDMARK_SMOOTHING_ALPHA)

    lock = threading.Lock()
    new_frame_ready = threading.Event()

    threads = [
        threading.Thread(target=capture_thread, daemon=True),
        threading.Thread(target=process_thread, daemon=True),
        threading.Thread(target=display_thread, daemon=True)
    ]
    threads[0].start()
    latest_gesture_probs = {
                    "ts": int(time.time() * 1000),
                    "probs": {
                        labels[i].capitalize(): 0
                        for i in range(len(labels))
                    }
                }

isAlreadyStart = False
def start():
    global threads, stopThreads, isAlreadyStart
    if threads != None:
        threads[1].start()
        threads[2].start()
        stopThreads = False
        isAlreadyStart = True

def stop():
    global stopThreads, isAlreadyStart
    stopThreads = True
    isAlreadyStart = False
