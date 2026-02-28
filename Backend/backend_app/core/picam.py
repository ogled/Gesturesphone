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

BASE_DIR = Path(__file__).resolve().parents[2]

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

fps = 0

prev_coords = None
prev_velocity = None
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
prev_coords = None
prev_velocity = None
recording_mode = False
recording_buffer = []

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

def create_feature_vector(multi_hand_landmarks, multi_handedness):
    global prev_coords, prev_velocity

    hands = np.zeros((2, 21, 3), dtype=np.float32)

    if multi_hand_landmarks and multi_handedness:
        for lm, hd in zip(multi_hand_landmarks, multi_handedness):
            idx = 0 if hd.classification[0].label == "Left" else 1
            coords = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
            hands[idx] = coords

    left_raw = hands[0]
    right_raw = hands[1]

    def norm_hand(J):
        wrist = J[0:1] 
        Jc = J - wrist
        palm = Jc[9] 
        scale = np.linalg.norm(palm) + 1e-6
        return Jc / scale

    left = norm_hand(left_raw)
    right = norm_hand(right_raw)

    coords = np.concatenate([left.ravel(), right.ravel()])

    if prev_coords is None:
        velocity = np.zeros_like(coords)
        acceleration = np.zeros_like(coords)
    else:
        velocity = coords - prev_coords
        if prev_velocity is None:
            acceleration = np.zeros_like(coords)
        else:
            acceleration = velocity - prev_velocity

    prev_coords = coords.copy()
    prev_velocity = velocity.copy()

    motion_energy = np.linalg.norm(velocity) 
    motion_energy = np.array([[motion_energy]], dtype=np.float32)

    left_present = float(np.abs(left).sum() > 0)
    right_present = float(np.abs(right).sum() > 0)
    hands_count = np.array([[(left_present + right_present) / 2.0]], dtype=np.float32)

    inter_hand_dist = np.linalg.norm(left[0] - right[0]).reshape(1,1)

    def polygon_area(points):
        x = points[:, 0]
        y = points[:, 1]
        x_shift = np.roll(x, -1)
        y_shift = np.roll(y, -1)
        return 0.5 * np.abs(np.sum(x * y_shift) - np.sum(y * x_shift))

    palm_points = [0, 1, 5, 9, 13, 17]
    left_area = np.array([[polygon_area(left[palm_points, :2])]], dtype=np.float32)
    right_area = np.array([[polygon_area(right[palm_points, :2])]], dtype=np.float32)

    key_pairs = [(0,4),(0,8),(0,12),(0,16),(0,20),
                 (4,8),(8,12),(12,16),(16,20)]
    def compute_dists(J):
        return np.array([np.linalg.norm(J[i] - J[j]) for i, j in key_pairs]).reshape(1,-1)

    left_dists = compute_dists(left)   
    right_dists = compute_dists(right) 

    spread_left = np.linalg.norm(left[4] - left[20]).reshape(1,1)
    spread_right = np.linalg.norm(right[4] - right[20]).reshape(1,1)

    if prev_velocity is None or np.linalg.norm(velocity) == 0:
        curvature = np.zeros((1,1), dtype=np.float32)
    else:
        vel_norm = velocity / (np.linalg.norm(velocity) + 1e-6)
        prev_norm = prev_velocity / (np.linalg.norm(prev_velocity) + 1e-6)
        curvature = np.linalg.norm(vel_norm - prev_norm).reshape(1,1)

    palm_vec_left = left[9] - left[0]
    palm_dir_left = (palm_vec_left / (np.linalg.norm(palm_vec_left) + 1e-6)).reshape(1,3)
    palm_vec_right = right[9] - right[0]
    palm_dir_right = (palm_vec_right / (np.linalg.norm(palm_vec_right) + 1e-6)).reshape(1,3)

    feats = np.concatenate([
        hands_count,        
        coords.reshape(1,-1),
        velocity.reshape(1,-1), 
        acceleration.reshape(1,-1), 
        motion_energy,       
        left_area,           
        right_area,          
        inter_hand_dist,     
        left_dists,          
        right_dists,         
        spread_left,         
        spread_right,        
        curvature,           
        palm_dir_left,       
        palm_dir_right       
    ], axis=1)

    return torch.tensor(feats, dtype=torch.float32)

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
    global latest_frame, processed_frame
    global prev_coords, prev_velocity 
    prev_coords = None
    prev_velocity = None

    gesture_name = ""
    confidence = 0.0
    all_probabilities = []

    frame_id = 0
    while not stopThreads:
        with lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.01)
            continue

        start_time = time.time()
        display_frame = frame.copy()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame_id % 2 == 0:
            results = hands.process(rgb_frame) 
        frame_id += 1

        if not results.multi_hand_landmarks:
            prev_coords = None
            prev_velocity = None
            feature_buffer.clear()
            gesture_name = ""
            confidence = 0.0
        else:
            feat_tensor = create_feature_vector(
                results.multi_hand_landmarks,
                results.multi_handedness
            )
            feature_buffer.append(feat_tensor.squeeze(0).cpu().numpy())

            for hand in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    display_frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

            if len(feature_buffer) >= sequence_length:
                current_sequence = list(feature_buffer)[-sequence_length:]
                seq_feats = np.stack(current_sequence, axis=0) 

                raw_me = seq_feats[:, 379:380] 
                mean = np.mean(raw_me)
                std = np.std(raw_me) + 1e-6
                norm_me = (raw_me - mean) / std
                seq_feats[:, 379:380] = norm_me

                sequence_tensor = torch.tensor(seq_feats, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    emb = model(sequence_tensor, return_embedding=True)
                    logits = model.arcface(emb)
                    probs = torch.softmax(logits, dim=1)

                    probs_np = probs.cpu().numpy()[0]
                    best_idx = int(np.argmax(probs_np))
                    best_conf = float(probs_np[best_idx])

                    if best_conf >= CONF_THRESHOLD:
                        gesture_name = labels[best_idx]
                        confidence = best_conf
                    else:
                        gesture_name = None
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
    global mp_hands, mp_drawing, hands, model
    global cap, out, device, model_path, state, labels, num_classes, real_input_size
    global feature_buffer, lock, new_frame_ready, threads, latest_gesture_probs

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=2, 
        model_complexity=0,
        min_detection_confidence=0.3,
        static_image_mode=False,
        min_tracking_confidence=0.2
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print_error("Камера не подключена")
        os._exit(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)

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