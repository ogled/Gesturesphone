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

os.chdir(Path(__file__).resolve().parent)

CONF_THRESHOLD = 0.4
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

mp_hands = None
mp_drawing = None
hands = None

CAPTURE_FPS = 30
DISPLAY_FPS = 24 
DISPLAY_INTERVAL = 1.0 / DISPLAY_FPS

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

def print_error(str):
    print(f"\033[31m[ERROR] {str}\033[0m")
    time.sleep(2)

class SimpleTCN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.attn = nn.Sequential(
            nn.Conv1d(128, 1, 1),
            nn.Softmax(dim=2)
        )
        self.norm = nn.LayerNorm(128)
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        feat = self.conv(x)
        feat = feat + torch.randn_like(feat) * 0.025  # шум
        w = self.attn(feat)
        out = (feat * w).sum(dim=2)
        out = self.norm(out)
        out = self.dropout(out)
        return self.fc(out)

def create_feature_vector(multi_hand_landmarks, multi_handedness):
    hands_feats = np.zeros((2, 21, 3), dtype=np.float32)

    if multi_hand_landmarks and multi_handedness:
        for hand_landmarks, handedness in zip(multi_hand_landmarks, multi_handedness):
            label = handedness.classification[0].label
            idx = 0 if label == "Left" else 1
            coords_centered = coords - wrist
            palm_vec = coords_centered[9, :]
            palm_size = np.linalg.norm(palm_vec) + 1e-6
            coords_normalized = coords_centered / palm_size
            
            hands_feats[idx] = coords_normalized
    feats = hands_feats.reshape(1, -1)  # [1, 126]

    feats = (feats - feats.mean(1, keepdims=True)) / (feats.std(1, keepdims=True) + 1e-6)

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

def process_thread():
    global latest_frame, processed_frame
    gesture_name = "None"
    confidence = 0.0
    all_probabilities = []

    dims = 3
    num_landmarks = 21
    expected_raw_features = num_landmarks * dims * 2
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
            # feature_buffer.clear()
            gesture_name = "None"
            confidence = 0.0
        else:
            features = create_feature_vector(
                results.multi_hand_landmarks,
                results.multi_handedness
            )
            feature_buffer.append(features.squeeze(0).cpu().numpy())

            for hand in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    display_frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

            # Предсказание
            if len(feature_buffer) >= sequence_length:
                current_sequence = list(feature_buffer)[-sequence_length:]
                seq_feats = np.stack(current_sequence, axis=0)  # [seq_len, 126]
                sequence_tensor = torch.tensor(seq_feats, dtype=torch.float32).unsqueeze(0).to(device)  # [1, seq_len, 126]

                with torch.no_grad():
                    preds = model(sequence_tensor)
                    probs = torch.softmax(preds, dim=1)
                    confidence, label_id = torch.max(probs, dim=1)
                    confidence = confidence.item()
                    label_id = label_id.item()
                    all_probabilities = probs.cpu().numpy()[0]
                    gesture_name = labels[label_id] if confidence > CONF_THRESHOLD else "None"

        # --- отрисовка ---
        latency = (time.time() - start_time) * 1000
        with lock:
            processed_frame = (display_frame, latency, gesture_name, confidence, all_probabilities)
        new_frame_ready.set()

def display_thread():
    global processed_frame, out, latest_web_frame, last_gesture, latest_gesture_probs
    frame_count = 0
    last_print = time.time()
    gesture_history = deque(maxlen=10)
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

        if gesture != "None":
            gesture_history.append(gesture)
        latest_web_frame = frame.copy()
        frame_count += 1
        now = time.time()
        if now - last_print >= 2.0:
            fps = frame_count / (now - last_print)
            cpu_usage = psutil.cpu_percent(interval=None)
            most_common = max(set(gesture_history), key=gesture_history.count) if gesture_history else "None"
            stability = gesture_history.count(most_common)/len(gesture_history) if gesture_history else 0.0
            print(f"[INFO] FPS: {fps:.1f}, Latency: {latency:.1f}ms, CPU: {cpu_usage:.1f}%")
            print(f"       Current: {gesture} (conf: {confidence:.2f})")
            print(f"       Most common: {most_common} (stability: {stability:.2f})")
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
        min_detection_confidence=0.5,
        static_image_mode=False,
        min_tracking_confidence=0.25,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print_error("Камера не подключена")
        os._exit(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)

    if save_video:
        if not os.path.exists('Output'):
            os.makedirs('Output')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('Output/output.avi', fourcc, 15.0, (hCam, wCam))
        if not out.isOpened():
            print_error("Не удалось открыть файл для записи видео")

    device = torch.device("cpu")
    print("Device:", device)

    model_path = "../Train/model.pth"
    if not os.path.exists(model_path):
        print_error("Файл ML не найден")
        os._exit(0)

    state = torch.load(model_path, map_location=device, weights_only=False)

    labels = state['labels']
    num_classes = len(labels)

    print("=== АНАЛИЗ СТРУКТУРЫ МОДЕЛИ ===")
    real_input_size = None
    for key in state['model_state'].keys():
        if 'tcn.0.weight' in key or 'tcn.0.conv.weight' in key:
            real_input_size = state['model_state'][key].shape[1]
            print(f"Найден ключ: {key}, shape: {state['model_state'][key].shape}")
            print(f"РЕАЛЬНЫЙ INPUT_SIZE: {real_input_size}")
            break

    if real_input_size is None:
        for key, weight in state['model_state'].items():
            if 'weight' in key and len(weight.shape) == 3:
                real_input_size = weight.shape[1]
                print(f"Альтернативный поиск: {key}, shape: {weight.shape}")
                print(f"РЕАЛЬНЫЙ INPUT_SIZE: {real_input_size}")
                break

    if real_input_size is None:
        real_input_size = 126
        print(f"[WARN] Не удалось определить input_size, используем {real_input_size}")

    print(f"[INFO] Используется input_size: {real_input_size}")
    print(f"[INFO] Классы: {labels}")

    model = SimpleTCN(input_size=real_input_size, num_classes=num_classes).to(device)
    model.load_state_dict(state['model_state'])
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