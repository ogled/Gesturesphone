import cv2
import mediapipe as mp
import numpy as np
import mmap
import os
import time
import threading
import psutil
import torch
import torch.nn as nn
from collections import deque
from PIL import ImageFont, ImageDraw, Image
from Nextion import Nextion
from queue import Queue
import subprocess
import struct
from flask import Flask, Response
import threading
import cv2

# -------------------------
# Настройка параметров
# -------------------------
isTestRun = False
CONF_THRESHOLD = 0.4
save_video = False
nextion_queue = Queue()
last_gesture = None
app = Flask(__name__)
latest_web_frame = None
web_lock = threading.Lock()

def gen_frames():
    global latest_web_frame
    while True:
        with web_lock:
            if latest_web_frame is None:
                continue
            frame = latest_web_frame.copy()
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def start_flask():
    app.run(host='0.0.0.0', port=5000, threaded=True)
    
# Автоматическое определение дисплея
def auto_detect_display():
    """Автоматически определяет параметры дисплея"""
    try:
        # Пробуем получить информацию через fbset
        result = subprocess.run(['fbset', '-s'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'geometry' in line:
                    parts = line.split()
                    xres = int(parts[1])
                    yres = int(parts[2])
                    return xres, yres, xres * 2  # stride для RGB565
    except:
        pass
    
    try:
        # Пробуем через xdpyinfo (если X11 доступен)
        result = subprocess.run(['xdpyinfo'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'dimensions:' in line:
                    res = line.split()[1]
                    xres, yres = map(int, res.split('x'))
                    return xres, yres, xres * 2
    except:
        pass
    
    # Фолбэк на стандартные разрешения
    print("[INFO] Using fallback resolution 1920x1080")
    return 1920, 1080, 3840

# Инициализация дисплея
xres, yres, stride = auto_detect_display()
fb_path = "/dev/fb0"
display_connected = os.path.exists(fb_path)

if display_connected:
    try:
        fb = open(fb_path, "r+b")
        fb_map = mmap.mmap(fb.fileno(), yres * stride, mmap.MAP_SHARED, mmap.PROT_WRITE | mmap.PROT_READ)
        print(f"[INFO] Дисплей подключён: {xres}x{yres}")
    except Exception as e:
        print(f"[WARN] Ошибка открытия дисплея: {e}")
        display_connected = False
else:
    print("[INFO] Дисплей не найден — вывод отключён.")

# Nextion дисплей
nx = Nextion("/dev/serial0", 9600)
time.sleep(2)
nx.send("page 2")
time.sleep(1.2)

def rgb565_fast(img):
    """Универсальная конвертация BGR в RGB565"""
    try:
        # Ресайз до нужного разрешения
        if img.shape[:2] != (yres, xres):
            img = cv2.resize(img, (xres, yres))
        
        # Конвертируем BGR to RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Извлекаем каналы
        r = rgb_img[:, :, 0].astype(np.uint16)  # Красный
        g = rgb_img[:, :, 1].astype(np.uint16)  # Зеленый  
        b = rgb_img[:, :, 2].astype(np.uint16)  # Синий
        
        # Конвертируем в RGB565 (5-6-5 бит)
        rgb565 = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)
        
        return rgb565.tobytes()
    except Exception as e:
        print(f"[ERROR] Color conversion failed: {e}")
        # Возвращаем черный экран при ошибке
        black_frame = np.zeros((yres, xres), dtype=np.uint16)
        return black_frame.tobytes()

def print_error(str):
    nx.send("vis errorMSG3,1")
    nx.set_text("errorMSG3", str)
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
        feat = feat + torch.randn_like(feat) * 0.025  # легкий шум
        w = self.attn(feat)
        out = (feat * w).sum(dim=2)
        out = self.norm(out)
        out = self.dropout(out)
        return self.fc(out)

# -------------------------
# Инициализация Mediapipe
# -------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2, 
    model_complexity=0,
    min_detection_confidence=0.5,
    static_image_mode=False,
    min_tracking_confidence=0.25
)

# -------------------------
# Камера
# -------------------------
hCam = 640
wCam = 480
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print_error("Камера не подключена")
    os._exit(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
CAPTURE_FPS = 30
DISPLAY_FPS = 24 
DISPLAY_INTERVAL = 1.0 / DISPLAY_FPS
# -------------------------
# Видео сохранение
# -------------------------
out = None
if save_video:
    if not os.path.exists('Output'):
        os.makedirs('Output')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('Output/output.avi', fourcc, 15.0, (wCam, hCam))
    if not out.isOpened():
        print_error("Не удалось открыть файл для записи видео")

# Шрифт
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        from PIL import ImageFont
        font = ImageFont.load_default()

# -------------------------
# Загрузка модели
# -------------------------
device = torch.device("cpu")
print("Device:", device)

model_path = "../Training/model.pth"
if not os.path.exists(model_path):
    print_error("Файл ML не найден")
    os._exit(0)

state = torch.load(model_path, map_location=device, weights_only=False)

labels = state['labels']
num_classes = len(labels)

# ПРАВИЛЬНОЕ определение input_size
print("=== АНАЛИЗ СТРУКТУРЫ МОДЕЛИ ===")
real_input_size = None
for key in state['model_state'].keys():
    if 'tcn.0.weight' in key or 'tcn.0.conv.weight' in key:
        real_input_size = state['model_state'][key].shape[1]
        print(f"Найден ключ: {key}, shape: {state['model_state'][key].shape}")
        print(f"РЕАЛЬНЫЙ INPUT_SIZE: {real_input_size}")
        break

if real_input_size is None:
    # Альтернативный способ
    for key, weight in state['model_state'].items():
        if 'weight' in key and len(weight.shape) == 3:  # Conv1d weights
            real_input_size = weight.shape[1]
            print(f"Альтернативный поиск: {key}, shape: {weight.shape}")
            print(f"РЕАЛЬНЫЙ INPUT_SIZE: {real_input_size}")
            break

if real_input_size is None:
    real_input_size = 126  # fallback
    print(f"[WARN] Не удалось определить input_size, используем {real_input_size}")

print(f"[INFO] Используется input_size: {real_input_size}")
print(f"[INFO] Классы: {labels}")

model = SimpleTCN(input_size=real_input_size, num_classes=num_classes).to(device)
model.load_state_dict(state['model_state'])
model.eval()
# -------------------------
# Буфер и глобальные переменные
# -------------------------
sequence_length = 14
feature_buffer = deque(maxlen=sequence_length)

latest_frame = None
processed_frame = None
lock = threading.Lock()
running = True
new_frame_ready = threading.Event()

# -------------------------
# Функции
# -------------------------

def create_feature_vector(multi_hand_landmarks, multi_handedness):
    hands_feats = np.zeros((2, 21, 3), dtype=np.float32)

    if multi_hand_landmarks and multi_handedness:
        for hand_landmarks, handedness in zip(multi_hand_landmarks, multi_handedness):
            label = handedness.classification[0].label  # "Left" или "Right"
            idx = 0 if label == "Left" else 1
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
            
            wrist = coords[0:1, :]
            coords_centered = coords - wrist
            palm_vec = coords_centered[9, :]
            palm_size = np.linalg.norm(palm_vec) + 1e-6
            coords_normalized = coords_centered / palm_size
            
            hands_feats[idx] = coords_normalized
    feats = hands_feats.reshape(1, -1)  # [1, 126]

    feats = (feats - feats.mean(1, keepdims=True)) / (feats.std(1, keepdims=True) + 1e-6)

    return torch.tensor(feats, dtype=torch.float32)

# -------------------------
# Потоки
# -------------------------
def capture_thread():
    global latest_frame, running
    while running:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            with lock:
                latest_frame = frame.copy()
        time.sleep(1.0 / CAPTURE_FPS)

def process_thread():
    global latest_frame, processed_frame, running
    gesture_name = "None"
    confidence = 0.0
    all_probabilities = []

    # параметры
    dims = 3
    num_landmarks = 21
    expected_raw_features = num_landmarks * dims * 2  # 126 признаков для двух рук
    frame_id = 0
    while running:
        with lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.01)
            continue

        start_time = time.time()
        display_frame = frame.copy()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame_id % 2 == 0:
            results = hands.process(rgb_frame)  # каждый второй кадр
        frame_id += 1
        if not results.multi_hand_landmarks:
            # feature_buffer.clear()
            gesture_name = "None"
            confidence = 0.0
            # all_probabilities = []
            cv2.putText(display_frame, "No hands detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # создаём вектор признаков обеих рук (126)
            features = create_feature_vector(
                results.multi_hand_landmarks,
                results.multi_handedness
            )
            feature_buffer.append(features.squeeze(0).cpu().numpy())

            # отрисовка только первой руки
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
        image_pil = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)
        draw.text((8, 8), f"Gesture: {gesture_name}", font=font, fill=(255, 255, 255))
        draw.text((8, 35), f"Conf: {confidence:.2f}", font=font, fill=(255, 255, 255))
        draw.text((8, 62), f"Buffer: {len(feature_buffer)}/{sequence_length}", font=font, fill=(255, 255, 255))
        draw.text((8, 89), f"Hands: {len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0}",
                  font=font, fill=(255, 255, 255))
        if len(all_probabilities) > 0:
            y_pos = 116
            for i, (label, prob) in enumerate(zip(labels, all_probabilities)):
                color = (0, 255, 0) if label == gesture_name and confidence > CONF_THRESHOLD else (255, 255, 255)
                draw.text((8, y_pos), f"{label}: {prob:.2f}", font=font, fill=color)
                y_pos += 27

        display_frame = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        latency = (time.time() - start_time) * 1000
        with lock:
            processed_frame = (display_frame, latency, gesture_name, confidence, all_probabilities)
        new_frame_ready.set()


def display_thread():
    global processed_frame, running, out, latest_web_frame, last_gesture
    frame_count = 0
    last_print = time.time()
    gesture_history = deque(maxlen=10)
    last_gesture = "None"
    while running:
        if not new_frame_ready.wait(timeout=1.0):
            continue
        new_frame_ready.clear()

        with lock:
            if processed_frame is None:
                continue
            frame, latency, gesture, confidence, probabilities = processed_frame

        # Сохраняем видео
        if save_video and out is not None:
            out.write(frame)

        # Сохраняем кадр для web
        with web_lock:
            latest_web_frame = frame.copy()
        # История жестов
        if gesture != "None":
            gesture_history.append(gesture)
        
        # Печать статистики
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

        # Nextion
        if gesture != last_gesture:
            nextion_queue.put(gesture)
            last_gesture = gesture


# -------------------------
# Запуск потоков
# -------------------------

if not isTestRun:
    threads = [
        threading.Thread(target=capture_thread, daemon=True),
        threading.Thread(target=process_thread, daemon=True),
        threading.Thread(target=display_thread, daemon=True),
        threading.Thread(target=start_flask, daemon=True)
    ]

    for t in threads:
        t.start()

    print("[INFO] Все потоки запущены. Нажмите Ctrl+C для остановки.")
    try:
        last_nx_msg = None
        while running:
            try:
                msg = nextion_queue.get_nowait()
            except:
                msg = None
    
            if msg and msg != "None" and msg != last_nx_msg:  # ИГНОРИРУЕМ "None"
                nx.set_text("t2", msg)
                last_nx_msg = msg
                print(f"[NX] Отправлено: {msg}")
                time.sleep(1)
    
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[INFO] Остановка...")
        running = False
    finally:
        if save_video and out is not None:
            out.release()
            print("[INFO] Видеофайл сохранён: output.avi")
        cap.release()
        hands.close()
        print("[INFO] Завершено корректно.")