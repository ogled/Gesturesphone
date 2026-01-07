import os
import csv
import shutil
from pathlib import Path
# ПУТИ

os.chdir(Path(__file__).resolve().parent)

DATASET_PATH = r"SlovoDS"
ALLOWED_GESTURES_CSV = r"AllowedGestures.csv"
OUTPUT_FOLDER = r"HowToSpeak"

ANNOTATIONS_PATH = os.path.join(DATASET_PATH, "annotations.csv")
TRAIN_FOLDER = os.path.join(DATASET_PATH, "train")
TEST_FOLDER = os.path.join(DATASET_PATH, "test")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

allowed_gestures = set()
with open(ALLOWED_GESTURES_CSV, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        allowed_gestures.add(row["text"].strip())

print("Разрешённые жесты:", allowed_gestures)

annotations = []
with open(ANNOTATIONS_PATH, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        if row["text"].strip() in allowed_gestures:
            annotations.append(row)

print(f"Найдено {len(annotations)} аннотаций с разрешёнными жестами.")

taken = set()
gesture_first_video = {}

for row in annotations:
    gesture = row["text"].strip()
    attachment_id = row["attachment_id"]

    if gesture not in gesture_first_video:
        gesture_first_video[gesture] = attachment_id

print(f"Будет выбрано {len(gesture_first_video)} видео (по 1 на жест).")

for gesture, attachment_id in gesture_first_video.items():
    filename = attachment_id + ".mp4"

    found_path = None

    path_train = os.path.join(TRAIN_FOLDER, filename)
    if os.path.exists(path_train):
        found_path = path_train

    path_test = os.path.join(TEST_FOLDER, filename)
    if os.path.exists(path_test):
        found_path = path_test

    if found_path:
        out_path = os.path.join(OUTPUT_FOLDER, f"{gesture}.mp4").replace("?", "")
        shutil.copy(found_path, out_path)
        print(f"✔ {gesture}: сохранено {out_path}")
    else:
        print(f"✖ Видео не найдено для жеста: {gesture}")
