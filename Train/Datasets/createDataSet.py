import argparse
import os
import csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import mediapipe as mp
import numpy as np
import torch
import tqdm

os.chdir(Path(__file__).resolve().parent)

# ======================================================
# Video → Torch Tensor [T, 2, 21, 3]
# ======================================================

def process_video(video_path: str, output_pt: str, target_fps: int):
    try:
        mp_hands = mp.solutions.hands

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return f"❌ Не удалось открыть {video_path}"

        original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_step = max(1, int(original_fps // target_fps))

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.25,
        ) as hands:

            frames = []
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_step != 0:
                    frame_idx += 1
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(frame_rgb)

                # [2, 21, 3] — Left, Right
                frame_data = np.zeros((2, 21, 3), dtype=np.float32)

                if result.multi_hand_landmarks and result.multi_handedness:
                    for lm, handedness in zip(
                        result.multi_hand_landmarks,
                        result.multi_handedness
                    ):
                        hand_id = 0 if handedness.classification[0].label == "Left" else 1
                        for i, p in enumerate(lm.landmark):
                            frame_data[hand_id, i] = (p.x, p.y, p.z)

                frames.append(frame_data)
                frame_idx += 1

        cap.release()

        if not frames:
            return f"⚠️ Пустое видео: {video_path}"

        data = np.stack(frames) 
        tensor = torch.from_numpy(data)

        os.makedirs(os.path.dirname(output_pt), exist_ok=True)
        torch.save(tensor, output_pt)

        return f"✅ {os.path.basename(video_path)} → {tensor.shape}"

    except Exception as e:
        return f"❌ Ошибка {os.path.basename(video_path)}: {e}"


# ======================================================
# MAIN
# ======================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", "--input", default="SlovoDS")
    parser.add_argument("-o", "--output", default="CreatedDS")
    parser.add_argument("-ag", "--allowedGestures", default="AllowedGestures.csv")
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()

    dataset_path = args.input
    output_path = args.output

    os.makedirs(output_path, exist_ok=True)

    # --------------------------------------------------
    # Allowed gestures
    # --------------------------------------------------
    allowed = set()
    with open(args.allowedGestures, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            allowed.add(row["text"].strip())

    # --------------------------------------------------
    # Filter annotations
    # --------------------------------------------------
    annotations = []
    with open(os.path.join(dataset_path, "annotations.csv"), encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["text"].strip() in allowed:
                annotations.append(row)

    if not annotations:
        print("Нет допустимых жестов")
        return

    with open(os.path.join(output_path, "annotations.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=annotations[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(annotations)

    allowed_ids = {a["attachment_id"] for a in annotations}

    # --------------------------------------------------
    # Collect videos
    # --------------------------------------------------
    jobs = []
    for split in ("train", "test"):
        src = os.path.join(dataset_path, split)
        if not os.path.exists(src):
            continue

        for f in os.listdir(src):
            if f.endswith(".mp4"):
                vid = os.path.splitext(f)[0]
                if vid in allowed_ids:
                    jobs.append((
                        os.path.join(src, f),
                        os.path.join(output_path, split, f"{vid}.pt")
                    ))

    print(f"Видео к обработке: {len(jobs)}")

    # --------------------------------------------------
    # Parallel processing
    # --------------------------------------------------
    with tqdm.tqdm(total=len(jobs), desc="Обработка") as bar:
        with ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as pool:
            futures = [
                pool.submit(process_video, v, o, args.fps)
                for v, o in jobs
            ]
            for f in as_completed(futures):
                bar.update(1)
                tqdm.tqdm.write(f.result())


if __name__ == "__main__":
    main()