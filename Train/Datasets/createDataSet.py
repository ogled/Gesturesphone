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
# Video → Torch Tensor [T, 2 - [2, 21, 3]  hands
#                          | - [24, 3]  pose
# ======================================================

POSE_UPPER_BODY_INDICES = list(range(0, 24))

def create_hand_landmarker(hand_landmarker_path: str):
    base = mp.tasks.BaseOptions(model_asset_path=hand_landmarker_path)
    return mp.tasks.vision.HandLandmarker.create_from_options(
        mp.tasks.vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4
        )
    )

def create_pose_landmarker(pose_landmarker_path: str):
    base = mp.tasks.BaseOptions(model_asset_path=pose_landmarker_path)
    return mp.tasks.vision.PoseLandmarker.create_from_options(
        mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.6,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4
        )
    )

def norm_hand(hand):
    if np.sum(np.abs(hand)) < 1e-6:
        return hand

    wrist = hand[0]
    hand = hand - wrist

    scale = np.linalg.norm(hand[9])
    scale = max(scale, 1e-6)

    return hand / scale


def norm_hands(hands):
    return np.stack([
        norm_hand(hands[0]),
        norm_hand(hands[1])
    ])


def norm_pose(pose):
    xyz = pose[:, :3]
    vis = pose[:, 3:4]

    center = (xyz[11] + xyz[12]) / 2
    xyz = xyz - center

    scale = np.linalg.norm(xyz[11] - xyz[12])
    scale = max(scale, 1e-6)

    xyz = xyz / scale

    return np.concatenate([
        xyz,
        vis
    ], axis=1)


def normalize_frame(hands, pose):
    hands = norm_hands(hands)
    pose = norm_pose(pose)

    return np.concatenate([
        hands.flatten(),
        pose.flatten()
    ]).astype(np.float32)

def process_video(video_path: str, output_pt: str, hand_landmarker_path: str, pose_landmarker_path: str, target_fps: int):
    try:
        hand_landmarker = create_hand_landmarker(hand_landmarker_path)
        pose_landmarker = create_pose_landmarker(pose_landmarker_path)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_step = max(1, int(fps // target_fps))
        if cap.isOpened():         
            frames = []
            frame_idx = 0
            last_timestamp = -1
            
            while True:
                if frame_idx % frame_step != 0:
                    skip_count = frame_step - 1
                    for _ in range(skip_count):
                        ret = cap.grab()
                        if not ret:
                            break
                    frame_idx += skip_count
                    continue
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_step != 0:
                    frame_idx += 1
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                timestamp = int((frame_idx / fps) * 1000)
                if timestamp <= last_timestamp:
                    timestamp = last_timestamp + 1
                last_timestamp = timestamp
                
                try:
                    pose_result = pose_landmarker.detect_for_video(mp_image, timestamp)
                    hand_result = hand_landmarker.detect_for_video(mp_image, timestamp)
                except Exception as e:
                    print(f"Ошибка детекции на кадре {frame_idx}: {e}")
                    frame_idx += 1
                    continue
                
                # [2, 21, 3] — Left, Right
                frame_hands_data = np.empty((2,21,3), np.float32)
                frame_hands_data.fill(0)

                # [24, 4]
                frame_pose_data = np.empty((24,4), np.float32)
                frame_pose_data.fill(0)

                if hand_result and hand_result.hand_landmarks:
                    for hand_idx, hand_landmarks in enumerate(hand_result.hand_landmarks[:2]):
                        for i, lm in enumerate(hand_landmarks[:21]):
                            frame_hands_data[hand_idx, i] = (lm.x, lm.y, lm.z)

                if pose_result and pose_result.pose_landmarks:
                    for i, idx in enumerate(POSE_UPPER_BODY_INDICES):
                        if idx < len(pose_result.pose_landmarks[0]):
                            lm = pose_result.pose_landmarks[0][idx]
                            frame_pose_data[i] = (
                                lm.x,
                                lm.y,
                                lm.z,
                                lm.visibility
                            )

                frame_data = normalize_frame(
                    frame_hands_data,
                    frame_pose_data
                )

                frames.append(frame_data)
                frame_idx += 1

        cap.release()
        hand_landmarker.close()
        pose_landmarker.close()

        if not frames:
            return f"⚠️ Пустое видео: {video_path}"

        tensor = torch.from_numpy(np.stack(frames))

        os.makedirs(os.path.dirname(output_pt), exist_ok=True)
        torch.save(tensor, output_pt)

        return

    except Exception as e:
        return f"❌ Ошибка {os.path.basename(video_path)}: {e}"


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# ======================================================
# MAIN
# ======================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", "--input", default="SlovoDS")
    parser.add_argument("-o", "--output", default="CreatedDS")
    parser.add_argument("-ag", "--allowedGestures", default="AllowedGestures.csv")
    parser.add_argument("--hand_landmarker_path", default="../../Assets/hand_landmarker.task")
    parser.add_argument("--pose_landmarker_path", default="../../Assets/pose_landmarker.task")
    parser.add_argument("--fps", type=int, default=12)
    
    args = parser.parse_args()

    dataset_path = args.input
    output_path = args.output
    hand_landmarker_path = args.hand_landmarker_path
    pose_landmarker_path = args.pose_landmarker_path
    workers = min(4, os.cpu_count() - 1)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)

    os.makedirs(output_path, exist_ok=True)

    if not os.path.isfile(hand_landmarker_path):
        print("hand_landmarker.task not found")
        return
    
    if not os.path.isfile(pose_landmarker_path):
        print("pose_landmarker.task not found")
        return
    
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
    
    executor = ProcessPoolExecutor(max_workers=workers)

    try:
        with tqdm.tqdm(total=len(jobs), desc="Обработка", smoothing=0.1) as bar:

            for batch in chunks(jobs, workers * 2):

                futures = [
                    executor.submit(
                        process_video,
                        v,
                        o,
                        hand_landmarker_path,
                        pose_landmarker_path,
                        args.fps
                    )
                    for v, o in batch
                ]

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            tqdm.tqdm.write(result)
                    except Exception as e:
                        tqdm.tqdm.write(f"Ошибка: {e}")

                    bar.update(1)

    except KeyboardInterrupt:
        print("\nОстановка...")

        executor.shutdown(
            wait=False,
            cancel_futures=True
        )

        import psutil
        for p in psutil.Process().children(recursive=True):
            p.kill()

        raise SystemExit

    finally:
        executor.shutdown(wait=False)
                
if __name__ == "__main__":
    main()