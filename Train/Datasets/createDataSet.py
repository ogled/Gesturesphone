import argparse
import mediapipe as mp
import cv2
import tqdm
import os
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import csv
import os
import cv2
import mediapipe as mp

def processVideo(video_path, output_csv, target_fps=None, round_decimals=5):
    try:
        if os.path.exists(output_csv):
            os.remove(output_csv)

        mp_hands = mp.solutions.hands
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.25,
            model_complexity=0
        ) as hands:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return f"Ошибка открытия: {video_path}"

            original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_step = max(1, round(original_fps / target_fps)) if target_fps and target_fps < original_fps else 1

            landmarks_data = []
            frame_count = 0

            while True:
                success, frame = cap.read()
                if not success:
                    break

                if frame_count % frame_step != 0:
                    frame_count += 1
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                hands_dict = {"Left": [0.0] * (21 * 3), "Right": [0.0] * (21 * 3)}

                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        label = hand_handedness.classification[0].label
                        coords = []
                        for lm in hand_landmarks.landmark:
                            coords.extend([round(lm.x, round_decimals),
                                           round(lm.y, round_decimals),
                                           round(lm.z, round_decimals)])
                        hands_dict[label] = coords

                frame_landmarks = hands_dict["Left"] + hands_dict["Right"]
                landmarks_data.append([frame_count] + frame_landmarks)
                frame_count += 1

            cap.release()
            del cap
            cv2.destroyAllWindows()

            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            with open(output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = ["frame"]
                for hand in range(1, 3):
                    for i in range(21):
                        header.extend([f"x{i}_{hand}", f"y{i}_{hand}"])
                writer.writerow(header)
                csv_rows = []
                for row in landmarks_data:
                    csv_row = [row[0]]
                    coords = row[1:]
                    csv_row.extend([round(coords[i], round_decimals) for i in range(len(coords)) if i % 3 != 2])
                    csv_rows.append(csv_row)
                writer.writerows(csv_rows)

            npz_path = os.path.splitext(output_csv)[0] + ".npz"
            np_data = np.array(landmarks_data, dtype=np.float32)
            np_data = np_data[:, 1:]
            np_data = np_data[:, :126] 
            np.savez_compressed(npz_path, data=np_data)

            return f"Готово: {output_csv} / {npz_path}"

    except Exception as e:
        return f"Ошибка обработки {os.path.basename(video_path)}: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', '--input', default=r"Datasets\\SlovoDS", help='Путь к датасету SlovoDS')
    parser.add_argument('-o', '--output', default="Datasets\\CreatedDS", help='Папка для сохранения обработанных данных')
    parser.add_argument('-ag', '--allowedGestures', default="Datasets\\AllowedGestures.csv", help='CSV с разрешёнными жестами')
    parser.add_argument('--fps', type=int, default=6) 
    args = parser.parse_args()
    
    dataset_path = args.input.strip('"\'')
    output_folder = args.output.strip('"\'')
    annotations_path = os.path.join(dataset_path, "annotations.csv")
    allowed_gestures_path = args.allowedGestures.strip('"\'')
    new_annotations_path = os.path.join(output_folder, "annotations.csv")

    os.makedirs(output_folder, exist_ok=True)

    allowed_gestures = set()
    with open(allowed_gestures_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gesture = row["text"].strip()
            allowed_gestures.add(gesture)
    print(f"Разрешённые жесты: {allowed_gestures}")

    allowed_annotations = []
    with open(annotations_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            gesture = row["text"].strip()
            if gesture in allowed_gestures:
                allowed_annotations.append(row)

    if allowed_annotations:
        with open(new_annotations_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=allowed_annotations[0].keys(), delimiter='\t')
            writer.writeheader()
            writer.writerows(allowed_annotations)
        print(f"Создан новый annotations.csv с {len(allowed_annotations)} строками.")
    else:
        print("Нет допустимых аннотаций. Новый файл не создан.")
        return

    allowed_ids = {row["attachment_id"] for row in allowed_annotations}

    video_paths = []
    for split in ["train", "test"]:
        folder = os.path.join(dataset_path, split)
        if not os.path.exists(folder):
            continue
        for f in os.listdir(folder):
            if f.endswith(".mp4"):
                attachment_id = os.path.splitext(f)[0]
                if attachment_id in allowed_ids:
                    video_path = os.path.join(folder, f)
                    output_csv = os.path.join(output_folder, split, f"{attachment_id}.csv")
                    video_paths.append((video_path, output_csv))

    print(f"Видео для обработки: {len(video_paths)}")

    for root, _, files in os.walk(output_folder):
        for f in files:
            if f.endswith(".csv") and f != "annotations.csv":
                attachment_id = os.path.splitext(f)[0]
                if attachment_id not in allowed_ids:
                    os.remove(os.path.join(root, f))
                    print(f"Удалён неразрешённый файл: {f}")

    with tqdm.tqdm(total=len(video_paths), desc="Обработка видео") as pbar:
        with ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
            futures = {executor.submit(processVideo, v, o, args.fps): v for v, o in video_paths}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    result = f"Ошибка в пуле: {e}"
                pbar.update(1)
                tqdm.tqdm.write(result)


if __name__ == "__main__":
    main()
