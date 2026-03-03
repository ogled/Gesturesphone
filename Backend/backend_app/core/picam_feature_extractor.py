import cv2
import numpy as np

from .feature_contract import FEATURE_CONTRACT


class FeatureExtractor:
    def __init__(self, smoothing_alpha=0.35):
        self.smoothing_alpha = smoothing_alpha
        self.prev_coords = None
        self.prev_velocity = None
        self.smoothed_hands = np.zeros((2, 21, 3), dtype=np.float32)
        self.hand_seen = np.zeros(2, dtype=np.bool_)

    def reset(self):
        self.prev_coords = None
        self.prev_velocity = None
        self.smoothed_hands.fill(0)
        self.hand_seen[:] = False

    def _smooth_hand_landmarks(self, raw_hands):
        for idx in range(2):
            is_present = float(np.abs(raw_hands[idx]).sum()) > 0
            if is_present:
                if self.hand_seen[idx]:
                    self.smoothed_hands[idx] = (
                        self.smoothing_alpha * raw_hands[idx]
                        + (1.0 - self.smoothing_alpha) * self.smoothed_hands[idx]
                    )
                else:
                    self.smoothed_hands[idx] = raw_hands[idx]
                self.hand_seen[idx] = True
            else:
                self.hand_seen[idx] = False
                self.smoothed_hands[idx] = 0.0

        return self.smoothed_hands.copy()

    def create_feature_vector(self, multi_hand_landmarks, multi_handedness):
        hands = np.zeros((2, 21, 3), dtype=np.float32)

        if multi_hand_landmarks and multi_handedness:
            for lm, hd in zip(multi_hand_landmarks, multi_handedness):
                idx = 0 if hd.classification[0].label == "Left" else 1
                coords = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
                hands[idx] = coords

        smoothed = self._smooth_hand_landmarks(hands)

        left_raw = smoothed[0]
        right_raw = smoothed[1]

        def norm_hand(hand):
            wrist = hand[0:1]
            centered = hand - wrist
            palm = centered[9]
            scale = np.linalg.norm(palm) + 1e-6
            return centered / scale

        left = norm_hand(left_raw)
        right = norm_hand(right_raw)

        coords = np.concatenate([left.ravel(), right.ravel()])

        if self.prev_coords is None:
            velocity = np.zeros_like(coords)
            acceleration = np.zeros_like(coords)
        else:
            velocity = coords - self.prev_coords
            if self.prev_velocity is None:
                acceleration = np.zeros_like(coords)
            else:
                acceleration = velocity - self.prev_velocity

        self.prev_coords = coords.copy()
        prev_velocity_for_curvature = self.prev_velocity.copy() if self.prev_velocity is not None else None
        self.prev_velocity = velocity.copy()

        motion_energy = np.array([[np.linalg.norm(velocity)]], dtype=np.float32)

        left_present = float(np.abs(left).sum() > 0)
        right_present = float(np.abs(right).sum() > 0)
        hands_count = np.array([[(left_present + right_present) / 2.0]], dtype=np.float32)

        inter_hand_dist = np.linalg.norm(left[0] - right[0]).reshape(1, 1)

        def polygon_area(points):
            x = points[:, 0]
            y = points[:, 1]
            x_shift = np.roll(x, -1)
            y_shift = np.roll(y, -1)
            return 0.5 * np.abs(np.sum(x * y_shift) - np.sum(y * x_shift))

        palm_points = [0, 1, 5, 9, 13, 17]
        left_area = np.array([[polygon_area(left[palm_points, :2])]], dtype=np.float32)
        right_area = np.array([[polygon_area(right[palm_points, :2])]], dtype=np.float32)

        key_pairs = [(0, 4), (0, 8), (0, 12), (0, 16), (0, 20), (4, 8), (8, 12), (12, 16), (16, 20)]

        def compute_dists(hand):
            return np.array([np.linalg.norm(hand[i] - hand[j]) for i, j in key_pairs]).reshape(1, -1)

        left_dists = compute_dists(left)
        right_dists = compute_dists(right)

        spread_left = np.linalg.norm(left[4] - left[20]).reshape(1, 1)
        spread_right = np.linalg.norm(right[4] - right[20]).reshape(1, 1)

        if prev_velocity_for_curvature is None or np.linalg.norm(velocity) == 0:
            curvature = np.zeros((1, 1), dtype=np.float32)
        else:
            vel_norm = velocity / (np.linalg.norm(velocity) + 1e-6)
            prev_norm = prev_velocity_for_curvature / (np.linalg.norm(prev_velocity_for_curvature) + 1e-6)
            curvature = np.linalg.norm(vel_norm - prev_norm).reshape(1, 1)

        palm_vec_left = left[9] - left[0]
        palm_dir_left = (palm_vec_left / (np.linalg.norm(palm_vec_left) + 1e-6)).reshape(1, 3)
        palm_vec_right = right[9] - right[0]
        palm_dir_right = (palm_vec_right / (np.linalg.norm(palm_vec_right) + 1e-6)).reshape(1, 3)

        feats = np.concatenate(
            [
                hands_count,
                coords.reshape(1, -1),
                velocity.reshape(1, -1),
                acceleration.reshape(1, -1),
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
                palm_dir_right,
            ],
            axis=1,
        )

        vector = feats.squeeze(0).astype(np.float32)
        if vector.shape[0] != FEATURE_CONTRACT.feature_size:
            raise ValueError(
                f"Unexpected feature vector size {vector.shape[0]}, expected {FEATURE_CONTRACT.feature_size}"
            )

        return vector, smoothed


def draw_smoothed_landmarks(frame, hands_coords, hand_connections):
    for hand_coords in hands_coords:
        if np.abs(hand_coords).sum() == 0:
            continue

        points = []
        for point in hand_coords:
            x = int(point[0] * frame.shape[1])
            y = int(point[1] * frame.shape[0])
            points.append((x, y))
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        for start_idx, end_idx in hand_connections:
            cv2.line(frame, points[start_idx], points[end_idx], (0, 0, 255), 2)
