"""
analysis/pose_estimator.py
Stage 4: Pose estimation with MediaPipe BlazePose.
Extracts 33 landmarks (we use 17 COCO-compatible keypoints),
applies Kalman smoothing, and returns a structured Skeleton object.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from dataclasses import dataclass
from typing import Optional, Dict


COCO_LANDMARK_MAP = {
    "nose":           0,
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_elbow":     13,
    "right_elbow":    14,
    "left_wrist":     15,
    "right_wrist":    16,
    "left_hip":       23,
    "right_hip":      24,
    "left_knee":      25,
    "right_knee":     26,
    "left_ankle":     27,
    "right_ankle":    28,
    "left_ear":        7,
    "right_ear":       8,
    "left_eye":        2,
    "right_eye":       5,
}

SKELETON_CONNECTIONS = [
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip",       "right_hip"),
    ("left_hip",       "left_knee"),
    ("left_knee",      "left_ankle"),
    ("right_hip",      "right_knee"),
    ("right_knee",     "right_ankle"),
]


@dataclass
class Skeleton:
    keypoints: Dict[str, np.ndarray]   # name → [x, y, z, visibility]
    is_valid: bool
    confidence: float

    def get_xy(self, name: str) -> Optional[np.ndarray]:
        kp = self.keypoints.get(name)
        return kp[:2] if kp is not None else None


class KalmanSmoother:
    """Per-keypoint Kalman filter for temporal smoothing."""

    def __init__(self):
        self.filters: Dict[str, cv2.KalmanFilter] = {}

    def _make_filter(self) -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        kf.transitionMatrix  = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        kf.processNoiseCov   = np.eye(4, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        return kf

    def smooth(self, name: str, xy: np.ndarray) -> np.ndarray:
        if name not in self.filters:
            self.filters[name] = self._make_filter()
        kf = self.filters[name]
        measurement = np.array([[np.float32(xy[0])], [np.float32(xy[1])]])
        kf.correct(measurement)
        predicted = kf.predict()
        return np.array([predicted[0, 0], predicted[1, 0]])


class PoseEstimator:
    """
    Wraps MediaPipe Pose Landmarker (Tasks API) with Kalman smoothing.

    Requires the pose_landmarker model file — download with:
        wget -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task \
             -O models/pose_landmarker_full.task
    """

    MODEL_PATH = "models/pose_landmarker_full.task"

    def __init__(self, min_confidence: float = 0.5):
        options = PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=self.MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        self.smoother = KalmanSmoother()

    def estimate(self, frame: np.ndarray) -> Skeleton:
        """
        Runs Pose Landmarker on the BGR frame and returns a Skeleton.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return Skeleton(keypoints={}, is_valid=False, confidence=0.0)

        # Tasks API returns a list of poses; take the first
        lm = result.pose_landmarks[0]
        h, w = frame.shape[:2]
        keypoints = {}
        visibilities = []

        for name, idx in COCO_LANDMARK_MAP.items():
            pt = lm[idx]
            raw_xy = np.array([pt.x * w, pt.y * h])
            smoothed = self.smoother.smooth(name, raw_xy)
            visibility = getattr(pt, "visibility", 1.0)
            keypoints[name] = np.array([smoothed[0], smoothed[1], pt.z, visibility])
            visibilities.append(visibility)

        confidence = float(np.mean(visibilities))
        return Skeleton(keypoints=keypoints, is_valid=True, confidence=confidence)

    def draw(
        self,
        frame: np.ndarray,
        skeleton: Skeleton,
        joint_status: Optional[Dict[str, str]] = None,
    ) -> np.ndarray:
        """
        Draws skeleton on frame with color-coded joints:
          green  = correct, orange = warning, red = deviation
        """
        STATUS_COLORS = {
            "good": (29, 158, 117),
            "warn": (0, 165, 239),
            "bad":  (48, 90, 216),
        }
        DEFAULT_COLOR = (180, 180, 180)

        out = frame.copy()
        kp = skeleton.keypoints

        for a_name, b_name in SKELETON_CONNECTIONS:
            a = skeleton.get_xy(a_name)
            b = skeleton.get_xy(b_name)
            if a is None or b is None:
                continue
            col = STATUS_COLORS.get(
                (joint_status or {}).get(a_name, "good"), DEFAULT_COLOR
            )
            cv2.line(out, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), col, 2)

        for name, pt in kp.items():
            x, y = int(pt[0]), int(pt[1])
            status = (joint_status or {}).get(name, "good")
            color = STATUS_COLORS.get(status, DEFAULT_COLOR)
            cv2.circle(out, (x, y), 6, color, -1)
            cv2.circle(out, (x, y), 6, (255, 255, 255), 1)

        return out

    def close(self):
        self.landmarker.close()