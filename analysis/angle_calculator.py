"""
analysis/angle_calculator.py
Stage 5a: Joint angle computation using vector dot-product.
Calculates angles at every major joint and checks them
against per-exercise reference thresholds.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class JointAngle:
    name: str
    angle_deg: float
    status: str            # "good" | "warn" | "bad"
    deviation_deg: float   # signed deviation from ideal


# Reference angles per exercise type
# Format: joint_name → (ideal_deg, tolerance_warn, tolerance_bad)
EXERCISE_THRESHOLDS: Dict[str, Dict[str, Tuple[float, float, float]]] = {
    "squat": {
        "left_knee":  (90.0, 15.0, 30.0),
        "right_knee": (90.0, 15.0, 30.0),
        "left_hip":   (90.0, 15.0, 30.0),
        "right_hip":  (90.0, 15.0, 30.0),
        "spine":      (180.0, 10.0, 20.0),
    },
    "shoulder_raise": {
        "left_shoulder":  (90.0, 10.0, 25.0),
        "right_shoulder": (90.0, 10.0, 25.0),
        "left_elbow":     (180.0, 15.0, 30.0),
        "right_elbow":    (180.0, 15.0, 30.0),
    },
    "standing": {
        "left_knee":  (180.0, 8.0, 18.0),
        "right_knee": (180.0, 8.0, 18.0),
        "spine":      (180.0, 8.0, 15.0),
        "left_hip":   (180.0, 8.0, 18.0),
        "right_hip":  (180.0, 8.0, 18.0),
    },
}


def angle_between(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    """
    Returns the angle at `vertex` formed by points a–vertex–b,
    in degrees [0, 180].
    """
    va = a - vertex
    vb = b - vertex
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a < 1e-6 or norm_b < 1e-6:
        return 0.0
    cos_theta = np.dot(va, vb) / (norm_a * norm_b)
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


class AngleCalculator:
    """
    Computes and evaluates joint angles from skeleton keypoints.
    """

    JOINT_TRIPLETS: Dict[str, Tuple[str, str, str]] = {
        "left_elbow":     ("left_shoulder",  "left_elbow",   "left_wrist"),
        "right_elbow":    ("right_shoulder", "right_elbow",  "right_wrist"),
        "left_shoulder":  ("left_elbow",     "left_shoulder","left_hip"),
        "right_shoulder": ("right_elbow",    "right_shoulder","right_hip"),
        "left_hip":       ("left_shoulder",  "left_hip",     "left_knee"),
        "right_hip":      ("right_shoulder", "right_hip",    "right_knee"),
        "left_knee":      ("left_hip",       "left_knee",    "left_ankle"),
        "right_knee":     ("right_hip",      "right_knee",   "right_ankle"),
    }

    def compute_all(
        self,
        keypoints: Dict[str, np.ndarray],
        exercise: str = "standing",
    ) -> Dict[str, JointAngle]:
        """
        Computes angles for all defined joints and evaluates
        them against the exercise thresholds.
        """
        thresholds = EXERCISE_THRESHOLDS.get(exercise, EXERCISE_THRESHOLDS["standing"])
        results = {}

        for joint_name, (a_key, v_key, b_key) in self.JOINT_TRIPLETS.items():
            if not all(k in keypoints for k in (a_key, v_key, b_key)):
                continue
            a  = keypoints[a_key][:2]
            v  = keypoints[v_key][:2]
            b  = keypoints[b_key][:2]
            angle = angle_between(a, v, b)

            if joint_name in thresholds:
                ideal, warn_tol, bad_tol = thresholds[joint_name]
                deviation = abs(angle - ideal)
                if deviation <= warn_tol:
                    status = "good"
                elif deviation <= bad_tol:
                    status = "warn"
                else:
                    status = "bad"
                signed_dev = angle - ideal
            else:
                status = "good"
                signed_dev = 0.0

            results[joint_name] = JointAngle(
                name=joint_name,
                angle_deg=round(angle, 1),
                status=status,
                deviation_deg=round(signed_dev, 1),
            )

        results["spine"] = self._spine_angle(keypoints, thresholds)
        return results

    def _spine_angle(
        self, keypoints: Dict[str, np.ndarray], thresholds: Dict
    ) -> JointAngle:
        """Approximates spine straightness via shoulder-hip-knee midpoints."""
        ls = keypoints.get("left_shoulder")
        rs = keypoints.get("right_shoulder")
        lh = keypoints.get("left_hip")
        rh = keypoints.get("right_hip")
        lk = keypoints.get("left_knee")
        rk = keypoints.get("right_knee")

        if any(v is None for v in [ls, rs, lh, rh, lk, rk]):
            return JointAngle("spine", 0, "good", 0)

        shoulder_mid = ((ls[:2] + rs[:2]) / 2)
        hip_mid      = ((lh[:2] + rh[:2]) / 2)
        knee_mid     = ((lk[:2] + rk[:2]) / 2)

        angle = angle_between(shoulder_mid, hip_mid, knee_mid)
        ideal, warn_tol, bad_tol = thresholds.get("spine", (180.0, 10.0, 20.0))
        deviation = abs(angle - ideal)
        status = "good" if deviation <= warn_tol else ("warn" if deviation <= bad_tol else "bad")

        return JointAngle("spine", round(angle, 1), status, round(angle - ideal, 1))

    def overall_score(self, angles: Dict[str, JointAngle]) -> int:
        """Returns a posture score 0–100 based on joint statuses."""
        if not angles:
            return 0
        weights = {"good": 100, "warn": 60, "bad": 20}
        scores = [weights[j.status] for j in angles.values()]
        return int(np.mean(scores))
