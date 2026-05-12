"""
analysis/feedback_engine.py
Stage 5b: Real-time feedback generation.
Converts joint angle deviations into on-screen text corrections
and audio beep alerts. Throttles alerts to avoid noise fatigue.
"""

import time
import threading
import numpy as np
import cv2
from typing import Dict, List, Optional
from analysis.angle_calculator import JointAngle


try:
    import pygame
    pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False


CORRECTION_MESSAGES = {
    "left_knee":      {"warn": "Bend left knee more",     "bad": "Left knee angle critical"},
    "right_knee":     {"warn": "Bend right knee more",    "bad": "Right knee angle critical"},
    "left_elbow":     {"warn": "Straighten left arm",     "bad": "Left elbow severely bent"},
    "right_elbow":    {"warn": "Straighten right arm",    "bad": "Right elbow severely bent"},
    "left_shoulder":  {"warn": "Raise left arm higher",   "bad": "Left shoulder misaligned"},
    "right_shoulder": {"warn": "Raise right arm higher",  "bad": "Right shoulder misaligned"},
    "left_hip":       {"warn": "Adjust left hip angle",   "bad": "Left hip deviation"},
    "right_hip":      {"warn": "Adjust right hip angle",  "bad": "Right hip deviation"},
    "spine":          {"warn": "Keep back straighter",    "bad": "Severe spinal deviation!"},
}

STATUS_PRIORITY = {"bad": 2, "warn": 1, "good": 0}


class FeedbackEngine:
    """Generates real-time visual and audio posture corrections."""

    def __init__(self, alert_cooldown_s: float = 3.0):
        self.cooldown = alert_cooldown_s
        self._last_alert: Dict[str, float] = {}
        self._beep_thread: Optional[threading.Thread] = None

    def get_active_corrections(
        self, angles: Dict[str, JointAngle]
    ) -> List[str]:
        """Returns correction messages sorted by severity, respecting cooldown."""
        now = time.time()
        messages = []

        sorted_joints = sorted(
            angles.values(),
            key=lambda j: STATUS_PRIORITY.get(j.status, 0),
            reverse=True,
        )

        for joint in sorted_joints:
            if joint.status == "good":
                break
            last = self._last_alert.get(joint.name, 0)
            if now - last < self.cooldown:
                continue
            msg_map = CORRECTION_MESSAGES.get(joint.name, {})
            msg = msg_map.get(joint.status)
            if msg:
                messages.append((joint.status, msg))
                self._last_alert[joint.name] = now

        return messages

    def _play_beep(self, frequency: int = 880, duration_ms: int = 120):
        if not AUDIO_AVAILABLE:
            return
        sample_rate = 22050
        n_samples = int(sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, n_samples)
        mono = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
        # pygame mixer is stereo by default — shape must be (n_samples, 2)
        stereo = np.column_stack((mono, mono))
        sound = pygame.sndarray.make_sound(stereo)
        sound.play()

    def alert(self, has_bad: bool):
        """Plays audio alert if a 'bad' joint is detected and not cooling down."""
        if has_bad and (
            self._beep_thread is None or not self._beep_thread.is_alive()
        ):
            self._beep_thread = threading.Thread(
                target=self._play_beep, daemon=True
            )
            self._beep_thread.start()

    def draw_overlay(
        self,
        frame: np.ndarray,
        corrections: List[tuple],
        score: int,
    ) -> np.ndarray:
        """
        Draws correction messages and score HUD on the frame.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        score_color = (
            (29, 158, 117) if score >= 80
            else (0, 165, 239) if score >= 50
            else (48, 90, 216)
        )
        cv2.putText(
            out, f"Score: {score}", (w - 150, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, score_color, 2, cv2.LINE_AA,
        )

        y0 = 80
        for status, msg in corrections[:4]:
            color = (48, 90, 216) if status == "bad" else (0, 165, 239)
            cv2.rectangle(out, (10, y0 - 22), (10 + len(msg) * 11 + 10, y0 + 6), (0, 0, 0), -1)
            cv2.putText(
                out, msg, (15, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
            )
            y0 += 34

        return out