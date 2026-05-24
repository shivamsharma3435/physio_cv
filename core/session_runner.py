"""
core/session_runner.py
Orchestrates the full real-time CV pipeline for a single session.
Ties together: preprocessor → edge detector → contour analyzer
→ pose estimator → angle calculator → feedback engine → logger.
"""

import cv2
import time
import numpy as np
from pathlib import Path
from typing import Optional, Callable

from core.preprocessor import FramePreprocessor
from core.edge_detector import EdgeDetector
from core.contour_analyzer import ContourAnalyzer
from analysis.pose_estimator import PoseEstimator
from analysis.angle_calculator import AngleCalculator
from analysis.feedback_engine import FeedbackEngine
from data.session_logger import SessionLogger


class SessionRunner:
    """
    Runs a physiotherapy monitoring session end-to-end.

    Usage:
        runner = SessionRunner()
        runner.run(patient_id="P001", exercise="squat", camera=0)
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.preprocessor  = FramePreprocessor()
        self.edge_detector  = EdgeDetector()
        self.contour_analyzer = ContourAnalyzer()
        self.pose_estimator = PoseEstimator()
        self.angle_calculator = AngleCalculator()
        self.feedback_engine  = FeedbackEngine(alert_cooldown_s=10.0)
        self.logger = SessionLogger(db_path or Path("data/sessions.db"))

    def run(
        self,
        patient_id: str,
        exercise: str,
        camera: int = 0,
        on_frame_callback: Optional[Callable] = None,
        show_debug: bool = False,
    ):
        """
        Main loop. Captures from `camera`, processes each frame,
        and drives the feedback + logging pipeline.

        Args:
            patient_id: Patient identifier string.
            exercise: Exercise key (e.g. 'squat', 'standing').
            camera: OpenCV camera index or path to video file.
            on_frame_callback: Optional fn(annotated_frame, angles, score)
                               called each frame — used by the GUI.
            show_debug: If True, shows intermediate CV windows.
        """
        cap = cv2.VideoCapture(camera)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera/video: {camera}")

        session_id = self.logger.start_session(patient_id, exercise)
        print(f"[SessionRunner] Session {session_id} started — {exercise} for {patient_id}")

        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (960, 720))
                annotated = frame.copy()

                # Stage 1: Preprocess
                pre = self.preprocessor.process(frame)

                # Stage 2: Edge detection
                edges = self.edge_detector.detect(pre["equalized"])

                # Stage 3: Contour analysis (body silhouette)
                body = self.contour_analyzer.find_body(
                    edges["dilated_edges"], pre["fg_mask"]
                )
                if body and show_debug:
                    annotated = self.contour_analyzer.draw(annotated, body)

                # Stage 4: Pose estimation
                skeleton = self.pose_estimator.estimate(frame)

                angles = {}
                score = 0
                joint_status = {}

                if skeleton.is_valid:
                    # Stage 5a: Angle calculation
                    angles = self.angle_calculator.compute_all(
                        skeleton.keypoints, exercise=exercise
                    )
                    score = self.angle_calculator.overall_score(angles)
                    joint_status = {n: j.status for n, j in angles.items()}

                    # Stage 5b: Pose overlay
                    annotated = self.pose_estimator.draw(
                        annotated, skeleton, joint_status
                    )

                    # Stage 5c: Feedback overlay
                    corrections = self.feedback_engine.get_active_corrections(angles)
                    annotated = self.feedback_engine.draw_overlay(
                        annotated, corrections, score
                    )
                    has_bad = any(j.status == "bad" for j in angles.values())
                    self.feedback_engine.alert(has_bad)

                    # Log every 3rd frame to avoid DB overload
                    if frame_count % 3 == 0:
                        self.logger.log_frame(session_id, score, angles)

                if show_debug:
                    cv2.imshow("Edges", edges["canny_edges"])
                    cv2.imshow("Sobel", edges["sobel_mag"])

                if on_frame_callback:
                    on_frame_callback(annotated, angles, score)
                else:
                    cv2.imshow("PhysioCV", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

                frame_count += 1

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.logger.end_session(session_id)
            self.pose_estimator.close()
            print(f"[SessionRunner] Session {session_id} ended — {frame_count} frames processed")

        return session_id