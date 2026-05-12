"""
core/preprocessor.py
Stage 1: Frame capture and preprocessing pipeline.
Handles grayscale conversion, Gaussian blur, CLAHE equalization,
and background subtraction for patient isolation.
"""

import cv2
import numpy as np


class FramePreprocessor:
    """Prepares raw video frames for edge and contour analysis."""

    def __init__(self, blur_kernel=(5, 5), clahe_clip=2.0, clahe_tile=(8, 8)):
        self.blur_kernel = blur_kernel
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=40, detectShadows=False
        )

    def to_gray(self, frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def apply_blur(self, gray: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(gray, self.blur_kernel, 0)

    def equalize(self, blurred: np.ndarray) -> np.ndarray:
        """CLAHE equalization — handles uneven clinic lighting."""
        return self.clahe.apply(blurred)

    def get_foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        """MOG2 background subtraction to isolate the patient."""
        mask = self.bg_subtractor.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    def process(self, frame: np.ndarray) -> dict:
        """
        Full preprocessing pipeline.

        Returns:
            dict with keys: original, gray, blurred, equalized, fg_mask
        """
        gray = self.to_gray(frame)
        blurred = self.apply_blur(gray)
        equalized = self.equalize(blurred)
        fg_mask = self.get_foreground_mask(frame)

        return {
            "original": frame,
            "gray": gray,
            "blurred": blurred,
            "equalized": equalized,
            "fg_mask": fg_mask,
        }
