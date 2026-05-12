"""
core/edge_detector.py
Stage 2: Edge detection using Canny and Sobel operators.
Auto-thresholding via Otsu's method ensures robustness across
varying lighting conditions in physiotherapy clinics.
"""

import cv2
import numpy as np


class EdgeDetector:
    """Extracts body edges using classical gradient-based techniques."""

    def __init__(self, canny_low_ratio=0.5, canny_high_ratio=1.5):
        self.canny_low_ratio = canny_low_ratio
        self.canny_high_ratio = canny_high_ratio

    def auto_canny(self, equalized: np.ndarray) -> np.ndarray:
        """
        Canny with automatic threshold selection via Otsu's method.
        Adapts to varying brightness without manual tuning.
        """
        otsu_thresh, _ = cv2.threshold(
            equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        low = int(otsu_thresh * self.canny_low_ratio)
        high = int(otsu_thresh * self.canny_high_ratio)
        return cv2.Canny(equalized, low, high, apertureSize=3, L2gradient=True)

    def sobel_magnitude(self, equalized: np.ndarray) -> np.ndarray:
        """
        Sobel gradient magnitude — used to verify edge directionality.
        Useful for detecting spine/limb orientation.
        """
        sobel_x = cv2.Sobel(equalized, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(equalized, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    def sobel_direction(self, equalized: np.ndarray) -> np.ndarray:
        """Returns gradient direction angle map (radians)."""
        sobel_x = cv2.Sobel(equalized, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(equalized, cv2.CV_64F, 0, 1, ksize=3)
        return np.arctan2(np.abs(sobel_y), np.abs(sobel_x))

    def dilate_edges(self, edges: np.ndarray, iterations: int = 1) -> np.ndarray:
        """Dilates thin edges to improve contour connectivity."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.dilate(edges, kernel, iterations=iterations)

    def detect(self, equalized: np.ndarray) -> dict:
        """
        Full edge detection stage.

        Returns:
            dict with canny_edges, sobel_mag, sobel_dir, dilated_edges
        """
        canny = self.auto_canny(equalized)
        sobel_mag = self.sobel_magnitude(equalized)
        sobel_dir = self.sobel_direction(equalized)
        dilated = self.dilate_edges(canny)

        return {
            "canny_edges": canny,
            "sobel_mag": sobel_mag,
            "sobel_dir": sobel_dir,
            "dilated_edges": dilated,
        }
