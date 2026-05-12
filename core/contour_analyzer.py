"""
core/contour_analyzer.py
Stage 3: Contour extraction and body silhouette analysis.
Finds the dominant body contour, computes convex hull,
bounding box, aspect ratio, and centroid for downstream use.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BodyContour:
    contour: np.ndarray
    hull: np.ndarray
    bounding_box: tuple        # (x, y, w, h)
    centroid: tuple            # (cx, cy)
    area: float
    aspect_ratio: float
    solidity: float            # area / hull_area — low = limbs extended
    defects: Optional[np.ndarray] = field(default=None)


class ContourAnalyzer:
    """
    Isolates and characterizes the patient's body contour.
    Uses area + aspect-ratio filtering to discard noise and
    background objects.
    """

    def __init__(
        self,
        min_area: int = 8000,
        aspect_min: float = 1.5,
        aspect_max: float = 6.0,
        solidity_min: float = 0.30,
    ):
        self.min_area = min_area
        self.aspect_min = aspect_min
        self.aspect_max = aspect_max
        self.solidity_min = solidity_min

    def _is_valid_body_contour(self, cnt: np.ndarray) -> bool:
        area = cv2.contourArea(cnt)
        if area < self.min_area:
            return False
        _, _, w, h = cv2.boundingRect(cnt)
        if w == 0:
            return False
        aspect = h / w
        if not (self.aspect_min <= aspect <= self.aspect_max):
            return False
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return False
        solidity = area / hull_area
        if solidity < self.solidity_min:
            return False
        return True

    def find_body(self, edge_img: np.ndarray, fg_mask: np.ndarray) -> Optional[BodyContour]:
        """
        Finds the largest valid body contour in the combined
        edge + foreground mask.
        """
        combined = cv2.bitwise_and(edge_img, edge_img, mask=fg_mask)
        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        valid = [c for c in contours if self._is_valid_body_contour(c)]
        if not valid:
            return None

        body = max(valid, key=cv2.contourArea)
        hull = cv2.convexHull(body)
        x, y, w, h = cv2.boundingRect(body)
        area = cv2.contourArea(body)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        M = cv2.moments(body)
        cx = int(M["m10"] / M["m00"]) if M["m00"] else x + w // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] else y + h // 2

        try:
            hull_idx = cv2.convexHull(body, returnPoints=False)
            defects = cv2.convexityDefects(body, hull_idx)
        except Exception:
            defects = None

        return BodyContour(
            contour=body,
            hull=hull,
            bounding_box=(x, y, w, h),
            centroid=(cx, cy),
            area=area,
            aspect_ratio=h / w if w > 0 else 0,
            solidity=solidity,
            defects=defects,
        )

    def draw(self, frame: np.ndarray, body: BodyContour) -> np.ndarray:
        """Draws contour + hull on the frame (in-place copy)."""
        out = frame.copy()
        cv2.drawContours(out, [body.contour], -1, (0, 200, 150), 2)
        cv2.drawContours(out, [body.hull], -1, (100, 200, 255), 1)
        cx, cy = body.centroid
        cv2.circle(out, (cx, cy), 5, (255, 200, 0), -1)
        return out
