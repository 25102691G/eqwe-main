"""Tongue image quality control helpers.

This module implements a paper-compatible five-state quality gate for tongue
diagnosis images. The original paper uses a ResNet34 transfer-learning
classifier with five labels:

1. `non_face`
2. `tongue_not_straight`
3. `eyes_closed`
4. `tongue_not_straight_and_eyes_closed`
5. `qualified`

The repository does not currently include the paper's trained weights, so the
runtime path below uses engineering heuristics built on top of the existing face
mesh and tongue segmentation outputs while keeping the same label taxonomy.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
import math
from typing import Any

import cv2
import numpy as np

FaceRect = tuple[int, int, int, int]
Point = tuple[int, int]
FaceDetector = Callable[[np.ndarray], Sequence[FaceRect]]
LandmarkDetector = Callable[[np.ndarray], Sequence[Point] | None]

QUALITY_LABEL_SUMMARIES: dict[str, str] = {
    "non_face": "No reliable frontal face was detected in the image.",
    "tongue_not_straight": "The tongue is not extended straight enough for diagnosis.",
    "eyes_closed": "The subject's eyes appear closed.",
    "tongue_not_straight_and_eyes_closed": (
        "The tongue is not extended straight and the eyes appear closed."
    ),
    "qualified": "The image meets the five-state tongue diagnosis quality gate.",
}

PAPER_REFERENCE: dict[str, str] = {
    "doi": "https://doi.org/10.3233/THC-248018",
    "pmc": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11191470/",
    "implementation_mode": "paper_compatible_heuristic",
}

LEFT_EYE_INDICES: tuple[int, int, int, int, int, int] = (33, 133, 159, 145, 158, 153)
RIGHT_EYE_INDICES: tuple[int, int, int, int, int, int] = (362, 263, 386, 374, 387, 373)
MOUTH_CENTER_INDICES: tuple[int, ...] = (61, 291, 13, 14, 17)
FACE_HEIGHT_INDICES: tuple[int, int] = (10, 152)

EYE_OPEN_THRESHOLD = 0.18
TONGUE_ANGLE_THRESHOLD_DEG = 18.0
TONGUE_TIP_OFFSET_THRESHOLD = 0.22
TONGUE_EXTENSION_THRESHOLD = 0.08
TONGUE_HEIGHT_WIDTH_THRESHOLD = 0.65
TONGUE_MASK_AREA_THRESHOLD = 0.008


@dataclass(frozen=True)
class EyeMetrics:
    """Eye openness metrics extracted from face landmarks."""

    left_ear: float | None
    right_ear: float | None
    average_ear: float | None
    threshold: float
    eyes_open: bool | None
    reliable: bool


@dataclass(frozen=True)
class TongueMetrics:
    """Tongue alignment metrics extracted from the segmentation mask."""

    detected: bool
    mask_area_ratio: float
    bbox_area_ratio: float
    height_width_ratio: float
    angle_from_vertical_deg: float | None
    tip_offset_ratio: float | None
    extension_ratio: float | None
    tongue_straight: bool


def _distance(a: Point, b: Point) -> float:
    """Return the Euclidean distance between two 2D points."""

    return math.dist(a, b)


def _safe_point(landmarks: Sequence[Point], index: int) -> Point | None:
    """Return one landmark when the index exists."""

    if index < 0 or index >= len(landmarks):
        return None
    return landmarks[index]


def _largest_face_rect(rects: Sequence[FaceRect]) -> FaceRect | None:
    """Return the largest face rectangle by area."""

    if not rects:
        return None
    return max(rects, key=lambda rect: rect[2] * rect[3])


def _normalize_face_rects(rects: Sequence[FaceRect] | np.ndarray | None) -> list[FaceRect]:
    """Convert face rectangle outputs into a plain Python list."""

    if rects is None:
        return []

    normalized: list[FaceRect] = []
    for rect in rects:
        if len(rect) < 4:
            continue
        normalized.append((int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])))
    return normalized


def _normalize_landmarks(
    landmarks: Sequence[Point] | np.ndarray | None,
) -> list[Point] | None:
    """Convert landmark outputs into a plain Python list."""

    if landmarks is None:
        return None

    normalized: list[Point] = []
    for point in landmarks:
        if len(point) < 2:
            continue
        normalized.append((int(point[0]), int(point[1])))
    return normalized or None


def _compute_eye_aspect_ratio(
    landmarks: Sequence[Point],
    indices: tuple[int, int, int, int, int, int],
) -> float | None:
    """Compute the classic eye aspect ratio from six face mesh landmarks."""

    p1 = _safe_point(landmarks, indices[0])
    p4 = _safe_point(landmarks, indices[1])
    p2 = _safe_point(landmarks, indices[2])
    p6 = _safe_point(landmarks, indices[3])
    p3 = _safe_point(landmarks, indices[4])
    p5 = _safe_point(landmarks, indices[5])
    if None in {p1, p2, p3, p4, p5, p6}:
        return None

    assert p1 is not None
    assert p2 is not None
    assert p3 is not None
    assert p4 is not None
    assert p5 is not None
    assert p6 is not None

    horizontal = _distance(p1, p4)
    if horizontal <= 1e-6:
        return None
    return ((_distance(p2, p6) + _distance(p3, p5)) / 2.0) / horizontal


def _face_height(landmarks: Sequence[Point] | None, face_rect: FaceRect | None) -> float | None:
    """Estimate face height from landmarks first and Haar face box second."""

    if landmarks is not None:
        top = _safe_point(landmarks, FACE_HEIGHT_INDICES[0])
        bottom = _safe_point(landmarks, FACE_HEIGHT_INDICES[1])
        if top is not None and bottom is not None:
            return max(float(abs(bottom[1] - top[1])), 1.0)
    if face_rect is not None:
        return max(float(face_rect[3]), 1.0)
    return None


def _mouth_metrics(
    landmarks: Sequence[Point] | None,
) -> tuple[float | None, float | None, float | None]:
    """Return mouth center X, mouth bottom Y, and mouth width."""

    if landmarks is None:
        return None, None, None

    mouth_points = [_safe_point(landmarks, index) for index in MOUTH_CENTER_INDICES]
    if any(point is None for point in mouth_points):
        return None, None, None

    left = _safe_point(landmarks, 61)
    right = _safe_point(landmarks, 291)
    if left is None or right is None:
        return None, None, None

    assert left is not None
    assert right is not None
    valid_points = [point for point in mouth_points if point is not None]
    center_x = float(sum(point[0] for point in valid_points) / len(valid_points))
    bottom_y = float(max(point[1] for point in valid_points))
    mouth_width = max(_distance(left, right), 1.0)
    return center_x, bottom_y, mouth_width


def _to_binary_mask(mask: np.ndarray | None) -> np.ndarray | None:
    """Normalize a segmentation mask to a 2D binary array."""

    if mask is None:
        return None

    if mask.ndim == 3:
        mask_2d = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_2d = mask.copy()

    binary_mask = np.where(mask_2d > 0, 255, 0).astype(np.uint8)
    if not np.any(binary_mask):
        return None
    return binary_mask


class TongueQualityControlService:
    """Assess tongue image quality using the paper's five-state taxonomy."""

    def __init__(
        self,
        *,
        face_detector: FaceDetector | None = None,
        landmark_detector: LandmarkDetector | None = None,
    ) -> None:
        self._face_detector = face_detector
        self._landmark_detector = landmark_detector

    def assess(
        self,
        image: np.ndarray,
        *,
        tongue_bbox: dict[str, int] | None = None,
        tongue_mask: np.ndarray | None = None,
        face_rects: Sequence[FaceRect] | np.ndarray | None = None,
        landmarks: Sequence[Point] | np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Return the five-state tongue quality assessment payload."""

        normalized_landmarks = _normalize_landmarks(
            landmarks if landmarks is not None else self._detect_landmarks(image)
        )
        normalized_face_rects = _normalize_face_rects(
            face_rects if face_rects is not None else self._detect_faces(image)
        )
        face_rect = _largest_face_rect(normalized_face_rects)
        face_detected = bool(normalized_face_rects) or normalized_landmarks is not None

        eye_metrics = self._build_eye_metrics(normalized_landmarks)
        eyes_open = True if eye_metrics.eyes_open is None else eye_metrics.eyes_open

        tongue_metrics = self._build_tongue_metrics(
            image=image,
            tongue_bbox=tongue_bbox,
            tongue_mask=tongue_mask,
            landmarks=normalized_landmarks,
            face_rect=face_rect,
        )
        tongue_straight = tongue_metrics.tongue_straight

        label = self._classify_label(
            face_detected=face_detected,
            eyes_open=eyes_open,
            tongue_straight=tongue_straight,
        )
        reasons = self._build_reasons(
            face_detected=face_detected,
            eyes_open=eyes_open,
            tongue_straight=tongue_straight,
            eye_metrics=eye_metrics,
            tongue_metrics=tongue_metrics,
        )

        return {
            "label": label,
            "passed": label == "qualified",
            "score": self._score_assessment(
                face_detected=face_detected,
                eye_metrics=eye_metrics,
                tongue_metrics=tongue_metrics,
            ),
            "summary": QUALITY_LABEL_SUMMARIES[label],
            "reasons": reasons,
            "paper_reference": PAPER_REFERENCE,
            "metrics": {
                "face_detected": face_detected,
                "face_count": len(normalized_face_rects),
                "face_box": (
                    {
                        "x": face_rect[0],
                        "y": face_rect[1],
                        "w": face_rect[2],
                        "h": face_rect[3],
                    }
                    if face_rect is not None
                    else None
                ),
                "eyes": asdict(eye_metrics),
                "tongue": asdict(tongue_metrics),
            },
        }

    def _detect_faces(self, image: np.ndarray) -> Sequence[FaceRect]:
        """Run face detection when explicit rectangles were not provided."""

        if self._face_detector is None:
            self._face_detector = self._build_face_detector()
        if self._face_detector is None:
            return []
        return self._face_detector(image)

    def _detect_landmarks(self, image: np.ndarray) -> Sequence[Point] | None:
        """Run face landmark detection when explicit landmarks were not provided."""

        if self._landmark_detector is None:
            self._landmark_detector = self._build_landmark_detector()
        if self._landmark_detector is None:
            return None
        return self._landmark_detector(image)

    def _build_face_detector(self) -> FaceDetector | None:
        """Build a face detector backed by the shared model manager when possible."""

        try:
            from api.services.model_manager import get_model_manager
        except Exception:
            return None

        model_manager = get_model_manager()
        cascade = model_manager.get_face_cascade()
        if cascade is None:
            return None

        def detect(image: np.ndarray) -> Sequence[FaceRect]:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                flags=cv2.CASCADE_SCALE_IMAGE,
                minSize=(80, 80),
            )

        return detect

    def _build_landmark_detector(self) -> LandmarkDetector | None:
        """Build a MediaPipe face landmark detector with a model-manager fallback."""

        face_mesh: Any | None = None
        try:
            from api.services.model_manager import get_model_manager

            face_mesh = get_model_manager().get_mediapipe_face_mesh()
        except Exception:
            face_mesh = None

        if face_mesh is None:
            try:
                from api.services.mediapipe_compat import create_face_mesh
            except Exception:
                return None
            face_mesh = create_face_mesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
            if face_mesh is None:
                return None

        def detect(image: np.ndarray) -> Sequence[Point] | None:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if not getattr(results, "multi_face_landmarks", None):
                return None

            h, w = image.shape[:2]
            points: list[Point] = []
            for landmark in results.multi_face_landmarks[0].landmark:
                points.append((int(landmark.x * w), int(landmark.y * h)))
            return points

        return detect

    def _build_eye_metrics(self, landmarks: Sequence[Point] | None) -> EyeMetrics:
        """Calculate eye openness metrics."""

        if landmarks is None:
            return EyeMetrics(
                left_ear=None,
                right_ear=None,
                average_ear=None,
                threshold=EYE_OPEN_THRESHOLD,
                eyes_open=None,
                reliable=False,
            )

        left_ear = _compute_eye_aspect_ratio(landmarks, LEFT_EYE_INDICES)
        right_ear = _compute_eye_aspect_ratio(landmarks, RIGHT_EYE_INDICES)
        valid_ears = [ear for ear in (left_ear, right_ear) if ear is not None]
        if not valid_ears:
            return EyeMetrics(
                left_ear=left_ear,
                right_ear=right_ear,
                average_ear=None,
                threshold=EYE_OPEN_THRESHOLD,
                eyes_open=None,
                reliable=False,
            )

        average_ear = float(sum(valid_ears) / len(valid_ears))
        return EyeMetrics(
            left_ear=left_ear,
            right_ear=right_ear,
            average_ear=average_ear,
            threshold=EYE_OPEN_THRESHOLD,
            eyes_open=average_ear >= EYE_OPEN_THRESHOLD,
            reliable=True,
        )

    def _build_tongue_metrics(
        self,
        *,
        image: np.ndarray,
        tongue_bbox: dict[str, int] | None,
        tongue_mask: np.ndarray | None,
        landmarks: Sequence[Point] | None,
        face_rect: FaceRect | None,
    ) -> TongueMetrics:
        """Calculate tongue visibility and alignment metrics."""

        binary_mask = _to_binary_mask(tongue_mask)
        if binary_mask is None:
            return TongueMetrics(
                detected=False,
                mask_area_ratio=0.0,
                bbox_area_ratio=0.0,
                height_width_ratio=0.0,
                angle_from_vertical_deg=None,
                tip_offset_ratio=None,
                extension_ratio=None,
                tongue_straight=False,
            )

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return TongueMetrics(
                detected=False,
                mask_area_ratio=0.0,
                bbox_area_ratio=0.0,
                height_width_ratio=0.0,
                angle_from_vertical_deg=None,
                tip_offset_ratio=None,
                extension_ratio=None,
                tongue_straight=False,
            )

        contour = max(contours, key=cv2.contourArea)
        mask_area = float(cv2.contourArea(contour))
        image_area = float(image.shape[0] * image.shape[1])
        if image_area <= 1.0:
            image_area = 1.0

        contour_x, contour_y, contour_w, contour_h = cv2.boundingRect(contour)
        if tongue_bbox is not None:
            bbox_x = int(tongue_bbox.get("x", contour_x))
            bbox_y = int(tongue_bbox.get("y", contour_y))
            bbox_w = max(int(tongue_bbox.get("w", contour_w)), 1)
            bbox_h = max(int(tongue_bbox.get("h", contour_h)), 1)
        else:
            bbox_x, bbox_y, bbox_w, bbox_h = contour_x, contour_y, contour_w, contour_h

        contour_points = contour.reshape(-1, 2)
        fit_line = cv2.fitLine(contour_points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        vx, vy = float(fit_line[0]), float(fit_line[1])
        angle_from_vertical = abs(math.degrees(math.atan2(vx, vy)))

        local_top_cutoff = contour_points[:, 1].min() + max(4, int(contour_h * 0.2))
        local_bottom_cutoff = contour_points[:, 1].max() - max(4, int(contour_h * 0.2))
        top_band = contour_points[contour_points[:, 1] <= local_top_cutoff]
        bottom_band = contour_points[contour_points[:, 1] >= local_bottom_cutoff]

        root_center_x = float(np.mean(top_band[:, 0])) if len(top_band) else float(bbox_x + bbox_w / 2.0)
        tip_center_x = float(
            np.mean(bottom_band[:, 0])
        ) if len(bottom_band) else float(bbox_x + bbox_w / 2.0)

        mouth_center_x, mouth_bottom_y, mouth_width = _mouth_metrics(landmarks)
        face_height = _face_height(landmarks, face_rect)

        reference_center_x = mouth_center_x if mouth_center_x is not None else float(bbox_x + bbox_w / 2.0)
        tip_offset_ratio = abs(tip_center_x - reference_center_x) / max(float(bbox_w), 1.0)

        extension_ratio: float | None = None
        if mouth_bottom_y is not None and face_height is not None:
            tongue_bottom = float(bbox_y + bbox_h)
            extension_ratio = max((tongue_bottom - mouth_bottom_y) / face_height, 0.0)

        height_width_ratio = bbox_h / max(float(bbox_w), 1.0)
        mask_area_ratio = mask_area / image_area
        bbox_area_ratio = float(bbox_w * bbox_h) / image_area
        elongated_enough = height_width_ratio >= TONGUE_HEIGHT_WIDTH_THRESHOLD
        visible_enough = mask_area_ratio >= TONGUE_MASK_AREA_THRESHOLD
        aligned_enough = angle_from_vertical <= TONGUE_ANGLE_THRESHOLD_DEG
        centered_enough = tip_offset_ratio <= TONGUE_TIP_OFFSET_THRESHOLD
        extension_enough = True
        if extension_ratio is not None:
            extension_enough = extension_ratio >= TONGUE_EXTENSION_THRESHOLD
        elif mouth_width is not None:
            extension_enough = bbox_h >= mouth_width * 0.7

        tongue_straight = all(
            (
                visible_enough,
                elongated_enough,
                aligned_enough,
                centered_enough,
                extension_enough,
            )
        )

        return TongueMetrics(
            detected=True,
            mask_area_ratio=round(mask_area_ratio, 4),
            bbox_area_ratio=round(bbox_area_ratio, 4),
            height_width_ratio=round(height_width_ratio, 4),
            angle_from_vertical_deg=round(angle_from_vertical, 2),
            tip_offset_ratio=round(tip_offset_ratio, 4),
            extension_ratio=None if extension_ratio is None else round(extension_ratio, 4),
            tongue_straight=tongue_straight,
        )

    def _classify_label(
        self,
        *,
        face_detected: bool,
        eyes_open: bool,
        tongue_straight: bool,
    ) -> str:
        """Map the geometric checks to the paper's five labels."""

        if not face_detected:
            return "non_face"
        if not tongue_straight and not eyes_open:
            return "tongue_not_straight_and_eyes_closed"
        if not tongue_straight:
            return "tongue_not_straight"
        if not eyes_open:
            return "eyes_closed"
        return "qualified"

    def _build_reasons(
        self,
        *,
        face_detected: bool,
        eyes_open: bool,
        tongue_straight: bool,
        eye_metrics: EyeMetrics,
        tongue_metrics: TongueMetrics,
    ) -> list[str]:
        """Build a short reason list for UI and debugging."""

        reasons: list[str] = []
        if not face_detected:
            return ["No frontal face was detected."]
        if eye_metrics.reliable is False:
            reasons.append("Eye landmarks were unavailable; eye status is approximate.")
        elif not eyes_open:
            reasons.append("Eye aspect ratio is below the open-eye threshold.")
        if not tongue_metrics.detected:
            reasons.append("No tongue region was segmented.")
            return reasons
        if not tongue_straight:
            if tongue_metrics.extension_ratio is not None and (
                tongue_metrics.extension_ratio < TONGUE_EXTENSION_THRESHOLD
            ):
                reasons.append("Tongue extension below the minimum threshold.")
            if tongue_metrics.angle_from_vertical_deg is not None and (
                tongue_metrics.angle_from_vertical_deg > TONGUE_ANGLE_THRESHOLD_DEG
            ):
                reasons.append("Tongue axis is too tilted from vertical.")
            if tongue_metrics.tip_offset_ratio is not None and (
                tongue_metrics.tip_offset_ratio > TONGUE_TIP_OFFSET_THRESHOLD
            ):
                reasons.append("Tongue tip is too far from the mouth centerline.")
            if tongue_metrics.height_width_ratio < TONGUE_HEIGHT_WIDTH_THRESHOLD:
                reasons.append("Tongue region is too short or too wide to be considered extended.")
            if tongue_metrics.mask_area_ratio < TONGUE_MASK_AREA_THRESHOLD:
                reasons.append("Tongue region is too small for reliable diagnosis.")
        if not reasons:
            reasons.append("Face, eyes, and tongue posture passed the quality gate.")
        return reasons

    def _score_assessment(
        self,
        *,
        face_detected: bool,
        eye_metrics: EyeMetrics,
        tongue_metrics: TongueMetrics,
    ) -> float:
        """Produce a coarse confidence-like quality score in `[0, 1]`."""

        if not face_detected:
            return 0.0

        score = 0.35
        if eye_metrics.eyes_open is True:
            score += 0.25
        elif eye_metrics.eyes_open is False:
            score -= 0.15
        else:
            score += 0.05

        if tongue_metrics.detected:
            score += 0.15
        if tongue_metrics.tongue_straight:
            score += 0.25
        else:
            score -= 0.15

        return round(min(max(score, 0.0), 1.0), 4)


__all__ = ["TongueQualityControlService", "QUALITY_LABEL_SUMMARIES", "PAPER_REFERENCE"]
