"""MediaPipe compatibility helpers.

The project was originally written against the classic ``mp.solutions`` API.
Recent MediaPipe wheels expose the Tasks API instead. These helpers keep import
paths stable and provide a conservative Haar-based landmark fallback when the
classic FaceMesh API is unavailable.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Iterable

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:  # pragma: no cover - depends on optional runtime package
    mp = None


def has_classic_solutions() -> bool:
    """Return whether the installed MediaPipe exposes ``mp.solutions``."""

    return mp is not None and hasattr(mp, "solutions")


def get_drawing_utils():
    """Return MediaPipe drawing utils when available."""

    if not has_classic_solutions():
        return None
    return getattr(mp.solutions, "drawing_utils", None)


def get_drawing_styles():
    """Return MediaPipe drawing styles when available."""

    if not has_classic_solutions():
        return None
    return getattr(mp.solutions, "drawing_styles", None)


def create_face_mesh(
    *,
    static_image_mode: bool = True,
    max_num_faces: int = 1,
    refine_landmarks: bool = True,
    min_detection_confidence: float = 0.5,
):
    """Create a classic FaceMesh instance when the installed package supports it."""

    if not has_classic_solutions():
        return None
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=static_image_mode,
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
    )


def detect_face_rect(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Detect one face rectangle with OpenCV Haar cascade."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        return None

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(60, 60),
    )
    if faces is None or len(faces) == 0:
        return None
    x, y, width, height = max(faces, key=lambda item: int(item[2]) * int(item[3]))
    return int(x), int(y), int(width), int(height)


def synthesize_face_landmarks(
    image: np.ndarray,
    face_rect: tuple[int, int, int, int] | None = None,
) -> list[tuple[int, int]] | None:
    """Build approximate 468-point landmarks from a detected face rectangle."""

    rect = face_rect or detect_face_rect(image)
    if rect is None:
        return None

    x, y, width, height = rect
    if width <= 0 or height <= 0:
        return None

    cx = x + width / 2.0
    cy = y + height / 2.0
    points: list[tuple[int, int]] = []
    for index in range(468):
        theta = (2.0 * np.pi * index) / 468.0
        px = cx + np.cos(theta) * width * 0.36
        py = cy + np.sin(theta) * height * 0.46
        points.append(_clip_point(px, py, image.shape))

    _assign_polyline(
        points,
        [
            10,
            338,
            297,
            332,
            284,
            251,
            389,
            356,
            454,
            323,
            361,
            288,
            397,
            365,
            379,
            378,
            400,
            377,
            152,
            148,
            176,
            149,
            150,
            136,
            172,
            58,
            132,
            93,
            234,
            127,
            162,
            21,
            54,
            103,
            67,
            109,
        ],
        _ellipse_points(cx, cy, width * 0.48, height * 0.52, -np.pi / 2, 3 * np.pi / 2, 36),
        image.shape,
    )
    _assign_ellipse(points, [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246], x + width * 0.35, y + height * 0.38, width * 0.11, height * 0.045, image.shape)
    _assign_ellipse(points, [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398], x + width * 0.65, y + height * 0.38, width * 0.11, height * 0.045, image.shape)
    _assign_ellipse(points, [70, 63, 105, 66, 107, 55, 65, 52, 53, 46], x + width * 0.35, y + height * 0.30, width * 0.13, height * 0.025, image.shape)
    _assign_ellipse(points, [300, 293, 334, 296, 336, 285, 295, 282, 283, 276], x + width * 0.65, y + height * 0.30, width * 0.13, height * 0.025, image.shape)
    _assign_ellipse(points, [0, 37, 39, 40, 185, 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 308, 324, 318, 402, 317], x + width * 0.50, y + height * 0.70, width * 0.20, height * 0.055, image.shape)
    _assign_ellipse(points, [6, 122, 188, 174, 236, 198, 209, 129, 98, 97, 2, 326, 327, 58, 429, 420, 456, 399, 412, 351, 358, 275, 440, 344, 278, 48, 64, 94, 4, 45, 220, 115], x + width * 0.50, y + height * 0.53, width * 0.16, height * 0.17, image.shape)

    special_points = {
        8: (x + width * 0.50, y + height * 0.44),
        13: (x + width * 0.50, y + height * 0.67),
        14: (x + width * 0.50, y + height * 0.73),
        18: (x + width * 0.50, y + height * 0.82),
        152: (x + width * 0.50, y + height * 0.98),
        127: (x + width * 0.04, y + height * 0.50),
        356: (x + width * 0.96, y + height * 0.50),
    }
    for index, point in special_points.items():
        points[index] = _clip_point(point[0], point[1], image.shape)

    return points


def landmarks_to_mediapipe_result(points: Iterable[tuple[int, int]], image_shape: tuple[int, ...]):
    """Convert pixel landmarks into a small object shaped like FaceMesh output."""

    height, width = image_shape[:2]
    landmarks = [
        SimpleNamespace(x=float(x) / max(width, 1), y=float(y) / max(height, 1), z=0.0)
        for x, y in points
    ]
    return SimpleNamespace(multi_face_landmarks=[SimpleNamespace(landmark=landmarks)])


def _assign_ellipse(
    points: list[tuple[int, int]],
    indices: list[int],
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    image_shape: tuple[int, ...],
) -> None:
    ellipse_points = _ellipse_points(cx, cy, rx, ry, 0.0, 2.0 * np.pi, len(indices))
    _assign_polyline(points, indices, ellipse_points, image_shape)


def _assign_polyline(
    points: list[tuple[int, int]],
    indices: list[int],
    values: list[tuple[float, float]],
    image_shape: tuple[int, ...],
) -> None:
    for index, (px, py) in zip(indices, values, strict=False):
        if 0 <= index < len(points):
            points[index] = _clip_point(px, py, image_shape)


def _ellipse_points(
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    start: float,
    stop: float,
    count: int,
) -> list[tuple[float, float]]:
    if count <= 1:
        return [(cx, cy)]
    return [
        (cx + np.cos(theta) * rx, cy + np.sin(theta) * ry)
        for theta in np.linspace(start, stop, count, endpoint=False)
    ]


def _clip_point(px: float, py: float, image_shape: tuple[int, ...]) -> tuple[int, int]:
    height, width = image_shape[:2]
    return (
        int(np.clip(round(px), 0, max(width - 1, 0))),
        int(np.clip(round(py), 0, max(height - 1, 0))),
    )


__all__ = [
    "create_face_mesh",
    "detect_face_rect",
    "get_drawing_styles",
    "get_drawing_utils",
    "has_classic_solutions",
    "landmarks_to_mediapipe_result",
    "synthesize_face_landmarks",
]
