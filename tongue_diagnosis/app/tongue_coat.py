"""Candidate tongue-coat observation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tongue_diagnosis.app.tongue_moisture import extract_overexposed_mask


@dataclass(frozen=True)
class TongueCoatMeasurement:
    """Whole-tongue coat candidate observation."""

    coat_visibility: str
    coat_coverage_ratio: float
    coat_color_tendency: str
    coat_thickness_tendency: str
    white_coat_ratio: float
    yellow_coat_ratio: float
    gray_black_coat_ratio: float


@dataclass(frozen=True)
class RegionCoatMeasurement:
    """Region-level coat candidate observation."""

    region_id: str
    region_name: str
    pixel_count: int
    coverage_ratio: float
    coat_coverage_ratio: float
    coat_color_tendency: str
    coat_thickness_tendency: str


@dataclass(frozen=True)
class _CoatCandidateMasks:
    """Intermediate coat candidate masks."""

    white: np.ndarray
    yellow: np.ndarray
    gray_black: np.ndarray

    @property
    def combined(self) -> np.ndarray:
        """Return all coat candidates as one binary mask."""

        return self.white | self.yellow | self.gray_black


def _round_ratio(value: float) -> float:
    """Clamp one ratio to `0~1` and round for stable API output."""

    return round(float(np.clip(value, 0.0, 1.0)), 4)


def _build_coat_analysis_mask(image: np.ndarray, tongue_mask: np.ndarray) -> np.ndarray:
    """Remove overexposed pixels before coat analysis."""

    overexposed_mask = extract_overexposed_mask(image, tongue_mask)
    return tongue_mask & ~overexposed_mask


def _extract_coat_candidate_masks(image: np.ndarray, analysis_mask: np.ndarray) -> _CoatCandidateMasks:
    """Extract first-pass white/yellow/gray-black coat candidates."""

    if not np.any(analysis_mask):
        empty = np.zeros(analysis_mask.shape, dtype=bool)
        return _CoatCandidateMasks(white=empty, yellow=empty, gray_black=empty)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    lightness = lab[:, :, 0].astype(np.float32) * 100.0 / 255.0
    a_value = lab[:, :, 1].astype(np.float32) - 128.0
    b_value = lab[:, :, 2].astype(np.float32) - 128.0

    white = analysis_mask & (value >= 150) & (saturation <= 85) & (lightness >= 55.0) & (a_value <= 18.0)
    yellow = analysis_mask & (hue >= 15) & (hue <= 38) & (saturation >= 35) & (value >= 90) & (b_value >= 14.0)
    gray_black = analysis_mask & (value <= 90) & (saturation <= 110)

    kernel = np.ones((3, 3), np.uint8)
    white = cv2.morphologyEx(white.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
    yellow = cv2.morphologyEx(yellow.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
    gray_black = cv2.morphologyEx(gray_black.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
    return _CoatCandidateMasks(white=white, yellow=yellow, gray_black=gray_black)


def _describe_visibility(coat_coverage_ratio: float) -> str:
    """Describe coat visibility from candidate coverage."""

    if coat_coverage_ratio < 0.03:
        return "无明显"
    if coat_coverage_ratio < 0.18:
        return "少量可见"
    return "较明显"


def _describe_color_tendency(
    *,
    white_ratio: float,
    yellow_ratio: float,
    gray_black_ratio: float,
    coat_coverage_ratio: float,
) -> str:
    """Describe the dominant coat color candidate."""

    if coat_coverage_ratio < 0.03:
        return "无法判断"

    ratios = {
        "偏白": white_ratio,
        "偏黄": yellow_ratio,
        "偏灰黑": gray_black_ratio,
    }
    return max(ratios.items(), key=lambda item: item[1])[0]


def _describe_thickness_tendency(coat_coverage_ratio: float) -> str:
    """Describe first-pass coat thickness tendency from coverage only."""

    if coat_coverage_ratio < 0.03:
        return "无法判断"
    if coat_coverage_ratio < 0.25:
        return "偏薄"
    return "偏厚"


def analyze_tongue_coat(image: np.ndarray, tongue_mask: np.ndarray) -> TongueCoatMeasurement | None:
    """Calculate whole-tongue coat candidate observation.

    Args:
        image: Cropped BGR tongue image without overlays.
        tongue_mask: Binary tongue mask aligned with `image`.

    Returns:
        One whole-tongue coat observation, or `None` when no tongue pixels exist.
    """

    tongue_pixels = int(np.count_nonzero(tongue_mask))
    if tongue_pixels == 0:
        return None

    analysis_mask = _build_coat_analysis_mask(image, tongue_mask)
    masks = _extract_coat_candidate_masks(image, analysis_mask)
    coat_coverage_ratio = _round_ratio(np.count_nonzero(masks.combined) / tongue_pixels)
    white_ratio = _round_ratio(np.count_nonzero(masks.white) / tongue_pixels)
    yellow_ratio = _round_ratio(np.count_nonzero(masks.yellow) / tongue_pixels)
    gray_black_ratio = _round_ratio(np.count_nonzero(masks.gray_black) / tongue_pixels)

    return TongueCoatMeasurement(
        coat_visibility=_describe_visibility(coat_coverage_ratio),
        coat_coverage_ratio=coat_coverage_ratio,
        coat_color_tendency=_describe_color_tendency(
            white_ratio=white_ratio,
            yellow_ratio=yellow_ratio,
            gray_black_ratio=gray_black_ratio,
            coat_coverage_ratio=coat_coverage_ratio,
        ),
        coat_thickness_tendency=_describe_thickness_tendency(coat_coverage_ratio),
        white_coat_ratio=white_ratio,
        yellow_coat_ratio=yellow_ratio,
        gray_black_coat_ratio=gray_black_ratio,
    )


def analyze_region_coat(
    image: np.ndarray,
    region_masks: dict[str, np.ndarray],
    *,
    region_output_order: tuple[str, ...],
    region_labels: dict[str, str],
    total_tongue_pixels: int,
) -> tuple[RegionCoatMeasurement, ...]:
    """Calculate region-level coat candidate observations."""

    if total_tongue_pixels == 0:
        return ()

    tongue_mask = np.zeros(next(iter(region_masks.values())).shape, dtype=bool)
    for region_mask in region_masks.values():
        tongue_mask |= region_mask

    analysis_mask = _build_coat_analysis_mask(image, tongue_mask)
    masks = _extract_coat_candidate_masks(image, analysis_mask)
    measurements: list[RegionCoatMeasurement] = []

    for region_id in region_output_order:
        region_mask = region_masks[region_id]
        pixel_count = int(np.count_nonzero(region_mask))
        if pixel_count == 0:
            continue

        region_white_ratio = _round_ratio(np.count_nonzero(masks.white & region_mask) / pixel_count)
        region_yellow_ratio = _round_ratio(np.count_nonzero(masks.yellow & region_mask) / pixel_count)
        region_gray_black_ratio = _round_ratio(np.count_nonzero(masks.gray_black & region_mask) / pixel_count)
        coat_coverage_ratio = _round_ratio(np.count_nonzero(masks.combined & region_mask) / pixel_count)
        measurements.append(
            RegionCoatMeasurement(
                region_id=region_id,
                region_name=region_labels[region_id],
                pixel_count=pixel_count,
                coverage_ratio=round(pixel_count / total_tongue_pixels, 4),
                coat_coverage_ratio=coat_coverage_ratio,
                coat_color_tendency=_describe_color_tendency(
                    white_ratio=region_white_ratio,
                    yellow_ratio=region_yellow_ratio,
                    gray_black_ratio=region_gray_black_ratio,
                    coat_coverage_ratio=coat_coverage_ratio,
                ),
                coat_thickness_tendency=_describe_thickness_tendency(coat_coverage_ratio),
            )
        )

    return tuple(measurements)
