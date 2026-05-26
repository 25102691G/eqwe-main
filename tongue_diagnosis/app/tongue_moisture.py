"""Tongue moisture analysis helpers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RegionMoistureMeasurement:
    """Moisture summary extracted from one tongue region."""

    region_id: str
    region_name: str
    pixel_count: int
    coverage_ratio: float
    gloss_area_ratio: float
    highlight_blob_count: int
    highlight_blob_max_ratio: float
    highlight_blob_mean_area: float
    crack_length_ratio: float
    crack_area_ratio: float
    overexposed_ratio: float
    moisture_score: float
    moisture_label: str


@dataclass(frozen=True)
class TongueMoistureMeasurement:
    """Image-level moisture assessment for one segmented tongue."""

    moisture_score: float
    moisture_label: str
    quality_passed: bool
    quality_reasons: tuple[str, ...]
    focus_score: float
    overexposed_ratio: float
    segmentation_coverage: float
    gloss_area_ratio: float
    highlight_blob_count: int
    highlight_blob_max_ratio: float
    highlight_blob_mean_area: float
    crack_length_ratio: float
    crack_area_ratio: float


@dataclass(frozen=True)
class _MoistureFeatures:
    """Intermediate moisture features measured from one binary region mask."""

    gloss_area_ratio: float
    highlight_blob_count: int
    highlight_blob_max_ratio: float
    highlight_blob_mean_area: float
    crack_length_ratio: float
    crack_area_ratio: float
    overexposed_ratio: float


@dataclass(frozen=True)
class MoistureScoreBreakdown:
    """Human-readable breakdown of the rule-based moisture score."""

    gloss_area_score: float
    highlight_blob_score: float
    crack_score_inverse: float
    moisture_score: float


def _round_ratio(value: float) -> float:
    """Clamp one ratio to `0~1` and round for stable API output."""

    return round(float(np.clip(value, 0.0, 1.0)), 4)


def _filter_small_components(binary_mask: np.ndarray, *, min_area: int) -> np.ndarray:
    """Drop very small connected components from one binary mask."""

    if min_area <= 1 or not np.any(binary_mask):
        return binary_mask

    component_mask = binary_mask.astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(component_mask, connectivity=8)
    filtered = np.zeros_like(component_mask)

    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            filtered[labels == label] = 1

    return filtered.astype(bool)


def _component_areas(binary_mask: np.ndarray) -> np.ndarray:
    """Return connected-component areas from one binary mask."""

    if not np.any(binary_mask):
        return np.array([], dtype=np.float32)

    component_mask = binary_mask.astype(np.uint8)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(component_mask, connectivity=8)
    if component_count <= 1:
        return np.array([], dtype=np.float32)
    return stats[1:, cv2.CC_STAT_AREA].astype(np.float32)


def _skeletonize_mask(binary_mask: np.ndarray) -> np.ndarray:
    """Approximate the center lines of one binary mask."""

    skeleton = np.zeros(binary_mask.shape, dtype=np.uint8)
    work = binary_mask.copy()
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while cv2.countNonZero(work) > 0:
        eroded = cv2.erode(work, element)
        opened = cv2.dilate(eroded, element)
        residue = cv2.subtract(work, opened)
        skeleton = cv2.bitwise_or(skeleton, residue)
        work = eroded

    return skeleton > 0


def skeletonize_mask(binary_mask: np.ndarray) -> np.ndarray:
    """Expose binary-mask skeletonization for debug visualizations."""

    return _skeletonize_mask(binary_mask.astype(np.uint8))


def _extract_gloss_mask(hsv_image: np.ndarray, analysis_mask: np.ndarray) -> np.ndarray:
    """Extract specular-highlight candidates inside one analysis mask."""

    if not np.any(analysis_mask):
        return np.zeros_like(analysis_mask, dtype=bool)

    saturation_channel = hsv_image[:, :, 1]
    value_channel = hsv_image[:, :, 2]

    high_value_threshold = max(210.0, float(np.percentile(value_channel[analysis_mask], 92.0)))
    low_saturation_threshold = min(
        150.0,
        float(np.percentile(saturation_channel[analysis_mask], 45.0)) + 20.0,
    )

    gloss_mask = (
        analysis_mask
        & (value_channel >= high_value_threshold)
        & (saturation_channel <= low_saturation_threshold)
    )
    gloss_mask = cv2.morphologyEx(gloss_mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    gloss_mask = cv2.morphologyEx(gloss_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    return gloss_mask


def extract_gloss_mask(image: np.ndarray, analysis_mask: np.ndarray) -> np.ndarray:
    """Extract the gloss mask from one BGR image and analysis mask."""

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return _extract_gloss_mask(hsv_image, analysis_mask)


def _extract_crack_mask(gray_image: np.ndarray, analysis_mask: np.ndarray) -> np.ndarray:
    """Extract dark crack-like line structures inside one analysis mask."""

    if not np.any(analysis_mask):
        return np.zeros_like(analysis_mask, dtype=bool)

    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    blackhat = cv2.morphologyEx(
        blurred,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    crack_threshold = max(18.0, float(np.percentile(blackhat[analysis_mask], 88.0)))
    crack_mask = analysis_mask & (blackhat >= crack_threshold)
    crack_mask = _filter_small_components(crack_mask, min_area=max(int(np.count_nonzero(analysis_mask) // 400), 4))
    return crack_mask


def extract_crack_mask(image: np.ndarray, analysis_mask: np.ndarray) -> np.ndarray:
    """Extract the crack mask from one BGR image and analysis mask."""

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return _extract_crack_mask(gray_image, analysis_mask)


def extract_overexposed_mask(image: np.ndarray, analysis_mask: np.ndarray) -> np.ndarray:
    """Extract pixels currently treated as overexposed inside one analysis mask."""

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return analysis_mask & (hsv_image[:, :, 2] >= 250)


def _summarize_moisture_features(
    hsv_image: np.ndarray,
    gray_image: np.ndarray,
    analysis_mask: np.ndarray,
) -> _MoistureFeatures:
    """Measure moisture-related features from one mask."""

    pixel_count = int(np.count_nonzero(analysis_mask))
    if pixel_count == 0:
        return _MoistureFeatures(
            gloss_area_ratio=0.0,
            highlight_blob_count=0,
            highlight_blob_max_ratio=0.0,
            highlight_blob_mean_area=0.0,
            crack_length_ratio=0.0,
            crack_area_ratio=0.0,
            overexposed_ratio=0.0,
        )

    gloss_mask = _filter_small_components(
        _extract_gloss_mask(hsv_image, analysis_mask),
        min_area=max(pixel_count // 300, 3),
    )
    gloss_areas = _component_areas(gloss_mask)
    gloss_area_ratio = _round_ratio(np.count_nonzero(gloss_mask) / pixel_count)
    highlight_blob_max_ratio = _round_ratio(float(gloss_areas.max()) / pixel_count) if gloss_areas.size else 0.0
    highlight_blob_mean_area = round(float(gloss_areas.mean()), 2) if gloss_areas.size else 0.0

    crack_mask = _extract_crack_mask(gray_image, analysis_mask)
    crack_skeleton = _skeletonize_mask(crack_mask.astype(np.uint8))
    crack_length_ratio = _round_ratio(np.count_nonzero(crack_skeleton) / pixel_count)
    crack_area_ratio = _round_ratio(np.count_nonzero(crack_mask) / pixel_count)

    overexposed_ratio = _round_ratio(
        np.count_nonzero(analysis_mask & (hsv_image[:, :, 2] >= 250)) / pixel_count
    )

    return _MoistureFeatures(
        gloss_area_ratio=gloss_area_ratio,
        highlight_blob_count=int(gloss_areas.size),
        highlight_blob_max_ratio=highlight_blob_max_ratio,
        highlight_blob_mean_area=highlight_blob_mean_area,
        crack_length_ratio=crack_length_ratio,
        crack_area_ratio=crack_area_ratio,
        overexposed_ratio=overexposed_ratio,
    )


def _calculate_moisture_score(features: _MoistureFeatures) -> float:
    """Convert moisture features into one coarse `0~100` score."""

    return build_score_breakdown_from_features(features).moisture_score


def build_score_breakdown_from_features(features: _MoistureFeatures) -> MoistureScoreBreakdown:
    """Convert one feature bundle into a stable score breakdown."""

    gloss_area_score = min(features.gloss_area_ratio / 0.06, 1.0) * 100.0
    continuity_score = min(features.highlight_blob_max_ratio / 0.03, 1.0) * 100.0
    fragmentation_score = max(0.0, 100.0 - max(features.highlight_blob_count - 2, 0) * 20.0)
    highlight_blob_score = 0.0
    if features.highlight_blob_count > 0:
        highlight_blob_score = 0.7 * continuity_score + 0.3 * fragmentation_score
    crack_score_inverse = 100.0 - min(features.crack_length_ratio / 0.05, 1.0) * 100.0
    moisture_score = round(
        0.55 * gloss_area_score + 0.25 * highlight_blob_score + 0.20 * crack_score_inverse,
        2,
    )
    return MoistureScoreBreakdown(
        gloss_area_score=round(gloss_area_score, 2),
        highlight_blob_score=round(highlight_blob_score, 2),
        crack_score_inverse=round(crack_score_inverse, 2),
        moisture_score=moisture_score,
    )


def build_score_breakdown_from_measurement(
    measurement: RegionMoistureMeasurement | TongueMoistureMeasurement,
) -> MoistureScoreBreakdown:
    """Reconstruct the rule-score breakdown from one API-facing measurement."""

    features = _MoistureFeatures(
        gloss_area_ratio=measurement.gloss_area_ratio,
        highlight_blob_count=measurement.highlight_blob_count,
        highlight_blob_max_ratio=measurement.highlight_blob_max_ratio,
        highlight_blob_mean_area=measurement.highlight_blob_mean_area,
        crack_length_ratio=measurement.crack_length_ratio,
        crack_area_ratio=measurement.crack_area_ratio,
        overexposed_ratio=measurement.overexposed_ratio,
    )
    return build_score_breakdown_from_features(features)


def _describe_moisture_score(moisture_score: float) -> str:
    """Map one moisture score to a coarse label."""

    if moisture_score >= 70.0:
        return "润"
    if moisture_score >= 40.0:
        return "中间态"
    return "燥"


def _calculate_focus_score(gray_image: np.ndarray, analysis_mask: np.ndarray) -> float:
    """Measure image sharpness inside one mask."""

    if not np.any(analysis_mask):
        return 0.0

    laplacian = cv2.Laplacian(gray_image, cv2.CV_32F)
    return round(float(laplacian[analysis_mask].var()), 2)


def _calculate_tongue_area_ratio(analysis_mask: np.ndarray) -> float:
    """Measure how much of the current image is covered by tongue pixels."""

    image_area = max(int(analysis_mask.shape[0]) * int(analysis_mask.shape[1]), 1)
    return _round_ratio(np.count_nonzero(analysis_mask) / image_area)


def _bbox_touches_analysis_edge(
    bbox_rect: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
    *,
    margin_px: int = 3,
) -> bool:
    """Return whether the tongue bbox is too close to the analysis image edge."""

    x_value, y_value, width, height = bbox_rect
    image_height, image_width = image_shape[:2]
    return (
        x_value <= margin_px
        or y_value <= margin_px
        or x_value + width >= image_width - margin_px
        or y_value + height >= image_height - margin_px
    )


def _has_color_cast(image: np.ndarray, analysis_mask: np.ndarray) -> bool:
    """Detect strong blue or green casts that can destabilize tongue color analysis."""

    if not np.any(analysis_mask):
        return False

    pixels = image[analysis_mask].astype(np.float32)
    blue_mean, green_mean, red_mean = pixels.mean(axis=0)
    return bool(
        (blue_mean > red_mean * 1.25 and blue_mean - red_mean > 35.0)
        or (green_mean > red_mean * 1.20 and green_mean - red_mean > 30.0)
    )


def analyze_region_moisture(
    image: np.ndarray,
    region_masks: dict[str, np.ndarray],
    *,
    region_output_order: tuple[str, ...],
    region_labels: dict[str, str],
    total_tongue_pixels: int,
) -> tuple[RegionMoistureMeasurement, ...]:
    """Calculate moisture-related metrics for the provided tongue regions.

    Args:
        image: Cropped BGR tongue image without overlays.
        region_masks: Region masks keyed by region identifier.
        region_output_order: Ordered region identifiers to emit.
        region_labels: Human-readable labels for each region identifier.
        total_tongue_pixels: Total number of tongue pixels across all regions.

    Returns:
        One moisture summary per tongue region. Background pixels are excluded.
    """

    if total_tongue_pixels == 0:
        return ()

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    measurements: list[RegionMoistureMeasurement] = []

    for region_id in region_output_order:
        region_mask = region_masks[region_id]
        pixel_count = int(np.count_nonzero(region_mask))
        if pixel_count == 0:
            continue

        features = _summarize_moisture_features(hsv_image, gray_image, region_mask)
        moisture_score = _calculate_moisture_score(features)

        measurements.append(
            RegionMoistureMeasurement(
                region_id=region_id,
                region_name=region_labels[region_id],
                pixel_count=pixel_count,
                coverage_ratio=round(pixel_count / total_tongue_pixels, 4),
                gloss_area_ratio=features.gloss_area_ratio,
                highlight_blob_count=features.highlight_blob_count,
                highlight_blob_max_ratio=features.highlight_blob_max_ratio,
                highlight_blob_mean_area=features.highlight_blob_mean_area,
                crack_length_ratio=features.crack_length_ratio,
                crack_area_ratio=features.crack_area_ratio,
                overexposed_ratio=features.overexposed_ratio,
                moisture_score=moisture_score,
                moisture_label=_describe_moisture_score(moisture_score),
            )
        )

    return tuple(measurements)


def analyze_tongue_moisture(
    image: np.ndarray,
    tongue_mask: np.ndarray,
    bbox_rect: tuple[int, int, int, int],
) -> TongueMoistureMeasurement | None:
    """Calculate image-level tongue moisture metrics.

    Args:
        image: Cropped BGR tongue image without overlays.
        tongue_mask: Binary tongue mask aligned with `image`.
        bbox_rect: Bounding box expressed as `(x, y, width, height)` within `image`.

    Returns:
        One image-level moisture summary, or `None` when no tongue pixels exist.
    """

    total_tongue_pixels = int(np.count_nonzero(tongue_mask))
    if total_tongue_pixels == 0:
        return None

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = _summarize_moisture_features(hsv_image, gray_image, tongue_mask)
    focus_score = _calculate_focus_score(gray_image, tongue_mask)
    tongue_area_ratio = _calculate_tongue_area_ratio(tongue_mask)
    bbox_area = max(int(bbox_rect[2]) * int(bbox_rect[3]), 1)
    segmentation_coverage = _round_ratio(total_tongue_pixels / bbox_area)
    moisture_score = _calculate_moisture_score(features)

    quality_reasons: list[str] = []
    if focus_score < 20.0:
        quality_reasons.append("low_focus")
    if features.overexposed_ratio > 0.08:
        quality_reasons.append("overexposed")
    if segmentation_coverage < 0.35:
        quality_reasons.append("low_segmentation_coverage")
    if tongue_area_ratio < 0.12:
        quality_reasons.append("small_tongue_region")
    if _bbox_touches_analysis_edge(bbox_rect, image.shape):
        quality_reasons.append("tongue_touches_edge")
    if _has_color_cast(image, tongue_mask):
        quality_reasons.append("color_cast")

    return TongueMoistureMeasurement(
        moisture_score=moisture_score,
        moisture_label=_describe_moisture_score(moisture_score),
        quality_passed=not quality_reasons,
        quality_reasons=tuple(quality_reasons),
        focus_score=focus_score,
        overexposed_ratio=features.overexposed_ratio,
        segmentation_coverage=segmentation_coverage,
        gloss_area_ratio=features.gloss_area_ratio,
        highlight_blob_count=features.highlight_blob_count,
        highlight_blob_max_ratio=features.highlight_blob_max_ratio,
        highlight_blob_mean_area=features.highlight_blob_mean_area,
        crack_length_ratio=features.crack_length_ratio,
        crack_area_ratio=features.crack_area_ratio,
    )
