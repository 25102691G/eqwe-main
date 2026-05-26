"""Optional debug visualizations for tongue moisture analysis."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tongue_diagnosis.app.tongue_moisture import (
    MoistureScoreBreakdown,
    RegionMoistureMeasurement,
    TongueMoistureMeasurement,
    build_score_breakdown_from_measurement,
    extract_crack_mask,
    extract_gloss_mask,
    extract_overexposed_mask,
    skeletonize_mask,
)


@dataclass(frozen=True)
class MoistureVisualizationArtifacts:
    """Debug visualization artifacts derived from moisture analysis."""

    gloss_mask: np.ndarray
    gloss_overlay: np.ndarray
    crack_mask: np.ndarray
    crack_skeleton: np.ndarray
    crack_overlay: np.ndarray
    overexposed_mask: np.ndarray
    overexposed_overlay: np.ndarray
    moisture_heatmap: np.ndarray
    score_breakdown_image: np.ndarray
    score_breakdown: MoistureScoreBreakdown


def build_moisture_visualizations(
    image: np.ndarray,
    tongue_mask: np.ndarray,
    region_masks: dict[str, np.ndarray],
    region_measurements: tuple[RegionMoistureMeasurement, ...],
    tongue_measurement: TongueMoistureMeasurement | None,
) -> MoistureVisualizationArtifacts | None:
    """Build the optional debug visualizations for one segmented tongue.

    Args:
        image: Cropped BGR image aligned with the tongue mask.
        tongue_mask: Binary tongue mask aligned with `image`.
        region_masks: Region masks keyed by region identifier.
        region_measurements: Region-level moisture measurements.
        tongue_measurement: Image-level moisture measurement.

    Returns:
        A visualization bundle, or `None` when no tongue measurement exists.
    """

    if tongue_measurement is None or not np.any(tongue_mask):
        return None

    base_image = image.copy()
    base_image[~tongue_mask] = (255, 255, 255)

    gloss_mask = extract_gloss_mask(image, tongue_mask)
    crack_mask = extract_crack_mask(image, tongue_mask)
    crack_skeleton = skeletonize_mask(crack_mask)
    overexposed_mask = extract_overexposed_mask(image, tongue_mask)
    score_breakdown = build_score_breakdown_from_measurement(tongue_measurement)

    return MoistureVisualizationArtifacts(
        gloss_mask=_render_binary_mask(gloss_mask),
        gloss_overlay=_overlay_mask(base_image, gloss_mask, (255, 120, 0)),
        crack_mask=_render_binary_mask(crack_mask),
        crack_skeleton=_render_binary_mask(crack_skeleton),
        crack_overlay=_overlay_mask(base_image, crack_skeleton, (0, 0, 255)),
        overexposed_mask=_render_binary_mask(overexposed_mask),
        overexposed_overlay=_overlay_mask(base_image, overexposed_mask, (0, 220, 255)),
        moisture_heatmap=_draw_moisture_heatmap(base_image, region_masks, region_measurements),
        score_breakdown_image=_draw_score_breakdown_image(score_breakdown),
        score_breakdown=score_breakdown,
    )


def _render_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Render one binary mask as a 3-channel white-on-black image."""

    rendered = np.zeros(mask.shape, dtype=np.uint8)
    rendered[mask] = 255
    return cv2.cvtColor(rendered, cv2.COLOR_GRAY2BGR)


def _overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay one binary mask on top of an image."""

    result = image.copy()
    overlay = image.copy()
    overlay[mask] = color
    cv2.addWeighted(overlay, alpha, result, 1.0 - alpha, 0.0, result)
    return result


def _draw_moisture_heatmap(
    image: np.ndarray,
    region_masks: dict[str, np.ndarray],
    region_measurements: tuple[RegionMoistureMeasurement, ...],
) -> np.ndarray:
    """Render one region-level moisture heatmap."""

    overlay = image.copy()
    result = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for measurement in region_measurements:
        region_mask = region_masks.get(measurement.region_id)
        if region_mask is None or not np.any(region_mask):
            continue

        overlay[region_mask] = _score_to_bgr(measurement.moisture_score)
        center_x, center_y = _mask_center(region_mask)
        label = f"{measurement.region_id}:{measurement.moisture_score:.0f}"
        cv2.putText(
            result,
            label,
            (max(center_x - 30, 5), max(center_y, 20)),
            font,
            0.45,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, 0.42, result, 0.58, 0.0, result)
    return result


def _mask_center(mask: np.ndarray) -> tuple[int, int]:
    """Return the approximate center of one binary mask."""

    points = np.column_stack(np.nonzero(mask))
    if points.size == 0:
        return (0, 0)

    center_y, center_x = points.mean(axis=0)
    return (int(center_x), int(center_y))


def _score_to_bgr(score: float) -> tuple[int, int, int]:
    """Map one moisture score to a stable region color."""

    dry = np.array((60.0, 90.0, 180.0))
    middle = np.array((80.0, 220.0, 245.0))
    moist = np.array((90.0, 190.0, 90.0))

    if score <= 40.0:
        ratio = np.clip(score / 40.0, 0.0, 1.0)
        color = dry * (1.0 - ratio) + middle * ratio
    else:
        ratio = np.clip((score - 40.0) / 60.0, 0.0, 1.0)
        color = middle * (1.0 - ratio) + moist * ratio

    return tuple(int(value) for value in color)


def _draw_score_breakdown_image(score_breakdown: MoistureScoreBreakdown) -> np.ndarray:
    """Render one score-breakdown card."""

    canvas = np.full((260, 560, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(canvas, "tongue moisture score breakdown", (20, 32), font, 0.7, (40, 40, 40), 2, cv2.LINE_AA)

    rows = (
        ("gloss_area_score", score_breakdown.gloss_area_score, (255, 120, 0)),
        ("highlight_blob_score", score_breakdown.highlight_blob_score, (235, 190, 20)),
        ("crack_score_inverse", score_breakdown.crack_score_inverse, (70, 170, 80)),
    )

    start_x = 240
    bar_width = 260
    row_y = 82
    for label, score, color in rows:
        cv2.putText(canvas, label, (20, row_y), font, 0.5, (50, 50, 50), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (start_x, row_y - 16), (start_x + bar_width, row_y + 4), (225, 225, 225), -1)
        filled = int(bar_width * np.clip(score, 0.0, 100.0) / 100.0)
        cv2.rectangle(canvas, (start_x, row_y - 16), (start_x + filled, row_y + 4), color, -1)
        cv2.putText(
            canvas,
            f"{score:.2f}",
            (start_x + bar_width + 14, row_y),
            font,
            0.5,
            (50, 50, 50),
            1,
            cv2.LINE_AA,
        )
        row_y += 58

    cv2.putText(
        canvas,
        f"final_score: {score_breakdown.moisture_score:.2f}",
        (20, 228),
        font,
        0.7,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return canvas
