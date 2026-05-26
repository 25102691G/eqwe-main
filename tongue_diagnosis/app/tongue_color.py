"""Tongue color analysis helpers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RegionColorMeasurement:
    """Color summary extracted from one tongue region."""

    region_id: str
    region_name: str
    pixel_count: int
    coverage_ratio: float
    representative_rgb: tuple[int, int, int]
    representative_hsv: tuple[int, int, int]
    mean_lab: tuple[float, float, float]
    representative_hex: str
    color_name: str


def _trim_lab_pixels(lab_pixels: np.ndarray) -> np.ndarray:
    """Drop extreme Lab outliers so highlights do not dominate one region color."""

    if lab_pixels.shape[0] < 50:
        return lab_pixels

    lower = np.percentile(lab_pixels, 10.0, axis=0)
    upper = np.percentile(lab_pixels, 90.0, axis=0)
    trimmed = lab_pixels[np.all((lab_pixels >= lower) & (lab_pixels <= upper), axis=1)]
    min_pixels = max(lab_pixels.shape[0] // 5, 25)
    return trimmed if trimmed.shape[0] >= min_pixels else lab_pixels


def _opencv_lab_to_standard(lab_values: np.ndarray) -> tuple[float, float, float]:
    """Convert OpenCV Lab channel values to standard CIELAB ranges."""

    l_value = round(float(lab_values[0]) * 100.0 / 255.0, 2)
    a_value = round(float(lab_values[1]) - 128.0, 2)
    b_value = round(float(lab_values[2]) - 128.0, 2)
    return (l_value, a_value, b_value)


def _lab_to_bgr(lab_values: np.ndarray) -> np.ndarray:
    """Convert one OpenCV Lab color vector back to BGR."""

    lab_image = np.clip(np.rint(lab_values), 0, 255).astype(np.uint8).reshape((1, 1, 3))
    return cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR).reshape(3)


def _describe_tongue_color(mean_lab: tuple[float, float, float]) -> str:
    """Map one representative Lab color to a coarse tongue-color label."""

    l_value, a_value, b_value = mean_lab
    if b_value <= -4.0 and a_value >= 15.0:
        return "青紫"
    if l_value >= 72.0 and a_value < 12.0:
        return "淡白"
    if a_value >= 28.0 and l_value < 45.0:
        return "绛"
    if a_value >= 22.0:
        return "红"
    return "淡红"


def analyze_region_colors(
    image: np.ndarray,
    region_masks: dict[str, np.ndarray],
    *,
    region_output_order: tuple[str, ...],
    region_labels: dict[str, str],
    total_tongue_pixels: int,
) -> tuple[RegionColorMeasurement, ...]:
    """Calculate representative colors for the provided tongue regions.

    Args:
        image: Cropped BGR tongue image without overlays.
        region_masks: Region masks keyed by region identifier.
        region_output_order: Ordered region identifiers to emit.
        region_labels: Human-readable labels for each region identifier.
        total_tongue_pixels: Total number of tongue pixels across all regions.

    Returns:
        One color summary per tongue region. Background pixels are excluded.
    """

    if total_tongue_pixels == 0:
        return ()

    measurements: list[RegionColorMeasurement] = []

    for region_id in region_output_order:
        region_mask = region_masks[region_id]
        pixel_count = int(np.count_nonzero(region_mask))
        if pixel_count == 0:
            continue

        region_pixels = image[region_mask]
        lab_pixels = cv2.cvtColor(
            region_pixels.reshape((-1, 1, 3)),
            cv2.COLOR_BGR2LAB,
        ).reshape((-1, 3)).astype(np.float32)
        filtered_lab = _trim_lab_pixels(lab_pixels)
        mean_lab_cv = filtered_lab.mean(axis=0)
        mean_lab = _opencv_lab_to_standard(mean_lab_cv)

        representative_bgr = _lab_to_bgr(mean_lab_cv)
        representative_rgb = tuple(int(channel) for channel in representative_bgr[::-1])
        representative_hsv = tuple(
            int(channel)
            for channel in cv2.cvtColor(
                representative_bgr.reshape((1, 1, 3)),
                cv2.COLOR_BGR2HSV,
            ).reshape(3)
        )

        measurements.append(
            RegionColorMeasurement(
                region_id=region_id,
                region_name=region_labels[region_id],
                pixel_count=pixel_count,
                coverage_ratio=round(pixel_count / total_tongue_pixels, 4),
                representative_rgb=representative_rgb,
                representative_hsv=representative_hsv,
                mean_lab=mean_lab,
                representative_hex="#{:02x}{:02x}{:02x}".format(*representative_rgb),
                color_name=_describe_tongue_color(mean_lab),
            )
        )

    return tuple(measurements)


def analyze_tongue_color(image: np.ndarray, tongue_mask: np.ndarray) -> RegionColorMeasurement | None:
    """Calculate one whole-tongue representative color measurement.

    Args:
        image: Cropped BGR tongue image without overlays.
        tongue_mask: Binary tongue mask aligned with `image`.

    Returns:
        One whole-tongue color summary, or `None` when no tongue pixels exist.
    """

    total_tongue_pixels = int(np.count_nonzero(tongue_mask))
    if total_tongue_pixels == 0:
        return None

    measurements = analyze_region_colors(
        image,
        {"overall": tongue_mask},
        region_output_order=("overall",),
        region_labels={"overall": "whole tongue"},
        total_tongue_pixels=total_tongue_pixels,
    )
    return measurements[0] if measurements else None
