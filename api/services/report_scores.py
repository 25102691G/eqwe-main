"""Helpers for converting raw detector counts into report-friendly scores."""

from __future__ import annotations

import math


def _clamp_quality_score(value: float) -> int:
    """Clamp a quality score into the 0-100 range."""
    return max(0, min(100, int(round(value))))


def _area_units_per_10k(skin_area_pixels: int) -> float:
    """Normalize skin area into 10k-pixel units with a safe floor."""
    if skin_area_pixels <= 0:
        return 1.0
    return max(skin_area_pixels / 10000.0, 1.0)


def calculate_skin_tone_score(*, stain_count: int, skin_area_pixels: int) -> int:
    """Estimate a skin-tone uniformity score from stain density.

    The score intentionally ignores `ITA` because ITA reflects skin-tone category,
    not skin-quality. Lower stain density yields a higher quality score.
    """
    stain_density_per_10k = max(stain_count, 0) / _area_units_per_10k(skin_area_pixels)
    penalty = min(100.0, stain_density_per_10k * 12.0)
    return _clamp_quality_score(100.0 - penalty)


def calculate_smoothness_score(
    *,
    acne_group_total: int,
    blackheads_count: int,
    pores_count: int,
    skin_area_pixels: int,
) -> int:
    """Estimate a smoothness score from detected surface-issue density.

    Acne-like findings carry the highest weight, followed by blackheads and then
    pores. A logarithmic penalty keeps the score usable even when point counts
    spike on higher-resolution images.
    """
    weighted_issue_count = (
        max(acne_group_total, 0) * 4.0
        + max(blackheads_count, 0) * 2.0
        + max(pores_count, 0) * 0.5
    )
    weighted_density_per_10k = weighted_issue_count / _area_units_per_10k(skin_area_pixels)
    penalty = min(100.0, 12.0 * math.log1p(weighted_density_per_10k))
    return _clamp_quality_score(100.0 - penalty)


__all__ = ["calculate_skin_tone_score", "calculate_smoothness_score"]
