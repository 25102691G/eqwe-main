"""Tongue phase-1 analysis adapter for the Flask service.

This module intentionally avoids importing the standalone FastAPI entry point so
the skin service can reuse the phase-1 tongue algorithm without depending on
FastAPI at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

SKIN_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(SKIN_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(SKIN_PROJECT_ROOT))

from tongue_diagnosis.app.demo_report import build_demo_report_payload
from tongue_diagnosis.app.schemas import (
    AssistanceEvidence,
    AssistanceQuality,
    AssistanceSummary,
    BoundingBox,
    CrackObservation,
    DemoReport,
    HsvColor,
    ImageSize,
    LabColor,
    MoistureScoreBreakdown,
    MoistureVisualizations,
    RegionCoat,
    RegionColor,
    RegionDivision,
    RegionMoisture,
    RgbColor,
    SegmentResponse,
    SegmentationQuality,
    TongueCoat,
    TongueImageAssistance,
    TongueMoisture,
)
from tongue_diagnosis.app.segmenter import (
    REGION_LABELS,
    RegionColorMeasurement,
    RegionMoistureMeasurement,
    SegmentationArtifacts,
    SegmentationQualityMetrics,
    TongueMoistureMeasurement,
    TongueSegmenter,
    _build_expert_label_template,
    _to_jsonable_dataclass,
)
from tongue_diagnosis.app.tongue_coat import RegionCoatMeasurement, TongueCoatMeasurement
from tongue_diagnosis.app.tongue_moisture import build_score_breakdown_from_measurement
from tongue_diagnosis.app.tongue_moisture_documentation import (
    build_moisture_explanation,
    translate_quality_reasons,
)
from tongue_diagnosis.app.tongue_moisture_visualization import (
    MoistureVisualizationArtifacts,
    build_moisture_visualizations,
)

DEFAULT_MODEL_PATH = SKIN_PROJECT_ROOT / "api" / "models" / "best_epoch_weights.pth.onnx"
DEFAULT_OUTPUT_DIR = SKIN_PROJECT_ROOT / "upload_files"
ASSISTANCE_DISCLAIMER = "本结果仅基于舌象图片生成健康辅助参考，不作为医学诊断依据。"


@dataclass(frozen=True)
class TonguePhase1Analysis:
    """Complete phase-1 tongue analysis plus upload-ready artifacts."""

    response: dict[str, Any]
    artifacts: SegmentationArtifacts
    full_mask_image: np.ndarray
    analysis_payload: dict[str, Any]
    expert_label_template: dict[str, object]


class TonguePhase1Service:
    """Run the phase-1 tongue algorithm and build the mobile response payload."""

    def __init__(
        self,
        *,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        segmenter: TongueSegmenter | None = None,
    ) -> None:
        self.segmenter = segmenter or TongueSegmenter(
            model_path=model_path,
            output_dir=output_dir,
        )

    @property
    def model_path(self) -> Path:
        """Return the configured ONNX model path."""

        return Path(self.segmenter.model_path)

    def is_model_available(self) -> bool:
        """Return whether the configured ONNX model exists locally."""

        return self.segmenter.is_model_available()

    def analyze(
        self,
        image: np.ndarray,
        *,
        file_name: str,
        include_images: bool = False,
        include_visualizations: bool = False,
        generate_visualizations: bool = False,
    ) -> TonguePhase1Analysis:
        """Run phase-1 analysis and return JSON-safe response data."""

        artifacts = self.segmenter.segment_image(image)

        if (
            (include_visualizations or generate_visualizations)
            and artifacts.cropped_image is not None
            and artifacts.region_masks is not None
        ):
            moisture_visualizations = build_moisture_visualizations(
                image=artifacts.cropped_image,
                tongue_mask=artifacts.mask_image > 0,
                region_masks=artifacts.region_masks,
                region_measurements=artifacts.region_moisture,
                tongue_measurement=artifacts.tongue_moisture,
            )
            artifacts = replace(artifacts, moisture_visualizations=moisture_visualizations)

        response = self._build_response(
            image=image,
            file_name=file_name,
            artifacts=artifacts,
            include_images=include_images,
            include_visualizations=include_visualizations,
        )
        full_mask = build_full_mask_image(image.shape, artifacts)
        return TonguePhase1Analysis(
            response=response.model_dump(mode="json"),
            artifacts=artifacts,
            full_mask_image=full_mask,
            analysis_payload=build_analysis_payload(artifacts),
            expert_label_template=dict(_build_expert_label_template()),
        )

    def _build_response(
        self,
        *,
        image: np.ndarray,
        file_name: str,
        artifacts: SegmentationArtifacts,
        include_images: bool,
        include_visualizations: bool,
    ) -> SegmentResponse:
        """Build the phase-1 API response contract from segmenter artifacts."""

        segmented_image_base64 = None
        mask_image_base64 = None
        if include_images:
            segmented_image_base64 = self.segmenter.encode_image_to_base64(artifacts.segmented_image)
            mask_image_base64 = self.segmenter.encode_image_to_base64(artifacts.mask_image)

        moisture_visualizations = None
        if include_visualizations:
            moisture_visualizations = build_moisture_visualization_response(
                self.segmenter,
                artifacts.moisture_visualizations,
            )

        return SegmentResponse(
            status="success",
            message="tongue phase-1 analysis complete",
            file_name=file_name,
            image_size=ImageSize(width=int(image.shape[1]), height=int(image.shape[0])),
            bounding_box=BoundingBox(**artifacts.bounding_box),
            region_division=build_region_division(),
            segmentation_quality=build_segmentation_quality(artifacts.segmentation_quality),
            tongue_color=None if artifacts.tongue_color is None else build_region_color(artifacts.tongue_color),
            region_colors=[build_region_color(item) for item in artifacts.region_colors],
            tongue_coat=build_tongue_coat(artifacts.tongue_coat),
            region_coat=[build_region_coat(item) for item in artifacts.region_coat],
            region_moisture=[build_region_moisture(item) for item in artifacts.region_moisture],
            tongue_moisture=build_tongue_moisture(artifacts.tongue_moisture),
            crack_observation=build_crack_observation(artifacts.tongue_moisture),
            tongue_image_assistance=build_tongue_image_assistance(
                tongue_color=artifacts.tongue_color,
                region_colors=artifacts.region_colors,
                tongue_coat=artifacts.tongue_coat,
                tongue_moisture=artifacts.tongue_moisture,
                segmentation_quality=artifacts.segmentation_quality,
            ),
            demo_report=build_demo_report(
                tongue_color=artifacts.tongue_color,
                region_colors=artifacts.region_colors,
                tongue_coat=artifacts.tongue_coat,
                region_coat=artifacts.region_coat,
                tongue_moisture=artifacts.tongue_moisture,
                region_moisture=artifacts.region_moisture,
                segmentation_quality=artifacts.segmentation_quality,
            ),
            moisture_visualizations=moisture_visualizations,
            segmented_image_base64=segmented_image_base64,
            mask_image_base64=mask_image_base64,
            saved_files=[],
        )


def build_full_mask_image(image_shape: tuple[int, ...], artifacts: SegmentationArtifacts) -> np.ndarray:
    """Restore the cropped tongue mask into the original image coordinate space."""

    full_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    bbox = artifacts.bounding_box
    x_start = max(0, int(bbox["x"]) - 30)
    y_start = max(0, int(bbox["y"]) - 30)
    mask_height, mask_width = artifacts.mask_image.shape[:2]
    y_end = min(full_mask.shape[0], y_start + mask_height)
    x_end = min(full_mask.shape[1], x_start + mask_width)
    if y_end <= y_start or x_end <= x_start:
        return full_mask

    crop_height = y_end - y_start
    crop_width = x_end - x_start
    full_mask[y_start:y_end, x_start:x_end] = artifacts.mask_image[:crop_height, :crop_width]
    return full_mask


def build_analysis_payload(artifacts: SegmentationArtifacts) -> dict[str, Any]:
    """Build the structured analysis JSON saved to object storage."""

    return {
        "bounding_box": artifacts.bounding_box,
        "segmentation_quality": _to_jsonable_dataclass(artifacts.segmentation_quality),
        "tongue_color": _to_jsonable_dataclass(artifacts.tongue_color),
        "region_colors": _to_jsonable_dataclass(artifacts.region_colors),
        "tongue_coat": _to_jsonable_dataclass(artifacts.tongue_coat),
        "region_coat": _to_jsonable_dataclass(artifacts.region_coat),
        "tongue_moisture": _to_jsonable_dataclass(artifacts.tongue_moisture),
        "region_moisture": _to_jsonable_dataclass(artifacts.region_moisture),
        "demo_report": build_demo_report_payload(
            tongue_color=artifacts.tongue_color,
            region_colors=artifacts.region_colors,
            tongue_coat=artifacts.tongue_coat,
            region_coat=artifacts.region_coat,
            tongue_moisture=artifacts.tongue_moisture,
            region_moisture=artifacts.region_moisture,
            segmentation_quality=artifacts.segmentation_quality,
        ),
    }


def build_region_division() -> RegionDivision:
    """Return the standard five-region overlay description."""

    return RegionDivision(
        description="The tongue area is partitioned into a center zone plus left, top, right, and bottom zones.",
        labels=REGION_LABELS,
    )


def build_region_color(measurement: RegionColorMeasurement) -> RegionColor:
    """Convert one internal region-color measurement into the API schema."""

    return RegionColor(
        region_id=measurement.region_id,
        region_name=measurement.region_name,
        pixel_count=measurement.pixel_count,
        coverage_ratio=measurement.coverage_ratio,
        representative_rgb=RgbColor(
            r=measurement.representative_rgb[0],
            g=measurement.representative_rgb[1],
            b=measurement.representative_rgb[2],
        ),
        representative_hsv=HsvColor(
            h=measurement.representative_hsv[0],
            s=measurement.representative_hsv[1],
            v=measurement.representative_hsv[2],
        ),
        mean_lab=LabColor(
            l=measurement.mean_lab[0],
            a=measurement.mean_lab[1],
            b=measurement.mean_lab[2],
        ),
        representative_hex=measurement.representative_hex,
        color_name=measurement.color_name,
    )


def build_tongue_coat(measurement: TongueCoatMeasurement | None) -> TongueCoat | None:
    """Convert one internal whole-tongue coat measurement into the API schema."""

    if measurement is None:
        return None

    return TongueCoat(
        coat_visibility=measurement.coat_visibility,
        coat_coverage_ratio=measurement.coat_coverage_ratio,
        coat_color_tendency=measurement.coat_color_tendency,
        coat_thickness_tendency=measurement.coat_thickness_tendency,
        white_coat_ratio=measurement.white_coat_ratio,
        yellow_coat_ratio=measurement.yellow_coat_ratio,
        gray_black_coat_ratio=measurement.gray_black_coat_ratio,
    )


def build_region_coat(measurement: RegionCoatMeasurement) -> RegionCoat:
    """Convert one internal region-coat measurement into the API schema."""

    return RegionCoat(
        region_id=measurement.region_id,
        region_name=measurement.region_name,
        pixel_count=measurement.pixel_count,
        coverage_ratio=measurement.coverage_ratio,
        coat_coverage_ratio=measurement.coat_coverage_ratio,
        coat_color_tendency=measurement.coat_color_tendency,
        coat_thickness_tendency=measurement.coat_thickness_tendency,
    )


def build_score_breakdown_response(
    measurement: RegionMoistureMeasurement | TongueMoistureMeasurement,
) -> MoistureScoreBreakdown:
    """Convert one measurement into the score-breakdown schema."""

    breakdown = build_score_breakdown_from_measurement(measurement)
    return MoistureScoreBreakdown(
        gloss_area_score=breakdown.gloss_area_score,
        highlight_blob_score=breakdown.highlight_blob_score,
        crack_score_inverse=breakdown.crack_score_inverse,
        moisture_score=breakdown.moisture_score,
    )


def build_region_moisture(measurement: RegionMoistureMeasurement) -> RegionMoisture:
    """Convert one internal region-moisture measurement into the API schema."""

    return RegionMoisture(
        region_id=measurement.region_id,
        region_name=measurement.region_name,
        pixel_count=measurement.pixel_count,
        coverage_ratio=measurement.coverage_ratio,
        gloss_area_ratio=measurement.gloss_area_ratio,
        highlight_blob_count=measurement.highlight_blob_count,
        highlight_blob_max_ratio=measurement.highlight_blob_max_ratio,
        highlight_blob_mean_area=measurement.highlight_blob_mean_area,
        crack_length_ratio=measurement.crack_length_ratio,
        crack_area_ratio=measurement.crack_area_ratio,
        overexposed_ratio=measurement.overexposed_ratio,
        moisture_score=measurement.moisture_score,
        moisture_label=measurement.moisture_label,
        moisture_tendency=_describe_moisture_tendency(measurement.moisture_label),
        moisture_explanation=build_moisture_explanation(measurement),
        score_breakdown=build_score_breakdown_response(measurement),
    )


def build_tongue_moisture(measurement: TongueMoistureMeasurement | None) -> TongueMoisture | None:
    """Convert one image-level moisture measurement into the API schema."""

    if measurement is None:
        return None

    return TongueMoisture(
        moisture_score=measurement.moisture_score,
        moisture_label=measurement.moisture_label,
        moisture_tendency=_describe_moisture_tendency(measurement.moisture_label),
        moisture_explanation=build_moisture_explanation(measurement),
        quality_passed=measurement.quality_passed,
        quality_reasons=list(measurement.quality_reasons),
        quality_reasons_zh=list(translate_quality_reasons(measurement.quality_reasons)),
        focus_score=measurement.focus_score,
        overexposed_ratio=measurement.overexposed_ratio,
        segmentation_coverage=measurement.segmentation_coverage,
        gloss_area_ratio=measurement.gloss_area_ratio,
        highlight_blob_count=measurement.highlight_blob_count,
        highlight_blob_max_ratio=measurement.highlight_blob_max_ratio,
        highlight_blob_mean_area=measurement.highlight_blob_mean_area,
        crack_length_ratio=measurement.crack_length_ratio,
        crack_area_ratio=measurement.crack_area_ratio,
        score_breakdown=build_score_breakdown_response(measurement),
    )


def build_crack_observation(measurement: TongueMoistureMeasurement | None) -> CrackObservation:
    """Build a standalone crack candidate observation from moisture metrics."""

    if measurement is None:
        return CrackObservation(
            crack_level="无法判断",
            crack_length_ratio=None,
            crack_area_ratio=None,
            confidence="low",
        )
    if not measurement.quality_passed:
        return CrackObservation(
            crack_level="无法判断",
            crack_length_ratio=measurement.crack_length_ratio,
            crack_area_ratio=measurement.crack_area_ratio,
            confidence="low",
        )

    crack_ratio = max(measurement.crack_length_ratio, measurement.crack_area_ratio)
    if crack_ratio >= 0.03:
        crack_level = "明显候选"
        confidence = "medium"
    elif crack_ratio >= 0.01:
        crack_level = "轻度候选"
        confidence = "medium"
    else:
        crack_level = "无明显候选"
        confidence = "high"

    return CrackObservation(
        crack_level=crack_level,
        crack_length_ratio=measurement.crack_length_ratio,
        crack_area_ratio=measurement.crack_area_ratio,
        confidence=confidence,
    )


def build_segmentation_quality(metrics: SegmentationQualityMetrics | None) -> SegmentationQuality | None:
    """Convert internal segmentation metrics into the API schema."""

    if metrics is None:
        return None

    return SegmentationQuality(
        tongue_area_ratio=metrics.tongue_area_ratio,
        bbox_touches_edge=metrics.bbox_touches_edge,
        region_coverage_ratios=metrics.region_coverage_ratios,
    )


def build_moisture_visualization_response(
    segmenter: TongueSegmenter,
    visualizations: MoistureVisualizationArtifacts | None,
) -> MoistureVisualizations | None:
    """Convert optional moisture visualizations into the API schema."""

    if visualizations is None:
        return None

    return MoistureVisualizations(
        gloss_mask_base64=segmenter.encode_image_to_base64(visualizations.gloss_mask),
        gloss_overlay_base64=segmenter.encode_image_to_base64(visualizations.gloss_overlay),
        crack_mask_base64=segmenter.encode_image_to_base64(visualizations.crack_mask),
        crack_skeleton_base64=segmenter.encode_image_to_base64(visualizations.crack_skeleton),
        crack_overlay_base64=segmenter.encode_image_to_base64(visualizations.crack_overlay),
        overexposed_mask_base64=segmenter.encode_image_to_base64(visualizations.overexposed_mask),
        overexposed_overlay_base64=segmenter.encode_image_to_base64(visualizations.overexposed_overlay),
        moisture_heatmap_base64=segmenter.encode_image_to_base64(visualizations.moisture_heatmap),
        score_breakdown_image_base64=segmenter.encode_image_to_base64(visualizations.score_breakdown_image),
    )


def build_tongue_image_assistance(
    *,
    tongue_color: RegionColorMeasurement | None,
    region_colors: tuple[RegionColorMeasurement, ...],
    tongue_coat: TongueCoatMeasurement | None,
    tongue_moisture: TongueMoistureMeasurement | None,
    segmentation_quality: SegmentationQualityMetrics | None,
) -> TongueImageAssistance:
    """Build the product-facing health-assistance summary."""

    quality = _build_assistance_quality(tongue_moisture)
    summary = _build_assistance_summary(
        tongue_color=tongue_color,
        region_colors=region_colors,
        tongue_coat=tongue_coat,
        tongue_moisture=tongue_moisture,
        quality_passed=quality.passed,
    )
    evidence = _build_assistance_evidence(tongue_moisture, segmentation_quality, tongue_coat)
    return TongueImageAssistance(
        positioning="health_assistance",
        quality=quality,
        summary=summary,
        evidence=evidence,
        disclaimer=ASSISTANCE_DISCLAIMER,
    )


def build_demo_report(
    *,
    tongue_color: RegionColorMeasurement | None,
    region_colors: tuple[RegionColorMeasurement, ...],
    tongue_coat: TongueCoatMeasurement | None,
    region_coat: tuple[RegionCoatMeasurement, ...],
    tongue_moisture: TongueMoistureMeasurement | None,
    region_moisture: tuple[RegionMoistureMeasurement, ...],
    segmentation_quality: SegmentationQualityMetrics | None,
) -> DemoReport:
    """Build the front-end-oriented demo report."""

    return DemoReport(
        **build_demo_report_payload(
            tongue_color=tongue_color,
            region_colors=region_colors,
            tongue_coat=tongue_coat,
            region_coat=region_coat,
            tongue_moisture=tongue_moisture,
            region_moisture=region_moisture,
            segmentation_quality=segmentation_quality,
        )
    )


def _build_assistance_quality(tongue_moisture: TongueMoistureMeasurement | None) -> AssistanceQuality:
    """Build product-facing image quality guidance."""

    if tongue_moisture is None:
        return AssistanceQuality(
            passed=False,
            level="unusable",
            reasons=["舌体区域未稳定检出"],
            suggestion="当前图片暂不适合进行舌象图像特征参考，建议重新拍摄。",
        )

    reasons = list(translate_quality_reasons(tongue_moisture.quality_reasons))
    if tongue_moisture.quality_passed:
        return AssistanceQuality(
            passed=True,
            level="usable",
            reasons=[],
            suggestion="当前图片可用于舌象图像特征参考。",
        )

    return AssistanceQuality(
        passed=False,
        level="needs_retake",
        reasons=reasons,
        suggestion="当前图片质量会影响舌象图像特征参考，建议优先重新拍摄。",
    )


def _build_assistance_summary(
    *,
    tongue_color: RegionColorMeasurement | None,
    region_colors: tuple[RegionColorMeasurement, ...],
    tongue_coat: TongueCoatMeasurement | None,
    tongue_moisture: TongueMoistureMeasurement | None,
    quality_passed: bool,
) -> AssistanceSummary:
    """Build product-facing feature observations."""

    if tongue_moisture is None or not quality_passed:
        return AssistanceSummary(
            color_tendency="无法稳定判断",
            moisture_tendency="无法稳定判断",
            coat_visibility="暂未分析",
            main_observations=["当前图片质量不足，建议重新拍摄后再查看舌象图像特征参考。"],
        )

    color_tendency = tongue_color.color_name if tongue_color is not None else _summarize_color_tendency(region_colors)
    moisture_tendency = _describe_moisture_tendency(tongue_moisture.moisture_label)
    coat_visibility = "无法判断" if tongue_coat is None else tongue_coat.coat_visibility
    observations = [
        f"舌色整体有{color_tendency}倾向。",
        f"舌面润泽度{moisture_tendency}。",
    ]
    if tongue_coat is not None and tongue_coat.coat_visibility != "无明显":
        observations.append(
            f"可见{tongue_coat.coat_visibility}舌苔候选区域，苔色有{tongue_coat.coat_color_tendency}倾向。"
        )
    else:
        observations.append("未见明显舌苔候选区域。")
    return AssistanceSummary(
        color_tendency=color_tendency,
        moisture_tendency=moisture_tendency,
        coat_visibility=coat_visibility,
        main_observations=observations,
    )


def _build_assistance_evidence(
    tongue_moisture: TongueMoistureMeasurement | None,
    segmentation_quality: SegmentationQualityMetrics | None,
    tongue_coat: TongueCoatMeasurement | None,
) -> AssistanceEvidence:
    """Build product-facing evidence metrics."""

    tongue_area_ratio = None if segmentation_quality is None else segmentation_quality.tongue_area_ratio
    bbox_touches_edge = None if segmentation_quality is None else segmentation_quality.bbox_touches_edge
    coat_coverage_ratio = None if tongue_coat is None else tongue_coat.coat_coverage_ratio

    if tongue_moisture is None:
        return AssistanceEvidence(
            tongue_area_ratio=tongue_area_ratio,
            bbox_touches_edge=bbox_touches_edge,
            coat_coverage_ratio=coat_coverage_ratio,
        )

    return AssistanceEvidence(
        tongue_area_ratio=tongue_area_ratio,
        bbox_touches_edge=bbox_touches_edge,
        focus_score=tongue_moisture.focus_score,
        overexposed_ratio=tongue_moisture.overexposed_ratio,
        segmentation_coverage=tongue_moisture.segmentation_coverage,
        gloss_area_ratio=tongue_moisture.gloss_area_ratio,
        crack_length_ratio=tongue_moisture.crack_length_ratio,
        coat_coverage_ratio=coat_coverage_ratio,
    )


def _summarize_color_tendency(region_colors: tuple[RegionColorMeasurement, ...]) -> str:
    """Summarize region colors into one coarse whole-tongue tendency."""

    if not region_colors:
        return "无法稳定判断"

    weighted_scores: dict[str, float] = {}
    for measurement in region_colors:
        weighted_scores[measurement.color_name] = weighted_scores.get(measurement.color_name, 0.0) + (
            measurement.coverage_ratio
        )
    return max(weighted_scores.items(), key=lambda item: item[1])[0]


def _describe_moisture_tendency(moisture_label: str) -> str:
    """Convert the internal moisture label into product-facing wording."""

    if moisture_label == "燥":
        return "偏低"
    if moisture_label == "润":
        return "偏高"
    return "处于中间状态"


__all__ = [
    "DEFAULT_MODEL_PATH",
    "TonguePhase1Analysis",
    "TonguePhase1Service",
    "build_analysis_payload",
    "build_full_mask_image",
]
