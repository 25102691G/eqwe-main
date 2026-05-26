"""Front-end demo report aggregation from first-phase tongue image features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tongue_diagnosis.app.tongue_coat import RegionCoatMeasurement, TongueCoatMeasurement
from tongue_diagnosis.app.tongue_color import RegionColorMeasurement
from tongue_diagnosis.app.tongue_moisture import RegionMoistureMeasurement, TongueMoistureMeasurement
from tongue_diagnosis.app.tongue_moisture_documentation import translate_quality_reasons

if TYPE_CHECKING:
    from tongue_diagnosis.app.segmenter import SegmentationQualityMetrics

REPORT_TYPE = "tongue_image_health_assistance_demo"
DISCLAIMER = (
    "本结果为基于舌象图片的健康辅助参考，不作为医学诊断依据。"
    "九大体质结果为倾向参考，需要结合问卷、生活习惯、症状信息和专家复核。"
)


@dataclass(frozen=True)
class _FeatureFlags:
    """Coarse rule flags derived from first-phase image features."""

    color_light_red: bool
    color_pale_white: bool
    color_red_or_crimson: bool
    color_purple_dark: bool
    moisture_low: bool
    moisture_middle: bool
    moisture_high: bool
    crack_none: bool
    crack_visible: bool
    crack_obvious: bool
    coat_none_or_light: bool
    coat_visible: bool
    coat_more_visible: bool
    coat_white: bool
    coat_yellow: bool
    coat_thick: bool
    local_red_or_uneven: bool


def build_demo_report_payload(
    *,
    tongue_color: RegionColorMeasurement | None,
    region_colors: tuple[RegionColorMeasurement, ...],
    tongue_coat: TongueCoatMeasurement | None,
    region_coat: tuple[RegionCoatMeasurement, ...],
    tongue_moisture: TongueMoistureMeasurement | None,
    region_moisture: tuple[RegionMoistureMeasurement, ...],
    segmentation_quality: SegmentationQualityMetrics | None,
) -> dict[str, object]:
    """Build the front-end demo report payload from current algorithm outputs.

    Args:
        tongue_color: Whole-tongue color measurement.
        region_colors: Region-level color measurements.
        tongue_coat: Whole-tongue coat candidate measurement.
        region_coat: Region-level coat candidate measurements.
        tongue_moisture: Whole-tongue moisture measurement.
        region_moisture: Region-level moisture measurements.
        segmentation_quality: Segmentation stability metrics.

    Returns:
        A JSON-serializable report payload aligned with `DemoReport`.
    """

    quality_gate = _build_quality_gate(tongue_moisture, segmentation_quality)
    if not bool(quality_gate["passed"]):
        return {
            "report_type": REPORT_TYPE,
            "quality_gate": quality_gate,
            "feature_cards": _build_feature_cards(
                tongue_color=tongue_color,
                region_colors=region_colors,
                tongue_coat=tongue_coat,
                region_coat=region_coat,
                tongue_moisture=tongue_moisture,
                region_moisture=region_moisture,
                segmentation_quality=segmentation_quality,
            ),
            "constitution_tendencies": [],
            "primary_tendencies": [],
            "analysis_summary": "当前图片质量会影响舌象图像特征稳定性，建议重拍后再查看九大体质倾向参考。",
            "frontend_sections": _build_frontend_sections(include_constitution=False),
            "disclaimer": DISCLAIMER,
        }

    flags = _build_feature_flags(
        tongue_color=tongue_color,
        region_colors=region_colors,
        tongue_coat=tongue_coat,
        tongue_moisture=tongue_moisture,
    )
    all_tendencies = _build_constitution_tendencies(flags)
    visible_tendencies = [item for item in all_tendencies if float(item["score"]) >= 40.0]
    primary_tendencies = [str(item["constitution"]) for item in visible_tendencies if float(item["score"]) >= 60.0][:3]
    return {
        "report_type": REPORT_TYPE,
        "quality_gate": quality_gate,
        "feature_cards": _build_feature_cards(
            tongue_color=tongue_color,
            region_colors=region_colors,
            tongue_coat=tongue_coat,
            region_coat=region_coat,
            tongue_moisture=tongue_moisture,
            region_moisture=region_moisture,
            segmentation_quality=segmentation_quality,
        ),
        "constitution_tendencies": visible_tendencies,
        "primary_tendencies": primary_tendencies,
        "analysis_summary": _build_analysis_summary(
            tongue_color=tongue_color,
            tongue_coat=tongue_coat,
            tongue_moisture=tongue_moisture,
            primary_tendencies=primary_tendencies,
            visible_tendencies=visible_tendencies,
        ),
        "frontend_sections": _build_frontend_sections(include_constitution=bool(visible_tendencies)),
        "disclaimer": DISCLAIMER,
    }


def _build_quality_gate(
    tongue_moisture: TongueMoistureMeasurement | None,
    segmentation_quality: SegmentationQualityMetrics | None,
) -> dict[str, object]:
    """Build the report-level quality gate."""

    if tongue_moisture is None:
        return {
            "passed": False,
            "level": "unusable",
            "reasons": ["未稳定检测到舌体区域"],
            "suggestion": "建议重新拍摄后再查看舌象图像辅助报告。",
        }

    reasons = list(translate_quality_reasons(tongue_moisture.quality_reasons))
    if segmentation_quality is not None and segmentation_quality.bbox_touches_edge:
        reasons.append("舌体区域贴近图片边缘")

    if tongue_moisture.quality_passed and not reasons:
        return {
            "passed": True,
            "level": "usable",
            "reasons": [],
            "suggestion": "当前图片可用于舌象图像特征参考。",
        }

    return {
        "passed": False,
        "level": "needs_retake",
        "reasons": reasons,
        "suggestion": "当前图片质量会影响结果稳定性，建议优先重新拍摄。",
    }


def _build_feature_cards(
    *,
    tongue_color: RegionColorMeasurement | None,
    region_colors: tuple[RegionColorMeasurement, ...],
    tongue_coat: TongueCoatMeasurement | None,
    region_coat: tuple[RegionCoatMeasurement, ...],
    tongue_moisture: TongueMoistureMeasurement | None,
    region_moisture: tuple[RegionMoistureMeasurement, ...],
    segmentation_quality: SegmentationQualityMetrics | None,
) -> list[dict[str, object]]:
    """Build front-end metric cards from first-phase outputs."""

    return [
        _quality_feature_card(tongue_moisture, segmentation_quality),
        _color_feature_card(tongue_color, region_colors),
        _moisture_feature_card(tongue_moisture, region_moisture),
        _coat_feature_card(tongue_coat, region_coat),
        _crack_feature_card(tongue_moisture),
        _region_feature_card(region_colors, region_coat, region_moisture),
    ]


def _quality_feature_card(
    tongue_moisture: TongueMoistureMeasurement | None,
    segmentation_quality: SegmentationQualityMetrics | None,
) -> dict[str, object]:
    """Build the image-quality card."""

    if tongue_moisture is None:
        return {
            "key": "image_quality",
            "title": "图片质量",
            "value": "建议重拍",
            "level": "warning",
            "evidence": ["未稳定检测到舌体区域"],
        }

    evidence = [
        f"清晰度分数 {tongue_moisture.focus_score}",
        f"过曝比例 {tongue_moisture.overexposed_ratio}",
        f"分割覆盖比例 {tongue_moisture.segmentation_coverage}",
    ]
    if segmentation_quality is not None:
        evidence.append(f"舌体面积占比 {segmentation_quality.tongue_area_ratio}")
    return {
        "key": "image_quality",
        "title": "图片质量",
        "value": "可分析" if tongue_moisture.quality_passed else "建议重拍",
        "level": "good" if tongue_moisture.quality_passed else "warning",
        "evidence": evidence,
    }


def _color_feature_card(
    tongue_color: RegionColorMeasurement | None,
    region_colors: tuple[RegionColorMeasurement, ...],
) -> dict[str, object]:
    """Build the tongue-color card."""

    if tongue_color is None:
        return {
            "key": "tongue_color",
            "title": "舌色倾向",
            "value": "无法判断",
            "level": "muted",
            "evidence": ["舌色候选不足"],
        }

    region_summary = "、".join(f"{item.region_name}:{item.color_name}" for item in region_colors[:5])
    return {
        "key": "tongue_color",
        "title": "舌色倾向",
        "value": tongue_color.color_name,
        "level": "info",
        "evidence": [
            f"整舌代表色 {tongue_color.representative_hex}",
            f"整舌 Lab 均值 L={tongue_color.mean_lab[0]}, a={tongue_color.mean_lab[1]}, b={tongue_color.mean_lab[2]}",
            f"五区颜色参考 {region_summary}" if region_summary else "暂无五区颜色参考",
        ],
    }


def _moisture_feature_card(
    tongue_moisture: TongueMoistureMeasurement | None,
    region_moisture: tuple[RegionMoistureMeasurement, ...],
) -> dict[str, object]:
    """Build the moisture card."""

    if tongue_moisture is None:
        return {
            "key": "moisture",
            "title": "舌面润泽度",
            "value": "无法判断",
            "level": "muted",
            "evidence": ["润燥候选不足"],
        }

    region_summary = "、".join(f"{item.region_name}:{item.moisture_label}" for item in region_moisture[:5])
    return {
        "key": "moisture",
        "title": "舌面润泽度",
        "value": _moisture_tendency(tongue_moisture),
        "level": "good" if 40.0 <= tongue_moisture.moisture_score < 70.0 else "info",
        "evidence": [
            f"润燥分 {tongue_moisture.moisture_score}",
            f"高光面积比例 {tongue_moisture.gloss_area_ratio}",
            f"高光块数量 {tongue_moisture.highlight_blob_count}",
            f"五区润燥参考 {region_summary}" if region_summary else "暂无五区润燥参考",
        ],
    }


def _coat_feature_card(
    tongue_coat: TongueCoatMeasurement | None,
    region_coat: tuple[RegionCoatMeasurement, ...],
) -> dict[str, object]:
    """Build the tongue-coat card."""

    if tongue_coat is None:
        return {
            "key": "tongue_coat",
            "title": "舌苔候选",
            "value": "无法判断",
            "level": "muted",
            "evidence": ["舌苔候选不足"],
        }

    region_summary = "、".join(f"{item.region_name}:{item.coat_coverage_ratio}" for item in region_coat[:5])
    return {
        "key": "tongue_coat",
        "title": "舌苔候选",
        "value": f"{tongue_coat.coat_visibility}，{tongue_coat.coat_color_tendency}",
        "level": "info" if tongue_coat.coat_coverage_ratio >= 0.03 else "muted",
        "evidence": [
            f"舌苔候选覆盖比例 {tongue_coat.coat_coverage_ratio}",
            f"白苔候选比例 {tongue_coat.white_coat_ratio}",
            f"黄苔候选比例 {tongue_coat.yellow_coat_ratio}",
            f"薄厚倾向 {tongue_coat.coat_thickness_tendency}",
            f"五区舌苔覆盖参考 {region_summary}" if region_summary else "暂无五区舌苔参考",
        ],
    }


def _crack_feature_card(tongue_moisture: TongueMoistureMeasurement | None) -> dict[str, object]:
    """Build the crack-candidate card."""

    if tongue_moisture is None:
        return {
            "key": "crack_observation",
            "title": "裂纹候选",
            "value": "无法判断",
            "level": "muted",
            "evidence": ["裂纹候选不足"],
        }

    crack_ratio = _crack_ratio(tongue_moisture)
    return {
        "key": "crack_observation",
        "title": "裂纹候选",
        "value": _crack_level(tongue_moisture),
        "level": "warning" if crack_ratio >= 0.01 else "muted",
        "evidence": [
            f"裂纹骨架长度比例 {tongue_moisture.crack_length_ratio}",
            f"裂纹候选面积比例 {tongue_moisture.crack_area_ratio}",
        ],
    }


def _region_feature_card(
    region_colors: tuple[RegionColorMeasurement, ...],
    region_coat: tuple[RegionCoatMeasurement, ...],
    region_moisture: tuple[RegionMoistureMeasurement, ...],
) -> dict[str, object]:
    """Build the region-observation card."""

    evidence = [
        f"颜色区域数 {len(region_colors)}",
        f"舌苔区域数 {len(region_coat)}",
        f"润燥区域数 {len(region_moisture)}",
    ]
    if region_colors:
        evidence.append("五区颜色：" + "、".join(f"{item.region_name}:{item.color_name}" for item in region_colors))
    return {
        "key": "region_observation",
        "title": "区域观察",
        "value": "五区参考已生成" if region_colors else "暂无区域参考",
        "level": "info" if region_colors else "muted",
        "evidence": evidence,
    }


def _build_feature_flags(
    *,
    tongue_color: RegionColorMeasurement | None,
    region_colors: tuple[RegionColorMeasurement, ...],
    tongue_coat: TongueCoatMeasurement | None,
    tongue_moisture: TongueMoistureMeasurement | None,
) -> _FeatureFlags:
    """Convert algorithm measurements into coarse constitution rules."""

    mean_lab = None if tongue_color is None else tongue_color.mean_lab
    lightness = 0.0 if mean_lab is None else mean_lab[0]
    a_value = 0.0 if mean_lab is None else mean_lab[1]
    b_value = 0.0 if mean_lab is None else mean_lab[2]
    region_color_names = [item.color_name for item in region_colors]
    moisture_score = -1.0 if tongue_moisture is None else tongue_moisture.moisture_score
    crack_ratio = 0.0 if tongue_moisture is None else _crack_ratio(tongue_moisture)
    coat_coverage = 0.0 if tongue_coat is None else tongue_coat.coat_coverage_ratio
    white_coat_ratio = 0.0 if tongue_coat is None else tongue_coat.white_coat_ratio
    yellow_coat_ratio = 0.0 if tongue_coat is None else tongue_coat.yellow_coat_ratio

    return _FeatureFlags(
        color_light_red=mean_lab is not None and 45.0 <= lightness < 72.0 and 12.0 <= a_value < 28.0,
        color_pale_white=mean_lab is not None and lightness >= 72.0 and a_value < 12.0,
        color_red_or_crimson=mean_lab is not None and a_value >= 22.0,
        color_purple_dark=mean_lab is not None and ((b_value <= -4.0 and a_value >= 15.0) or lightness < 42.0),
        moisture_low=0.0 <= moisture_score < 40.0,
        moisture_middle=40.0 <= moisture_score < 70.0,
        moisture_high=moisture_score >= 70.0,
        crack_none=crack_ratio < 0.01,
        crack_visible=crack_ratio >= 0.01,
        crack_obvious=crack_ratio >= 0.03,
        coat_none_or_light=coat_coverage < 0.18,
        coat_visible=coat_coverage >= 0.03,
        coat_more_visible=coat_coverage >= 0.18,
        coat_white=white_coat_ratio >= max(0.08, yellow_coat_ratio * 1.5),
        coat_yellow=yellow_coat_ratio >= 0.05 and yellow_coat_ratio >= white_coat_ratio * 0.8,
        coat_thick=coat_coverage >= 0.25,
        local_red_or_uneven=_has_local_red_or_uneven(region_color_names),
    )


def _build_constitution_tendencies(flags: _FeatureFlags) -> list[dict[str, object]]:
    """Build scored constitution tendencies according to the demo planning document."""

    specs = [
        (
            "平和质",
            [
                (flags.color_light_red, 20.0, "舌色淡红"),
                (flags.moisture_middle or flags.moisture_high, 20.0, "润泽度中间或偏高"),
                (flags.crack_none, 15.0, "裂纹无明显候选"),
                (flags.coat_none_or_light, 15.0, "舌苔无明显或少量可见"),
                (flags.coat_white, 10.0, "苔色偏白"),
            ],
            ["问卷平和状态未接入"],
        ),
        (
            "气虚质",
            [
                (flags.color_pale_white, 30.0, "舌色淡白倾向"),
                (flags.coat_white, 15.0, "苔色偏白"),
                (flags.coat_visible, 10.0, "舌苔候选可见"),
                (flags.moisture_middle or flags.moisture_high, 15.0, "润泽度中间或偏高"),
            ],
            ["齿痕未接入", "胖大舌未接入", "疲劳乏力问卷未接入"],
        ),
        (
            "阳虚质",
            [
                (flags.color_pale_white, 30.0, "舌色淡白倾向"),
                (flags.coat_white, 15.0, "苔色偏白"),
                (flags.moisture_middle or flags.moisture_high, 20.0, "润泽度中间或偏高"),
            ],
            ["畏寒怕冷问卷未接入", "胖大舌未接入", "齿痕未接入"],
        ),
        (
            "阴虚质",
            [
                (flags.color_red_or_crimson, 25.0, "舌色偏红或绛"),
                (flags.moisture_low, 25.0, "润泽度偏低"),
                (flags.crack_visible, 20.0, "裂纹候选可见"),
                (not flags.coat_more_visible, 10.0, "少苔或无明显舌苔候选"),
            ],
            ["口干、盗汗、睡眠问卷未接入"],
        ),
        (
            "痰湿质",
            [
                (flags.coat_more_visible, 25.0, "舌苔覆盖较明显"),
                (flags.coat_white, 20.0, "苔色偏白"),
                (flags.coat_thick, 15.0, "舌苔厚薄倾向偏厚"),
                (flags.moisture_high, 15.0, "润泽度偏高"),
            ],
            ["腻苔未接入", "胖大舌未接入", "齿痕未接入", "困重、痰多问卷未接入"],
        ),
        (
            "湿热质",
            [
                (flags.color_red_or_crimson, 20.0, "舌色偏红"),
                (flags.coat_yellow, 30.0, "苔色偏黄"),
                (flags.coat_more_visible, 20.0, "舌苔覆盖较明显"),
            ],
            ["黄腻苔未接入", "口苦、便黏、尿黄问卷未接入"],
        ),
        (
            "血瘀质",
            [
                (flags.color_purple_dark, 35.0, "舌色青紫候选"),
                (flags.crack_obvious, 20.0, "裂纹明显候选"),
            ],
            ["瘀点瘀斑未接入", "舌下络脉未接入"],
        ),
        (
            "气郁质",
            [
                (flags.local_red_or_uneven, 20.0, "局部颜色不均或边缘偏红候选"),
            ],
            ["情绪压力问卷未接入", "胸胁胀满等问卷未接入"],
        ),
        (
            "特禀质",
            [],
            ["过敏史问卷未接入", "鼻炎、荨麻疹等信息未接入", "一期舌图证据不足"],
        ),
    ]

    tendencies = []
    for constitution, rules, missing in specs:
        score = 0.0
        evidence = []
        for matched, weight, label in rules:
            if matched:
                score += weight
                evidence.append(label)
        tendencies.append(
            {
                "constitution": constitution,
                "score": round(min(score, 100.0), 2),
                "level": _score_level(score),
                "evidence": evidence,
                "missing_evidence": missing,
                "confidence": _confidence(score, evidence),
                "note": "仅基于舌象图片的倾向参考，不作为医学诊断依据。",
            }
        )
    return sorted(tendencies, key=lambda item: float(item["score"]), reverse=True)


def _build_analysis_summary(
    *,
    tongue_color: RegionColorMeasurement | None,
    tongue_coat: TongueCoatMeasurement | None,
    tongue_moisture: TongueMoistureMeasurement | None,
    primary_tendencies: list[str],
    visible_tendencies: list[dict[str, object]],
) -> str:
    """Build one natural-language summary for the front-end demo."""

    observations = []
    if tongue_color is not None:
        observations.append(f"舌色{tongue_color.color_name}")
    if tongue_moisture is not None:
        observations.append(f"舌面润泽度{_moisture_tendency(tongue_moisture)}")
    if tongue_coat is not None:
        observations.append(f"舌苔候选{tongue_coat.coat_visibility}、{tongue_coat.coat_color_tendency}")

    prefix = "根据当前舌象图片，系统检测到" + "、".join(observations) + "。" if observations else "当前舌象图片证据有限。"
    if primary_tendencies:
        return f"{prefix}综合图像证据，本次结果更支持{'、'.join(primary_tendencies)}相关倾向参考。"
    if visible_tendencies:
        names = "、".join(str(item["constitution"]) for item in visible_tendencies[:3])
        return f"{prefix}当前存在{names}等轻度倾向参考，但证据仍需结合问卷和专家复核。"
    return f"{prefix}当前九大体质倾向证据不足，建议结合问卷和专家复核。"


def _build_frontend_sections(*, include_constitution: bool) -> list[dict[str, object]]:
    """Build suggested front-end sections."""

    sections = [
        {
            "key": "overview",
            "title": "首页概览",
            "items": ["图片质量", "主要体质倾向参考", "一句话摘要", "免责声明"],
        },
        {
            "key": "evidence",
            "title": "数据支撑",
            "items": ["舌色倾向", "舌苔候选", "舌面润泽度", "裂纹候选", "区域观察"],
        },
    ]
    if include_constitution:
        sections.append(
            {
                "key": "constitution",
                "title": "九大体质倾向参考",
                "items": ["1-3 个主倾向", "每个倾向的证据", "缺失证据", "置信度"],
            }
        )
    sections.append(
        {
            "key": "expert_review",
            "title": "专家复核",
            "items": ["算法结果文件", "专家标注模板", "后续问卷校准"],
        }
    )
    return sections


def _score_level(score: float) -> str:
    """Map a score to the planned tendency level."""

    if score >= 80.0:
        return "明显倾向"
    if score >= 60.0:
        return "中等倾向"
    if score >= 40.0:
        return "轻度倾向"
    return "证据不足"


def _confidence(score: float, evidence: list[str]) -> str:
    """Map image-only evidence into low or medium confidence."""

    if score >= 60.0 and len(evidence) >= 3:
        return "medium"
    return "low"


def _moisture_tendency(tongue_moisture: TongueMoistureMeasurement) -> str:
    """Convert the internal moisture label into demo wording."""

    if tongue_moisture.moisture_score < 40.0:
        return "偏低"
    if tongue_moisture.moisture_score >= 70.0:
        return "偏高"
    return "处于中间状态"


def _crack_ratio(tongue_moisture: TongueMoistureMeasurement) -> float:
    """Return the stronger crack-candidate ratio."""

    return max(tongue_moisture.crack_length_ratio, tongue_moisture.crack_area_ratio)


def _crack_level(tongue_moisture: TongueMoistureMeasurement) -> str:
    """Return a crack-candidate level for the demo report."""

    ratio = _crack_ratio(tongue_moisture)
    if ratio >= 0.03:
        return "明显候选"
    if ratio >= 0.01:
        return "轻度候选"
    return "无明显候选"


def _has_local_red_or_uneven(region_color_names: list[str]) -> bool:
    """Return whether region colors support a weak local-red or uneven-color cue."""

    if len(region_color_names) < 2:
        return False
    red_count = sum(1 for name in region_color_names if name in {"红", "绛"})
    return red_count > 0 or len(set(region_color_names)) >= 3
