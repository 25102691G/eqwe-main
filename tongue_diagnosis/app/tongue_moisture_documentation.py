"""Documentation helpers for tongue moisture API responses."""

from __future__ import annotations

from tongue_diagnosis.app.tongue_moisture import (
    MoistureScoreBreakdown,
    RegionMoistureMeasurement,
    TongueMoistureMeasurement,
    build_score_breakdown_from_measurement,
)

MoistureMeasurement = RegionMoistureMeasurement | TongueMoistureMeasurement

# 使用 Unicode 转义是为了规避当前环境里的编码干扰。
# 下面这些值对应的中文分别是：
# - low_focus -> 图像清晰度不足
# - overexposed -> 图像局部过曝
# - low_segmentation_coverage -> 舌体分割不完整
# - small_tongue_region -> 舌体区域过小
# - tongue_touches_edge -> 舌体可能未拍全
# - color_cast -> 图像存在明显偏色
QUALITY_REASON_TEXTS_ZH: dict[str, str] = {
    "low_focus": "\u56fe\u50cf\u6e05\u6670\u5ea6\u4e0d\u8db3",
    "overexposed": "\u56fe\u50cf\u5c40\u90e8\u8fc7\u66dd",
    "low_segmentation_coverage": "\u820c\u4f53\u5206\u5272\u4e0d\u5b8c\u6574",
    "small_tongue_region": "\u820c\u4f53\u533a\u57df\u8fc7\u5c0f",
    "tongue_touches_edge": "\u820c\u4f53\u53ef\u80fd\u672a\u62cd\u5168",
    "color_cast": "\u56fe\u50cf\u5b58\u5728\u660e\u663e\u504f\u8272",
}

# 前端动作建议：
# - low_focus -> 提示重新对焦拍摄
# - overexposed -> 提示避开强反光或强光
# - low_segmentation_coverage -> 提示重新伸舌并居中拍摄
# - small_tongue_region -> 提示靠近舌体重新拍摄
# - tongue_touches_edge -> 提示完整拍到舌体边缘
# - color_cast -> 提示更换光源或避开彩色环境光
QUALITY_REASON_FRONTEND_ACTIONS: dict[str, str] = {
    "low_focus": "\u63d0\u793a\u91cd\u65b0\u5bf9\u7126\u62cd\u6444",
    "overexposed": "\u63d0\u793a\u907f\u5f00\u5f3a\u53cd\u5149\u6216\u5f3a\u5149",
    "low_segmentation_coverage": "\u63d0\u793a\u91cd\u65b0\u4f38\u820c\u5e76\u5c45\u4e2d\u62cd\u6444",
    "small_tongue_region": "\u63d0\u793a\u9760\u8fd1\u820c\u4f53\u91cd\u65b0\u62cd\u6444",
    "tongue_touches_edge": "\u63d0\u793a\u5b8c\u6574\u62cd\u5230\u820c\u4f53\u8fb9\u7f18",
    "color_cast": "\u63d0\u793a\u66f4\u6362\u5149\u6e90\u6216\u907f\u5f00\u5f69\u8272\u73af\u5883\u5149",
}

# 标签解释：
# - 燥 -> 当前图像特征更接近高光少、裂纹相对明显的状态。
# - 中间态 -> 当前图像特征处于中间区间。
# - 润 -> 当前图像特征更接近高光较明显、裂纹较少的状态。
MOISTURE_LABEL_EXPLANATIONS: dict[str, str] = {
    "\u71e5": "\u5f53\u524d\u56fe\u50cf\u7279\u5f81\u66f4\u63a5\u8fd1\u9ad8\u5149\u5c11\u3001\u88c2\u7eb9\u76f8\u5bf9\u660e\u663e\u7684\u72b6\u6001\u3002",
    "\u4e2d\u95f4\u6001": "\u5f53\u524d\u56fe\u50cf\u7279\u5f81\u5904\u4e8e\u4e2d\u95f4\u533a\u95f4\u3002",
    "\u6da6": "\u5f53\u524d\u56fe\u50cf\u7279\u5f81\u66f4\u63a5\u8fd1\u9ad8\u5149\u8f83\u660e\u663e\u3001\u88c2\u7eb9\u8f83\u5c11\u7684\u72b6\u6001\u3002",
}


def build_score_breakdown(measurement: MoistureMeasurement) -> MoistureScoreBreakdown:
    """Build one API-facing score breakdown from a moisture measurement.

    Args:
        measurement: Region-level or image-level moisture measurement.

    Returns:
        The rule-based breakdown used by the current moisture score.
    """

    return build_score_breakdown_from_measurement(measurement)


def translate_quality_reasons(reasons: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Translate quality-control reason codes into Chinese copy.

    Args:
        reasons: Quality-reason codes from the rule engine.

    Returns:
        Human-readable Chinese explanations in the same order.
    """

    return tuple(QUALITY_REASON_TEXTS_ZH.get(reason, reason) for reason in reasons)


def build_moisture_explanation(measurement: MoistureMeasurement) -> str:
    """Build one stable explanation string for frontends and testing.

    Args:
        measurement: Region-level or image-level moisture measurement.

    Returns:
        A deterministic Chinese explanation aligned with the current rules.
    """

    prefix = _build_quality_prefix(measurement)
    # 当前图像 / 当前区域
    scope = "\u5f53\u524d\u56fe\u50cf" if isinstance(measurement, TongueMoistureMeasurement) else "\u5f53\u524d\u533a\u57df"
    gloss_text = _describe_gloss(measurement)
    crack_text = _describe_cracks(measurement)
    conclusion = _describe_label(measurement.moisture_label)
    return f"{prefix}{scope}{gloss_text}\uff0c\u4e14{crack_text}\uff0c{conclusion}"


def _build_quality_prefix(measurement: MoistureMeasurement) -> str:
    """Return the quality-control prefix for one explanation string."""

    if not isinstance(measurement, TongueMoistureMeasurement):
        return ""
    if measurement.quality_passed:
        return ""

    # 使用中文顿号连接多个质控原因。
    reasons_text = "\u3001".join(translate_quality_reasons(measurement.quality_reasons))
    return (
        # 当前图像未通过润燥分析质控（...），建议优先重拍；
        "\u5f53\u524d\u56fe\u50cf\u672a\u901a\u8fc7\u6da6\u71e5\u5206\u6790\u8d28\u63a7"
        f"\uff08{reasons_text}\uff09\uff0c\u5efa\u8bae\u4f18\u5148\u91cd\u62cd\uff1b"
    )


def _describe_gloss(measurement: MoistureMeasurement) -> str:
    """Describe gloss features in one sentence fragment."""

    if measurement.gloss_area_ratio < 0.01 and measurement.highlight_blob_count == 0:
        # 未检测到稳定高光
        return "\u672a\u68c0\u6d4b\u5230\u7a33\u5b9a\u9ad8\u5149"
    if measurement.highlight_blob_max_ratio >= 0.03:
        # 检测到较连续的高光区域
        return "\u68c0\u6d4b\u5230\u8f83\u8fde\u7eed\u7684\u9ad8\u5149\u533a\u57df"
    if measurement.highlight_blob_count >= 3:
        # 检测到高光但连通性一般
        return "\u68c0\u6d4b\u5230\u9ad8\u5149\u4f46\u8fde\u901a\u6027\u4e00\u822c"
    # 检测到少量高光
    return "\u68c0\u6d4b\u5230\u5c11\u91cf\u9ad8\u5149"


def _describe_cracks(measurement: MoistureMeasurement) -> str:
    """Describe crack features in one sentence fragment."""

    crack_ratio = max(measurement.crack_length_ratio, measurement.crack_area_ratio)
    if crack_ratio >= 0.03:
        # 裂纹特征较明显
        return "\u88c2\u7eb9\u7279\u5f81\u8f83\u660e\u663e"
    if crack_ratio >= 0.01:
        # 存在一定裂纹特征
        return "\u5b58\u5728\u4e00\u5b9a\u88c2\u7eb9\u7279\u5f81"
    # 裂纹特征较少
    return "\u88c2\u7eb9\u7279\u5f81\u8f83\u5c11"


def _describe_label(moisture_label: str) -> str:
    """Describe the final rule-based label in one sentence fragment."""

    if moisture_label == "\u6da6":
        # 因此整体更接近润。
        return "\u56e0\u6b64\u6574\u4f53\u66f4\u63a5\u8fd1\u6da6\u3002"
    if moisture_label == "\u4e2d\u95f4\u6001":
        # 因此整体处于中间态区间。
        return "\u56e0\u6b64\u6574\u4f53\u5904\u4e8e\u4e2d\u95f4\u6001\u533a\u95f4\u3002"
    # 因此整体更接近燥。
    return "\u56e0\u6b64\u6574\u4f53\u66f4\u63a5\u8fd1\u71e5\u3002"
