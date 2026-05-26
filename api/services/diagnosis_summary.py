"""Build chat-safe assistance summaries from face or tongue analysis payloads."""

from __future__ import annotations

from typing import Any


def _pick_score(*values: object) -> float | None:
    """Return the first finite numeric score value."""
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return round(float(value), 1)
    return None


def _pick_number(*values: object) -> int | None:
    """Return the first finite numeric value as an integer."""
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(round(value))
    return None


def _pick_text(*values: object) -> str:
    """Return the first non-empty string value."""
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _is_tongue_analysis(analysis_payload: dict[str, Any]) -> bool:
    """Return whether the payload looks like a tongue phase-1 result."""
    return bool(
        analysis_payload.get("tongue_color")
        or analysis_payload.get("tongue_moisture")
        or analysis_payload.get("tongue_image_assistance")
        or analysis_payload.get("demo_report")
    )


def _build_fallback_summary(
    metric_reports: dict[str, dict[str, Any]],
    total_score: int | None,
) -> str:
    """Build a deterministic summary when the upstream report is incomplete."""
    reports = [
        report
        for report in metric_reports.values()
        if isinstance(report, dict) and report.get("title")
    ]
    if not reports:
        return "本次肤况辅助分析已生成，但关键指标不足，建议重新上传清晰正脸照片后再分析。"

    lowest_reports = sorted(
        reports,
        key=lambda report: int(_pick_number(report.get("score")) or 0),
    )[:2]
    focus_titles = "、".join(str(report.get("title", "")) for report in lowest_reports)

    if total_score is None:
        prefix = "本次肤况辅助分析已生成"
    elif total_score >= 85:
        prefix = "本次肤况辅助分析提示整体状态较稳定"
    elif total_score >= 70:
        prefix = "本次肤况辅助分析提示整体状态中等偏稳"
    elif total_score >= 55:
        prefix = "本次肤况辅助分析提示整体状态存在一定波动"
    else:
        prefix = "本次肤况辅助分析提示皮肤状态需要优先修护"

    return (
        f"{prefix}，当前更需要关注{focus_titles}。"
        "后续交流请优先围绕温和清洁、保湿修护、防晒和重点问题护理展开。"
    )


def _extract_tongue_summary_cards(analysis_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Build tongue phase-1 assistance cards for chat context and report cards."""
    assistance = dict(analysis_payload.get("tongue_image_assistance") or {})
    assistance_summary = dict(assistance.get("summary") or {})
    moisture = dict(analysis_payload.get("tongue_moisture") or {})
    coat = dict(analysis_payload.get("tongue_coat") or {})
    color = dict(analysis_payload.get("tongue_color") or {})
    crack = dict(analysis_payload.get("crack_observation") or {})
    quality = dict(assistance.get("quality") or {})

    return [
        {
            "key": "quality",
            "reportKey": "quality",
            "title": "图像质量",
            "score": None,
            "scoreText": _pick_text(
                quality.get("level"),
                "可分析" if quality.get("passed") else "需重拍",
            ),
            "description": _pick_text(
                quality.get("suggestion"),
                "建议使用光线均匀、舌体完整、画面清晰的照片。",
            ),
            "accent": "mist" if quality.get("passed") is False else "sky",
        },
        {
            "key": "tongue-color",
            "reportKey": "tongue_color",
            "title": "舌色倾向",
            "score": None,
            "scoreText": _pick_text(
                color.get("color_name"),
                assistance_summary.get("color_tendency"),
                "--",
            ),
            "description": (
                f"代表色 {color.get('representative_hex')}，仅作图像颜色辅助参考。"
                if color.get("representative_hex")
                else "未获得稳定舌色特征。"
            ),
            "accent": "harbor",
        },
        {
            "key": "tongue-moisture",
            "reportKey": "tongue_moisture",
            "title": "润燥倾向",
            "score": _pick_score(moisture.get("moisture_score")),
            "scoreText": _pick_text(
                moisture.get("moisture_label"),
                _score_text(_pick_score(moisture.get("moisture_score"))),
            ),
            "description": _pick_text(
                moisture.get("moisture_explanation"),
                assistance_summary.get("moisture_tendency"),
                "暂无润燥说明。",
            ),
            "accent": "ice",
        },
        {
            "key": "tongue-coat",
            "reportKey": "tongue_coat",
            "title": "舌苔候选",
            "score": _pick_score(coat.get("coat_coverage_ratio")),
            "scoreText": _pick_text(
                coat.get("coat_visibility"),
                assistance_summary.get("coat_visibility"),
                "--",
            ),
            "description": _pick_text(
                (
                    f"{coat.get('coat_color_tendency')}，"
                    f"{coat.get('coat_thickness_tendency') or '薄厚未定'}"
                )
                if coat.get("coat_color_tendency")
                else "",
                "当前未见稳定舌苔候选区域。",
            ),
            "accent": "mist",
        },
        {
            "key": "crack",
            "reportKey": "crack",
            "title": "裂纹候选",
            "score": _pick_score(crack.get("crack_area_ratio")),
            "scoreText": _pick_text(crack.get("crack_level"), "--"),
            "description": (
                "裂纹候选检测置信度较高，仍需结合人工观察。"
                if crack.get("confidence") == "high"
                else "裂纹候选结果仅作参考。"
            ),
            "accent": "ocean",
        },
    ]


def _extract_tongue_overall_report(
    analysis_payload: dict[str, Any],
) -> dict[str, Any]:
    """Build the tongue phase-1 overall report summary block."""
    report = dict(analysis_payload.get("demo_report") or {})
    assistance = dict(analysis_payload.get("tongue_image_assistance") or {})
    quality = dict(report.get("quality_gate") or assistance.get("quality") or {})
    tendencies = report.get("primary_tendencies")
    tendencies_text = "、".join(str(item) for item in tendencies) if isinstance(tendencies, list) else ""
    observations = dict(assistance.get("summary") or {}).get("main_observations")
    observations_text = "；".join(str(item) for item in observations) if isinstance(observations, list) else ""

    return {
        "totalScore": None,
        "totalScoreText": _pick_text(
            quality.get("level"),
            "可分析" if quality.get("passed") else "需重拍",
        ),
        "summary": _pick_text(
            report.get("analysis_summary"),
            observations_text,
            f"体质倾向参考：{tendencies_text}" if tendencies_text else "",
            "舌象一期辅助分析已完成。",
        ),
        "generationMode": _pick_text(
            report.get("report_type"),
            "tongue_phase1_rule_assistance",
        ),
    }


def _extract_tongue_gallery(analysis_payload: dict[str, Any]) -> list[dict[str, str]]:
    """Build the tongue report gallery from MinIO uploaded files."""
    existing_images = analysis_payload.get("result_images")
    if isinstance(existing_images, list) and existing_images:
        return [
            item
            for item in existing_images
            if isinstance(item, dict) and _pick_text(item.get("url"), item.get("filename"))
        ]

    storage = dict(analysis_payload.get("storage") or {})
    folder = _pick_text(storage.get("folder"))
    uploaded_files = analysis_payload.get("uploaded_files")
    if not folder or not isinstance(uploaded_files, list):
        return []

    image_names = {
        "tongue_segmented.jpg": "舌体分割图",
        "tongue_mask.jpg": "舌体掩膜",
        "tongue_gloss_overlay.jpg": "润泽高光叠加",
        "tongue_crack_overlay.jpg": "裂纹候选叠加",
        "tongue_overexposed_overlay.jpg": "过曝区域提示",
        "tongue_moisture_heatmap.jpg": "润燥热力图",
        "tongue_moisture_score_breakdown.jpg": "润燥评分拆解",
    }
    gallery: list[dict[str, str]] = []
    for item in uploaded_files:
        if not isinstance(item, dict):
            continue
        filename = _pick_text(item.get("filename"))
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        gallery.append(
            {
                "filename": filename,
                "label": image_names.get(filename, filename.rsplit(".", 1)[0].replace("_", " ")),
                "url": f"/v1/mobile/result-image/{folder}/{filename}",
                "object_key": _pick_text(item.get("object_key")),
            }
        )
    return gallery


def _score_text(score: float | None) -> str:
    """Return a compact display string for one score value."""
    if score is None:
        return "--"
    if score.is_integer():
        return str(int(score))
    return str(score)


def _extract_summary_cards(analysis_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the five primary skin assistance summary cards."""
    analysis = dict(analysis_payload.get("analysis_results") or {})
    llm_report = dict(analysis_payload.get("llm_report") or {})
    metric_reports = {
        key: dict(value)
        for key, value in dict(llm_report.get("metric_reports") or {}).items()
        if isinstance(value, dict)
    }

    oil_section = dict(analysis.get("oil_moi") or {})
    skin_color_section = dict(analysis.get("skin_color") or {})
    sensitivity_section = dict(analysis.get("sensitivity") or {})
    smoothness_section = dict(analysis.get("smoothness") or {})
    wrinkles_section = dict(analysis.get("wrinkles") or {})

    cards = [
        {
            "key": "oil-moisture",
            "reportKey": "oil_moisture",
            "title": "水油度",
            "score": _pick_score(
                dict(metric_reports.get("oil_moisture") or {}).get("score"),
                oil_section.get("score"),
                dict(oil_section.get("oil_analysis") or {}).get("oil_score"),
            ),
            "description": _pick_text(
                dict(metric_reports.get("oil_moisture") or {}).get("summary"),
                oil_section.get("description"),
                dict(oil_section.get("oil_analysis") or {}).get("description"),
                dict(oil_section.get("moisture_analysis") or {}).get("description"),
            ),
            "accent": "sky",
        },
        {
            "key": "sensitivity",
            "reportKey": "sensitivity",
            "title": "敏感度",
            "score": _pick_score(
                dict(metric_reports.get("sensitivity") or {}).get("score"),
                sensitivity_section.get("score"),
                dict(sensitivity_section.get("sensitivity_analysis") or {}).get(
                    "sensitivity_score"
                ),
            ),
            "description": _pick_text(
                dict(metric_reports.get("sensitivity") or {}).get("summary"),
                sensitivity_section.get("description"),
                dict(sensitivity_section.get("sensitivity_analysis") or {}).get(
                    "description"
                ),
            ),
            "accent": "mist",
        },
        {
            "key": "smoothness",
            "reportKey": "smoothness",
            "title": "平滑度",
            "score": _pick_score(
                dict(metric_reports.get("smoothness") or {}).get("score"),
                smoothness_section.get("score"),
            ),
            "description": _pick_text(
                dict(metric_reports.get("smoothness") or {}).get("summary"),
                smoothness_section.get("description"),
                dict(dict(smoothness_section.get("smooth") or {}).get("doudou") or {}).get(
                    "suggestion"
                ),
            ),
            "accent": "ice",
        },
        {
            "key": "wrinkles",
            "reportKey": "wrinkles",
            "title": "细纹",
            "score": _pick_score(
                dict(metric_reports.get("wrinkles") or {}).get("score"),
                dict(dict(wrinkles_section.get("black_eye") or {}).get("position_score") or {}).get(
                    "score"
                ),
            ),
            "description": _pick_text(
                dict(metric_reports.get("wrinkles") or {}).get("summary"),
                dict(wrinkles_section.get("wrinkles") or {}).get("suggest"),
                dict(dict(wrinkles_section.get("black_eye") or {}).get("suggest") or {}).get(
                    "talk_suggest"
                ),
            ),
            "accent": "ocean",
        },
        {
            "key": "skin-tone",
            "reportKey": "skin_tone",
            "title": "肤色",
            "score": _pick_score(
                dict(metric_reports.get("skin_tone") or {}).get("score"),
                skin_color_section.get("score"),
                dict(skin_color_section.get("hyperpigmentation") or {}).get("se_ban"),
            ),
            "description": _pick_text(
                dict(metric_reports.get("skin_tone") or {}).get("summary"),
                skin_color_section.get("description"),
                dict(skin_color_section.get("hyperpigmentation") or {}).get("suggestion"),
                dict(skin_color_section.get("skin_tone_classification") or {}).get(
                    "description"
                ),
            ),
            "accent": "harbor",
        },
    ]

    return [
        {
            **card,
            "description": card["description"] or "暂无护理建议",
            "scoreText": _score_text(card["score"]),
        }
        for card in cards
    ]


def _extract_overall_report(
    analysis_payload: dict[str, Any],
    summary_cards: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the overall report summary block."""
    llm_report = dict(analysis_payload.get("llm_report") or {})
    scores = [
        card["score"]
        for card in summary_cards
        if isinstance(card.get("score"), (int, float))
    ]
    average_score = round(sum(scores) / len(scores)) if scores else None
    total_score = _pick_score(llm_report.get("total_score"), average_score)
    return {
        "totalScore": total_score,
        "totalScoreText": _score_text(total_score),
        "summary": _pick_text(
            llm_report.get("overall_summary"),
            "整体结果已生成，建议结合五项评分安排日常护理。",
        ),
        "generationMode": _pick_text(llm_report.get("generation_mode")),
    }


def _extract_gallery(analysis_payload: dict[str, Any]) -> list[dict[str, str]]:
    """Build the report gallery from result images."""
    images: list[dict[str, str]] = []
    for item in list(analysis_payload.get("result_images") or []):
        if not isinstance(item, dict):
            continue
        filename = _pick_text(item.get("filename"))
        label = _pick_text(item.get("label"), filename)
        url = _pick_text(item.get("url"))
        if not filename or not url:
            continue
        images.append(
            {
                "filename": filename,
                "label": label,
                "url": url,
            }
        )
    return images


def build_diagnosis_context(analysis_payload: dict[str, Any]) -> dict[str, Any]:
    """Build a compact assistance context suitable for chat prompts and UI cards.

    Args:
        analysis_payload: The result payload returned by face or tongue analysis.

    Returns:
        A compact assistance context object for session storage.
    """
    if _is_tongue_analysis(analysis_payload):
        storage = dict(analysis_payload.get("storage") or {})
        metadata = dict(analysis_payload.get("metadata") or {})
        summary_cards = _extract_tongue_summary_cards(analysis_payload)
        overall_report = _extract_tongue_overall_report(analysis_payload)
        highlights = [
            {
                "key": _pick_text(card.get("reportKey"), card.get("key")),
                "title": _pick_text(card.get("title"), card.get("key")),
                "score": _pick_number(card.get("score")),
                "summary": _pick_text(card.get("description"), card.get("scoreText")),
            }
            for card in summary_cards
            if _pick_text(card.get("title"), card.get("key"))
        ]
        return {
            "sourceType": "tongue",
            "sourceLabel": "舌象一期",
            "sourceFolder": _pick_text(storage.get("folder"), metadata.get("folder")),
            "totalScore": overall_report["totalScore"],
            "totalScoreText": overall_report["totalScoreText"],
            "summary": overall_report["summary"],
            "metricHighlights": highlights,
            "reportUrl": _pick_text(analysis_payload.get("analysis_report_url")),
            "updatedAt": _pick_text(metadata.get("analysis_timestamp")),
        }

    llm_report = dict(analysis_payload.get("llm_report") or {})
    metric_reports = {
        key: dict(value)
        for key, value in dict(llm_report.get("metric_reports") or {}).items()
        if isinstance(value, dict)
    }
    total_score = _pick_number(llm_report.get("total_score"))
    overall_summary = _pick_text(llm_report.get("overall_summary"))
    summary = overall_summary or _build_fallback_summary(metric_reports, total_score)

    storage = dict(analysis_payload.get("storage") or {})
    metadata = dict(analysis_payload.get("metadata") or {})
    summary_cards = _extract_summary_cards(analysis_payload)
    highlights = [
        {
            "key": _pick_text(card.get("reportKey"), card.get("key")),
            "title": _pick_text(card.get("title"), card.get("key")),
            "score": _pick_number(card.get("score")),
            "summary": _pick_text(card.get("description")),
        }
        for card in summary_cards
        if _pick_text(card.get("title"), card.get("key")) and card.get("score") is not None
    ]

    return {
        "sourceType": "face",
        "sourceLabel": "面象肤况",
        "sourceFolder": _pick_text(storage.get("folder"), metadata.get("folder")),
        "totalScore": total_score,
        "totalScoreText": _score_text(float(total_score)) if total_score is not None else "--",
        "summary": summary,
        "metricHighlights": highlights,
        "reportUrl": _pick_text(analysis_payload.get("analysis_report_url")),
        "updatedAt": _pick_text(metadata.get("analysis_timestamp")),
    }


def build_diagnosis_report(analysis_payload: dict[str, Any]) -> dict[str, Any]:
    """Build a richer assistance report payload for chat expansion cards.

    Args:
        analysis_payload: The result payload returned by face or tongue analysis.

    Returns:
        A report payload suitable for chat message metadata.
    """
    if _is_tongue_analysis(analysis_payload):
        context = build_diagnosis_context(analysis_payload)
        summary_cards = _extract_tongue_summary_cards(analysis_payload)
        overall_report = _extract_tongue_overall_report(analysis_payload)
        return {
            "sourceType": "tongue",
            "sourceLabel": "舌象一期",
            "sourceFolder": context["sourceFolder"],
            "totalScore": overall_report["totalScore"],
            "totalScoreText": overall_report["totalScoreText"],
            "summary": overall_report["summary"],
            "generationMode": overall_report["generationMode"],
            "summaryCards": summary_cards,
            "gallery": _extract_tongue_gallery(analysis_payload),
            "reportUrl": context["reportUrl"],
            "updatedAt": context["updatedAt"],
        }

    context = build_diagnosis_context(analysis_payload)
    summary_cards = _extract_summary_cards(analysis_payload)
    overall_report = _extract_overall_report(analysis_payload, summary_cards)

    return {
        "sourceType": "face",
        "sourceLabel": "面象肤况",
        "sourceFolder": context["sourceFolder"],
        "totalScore": overall_report["totalScore"],
        "totalScoreText": overall_report["totalScoreText"],
        "summary": overall_report["summary"],
        "generationMode": overall_report["generationMode"],
        "summaryCards": summary_cards,
        "gallery": _extract_gallery(analysis_payload),
        "reportUrl": context["reportUrl"],
        "updatedAt": context["updatedAt"],
    }


def build_combined_diagnosis_context(
    analysis_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build one chat context from the latest tongue and face assistance reports."""
    contexts = [
        build_diagnosis_context(payload)
        for payload in analysis_payloads
        if isinstance(payload, dict)
    ]
    contexts = [context for context in contexts if context.get("summary")]
    if not contexts:
        return {}
    if len(contexts) == 1:
        return contexts[0]

    labels = [_pick_text(item.get("sourceLabel"), "辅助分析") for item in contexts]
    summaries = [
        f"{_pick_text(item.get('sourceLabel'), '辅助分析')}: {item.get('summary')}"
        for item in contexts
        if item.get("summary")
    ]
    highlights: list[dict[str, Any]] = []
    for item in contexts:
        source_label = _pick_text(item.get("sourceLabel"), "辅助分析")
        for highlight in list(item.get("metricHighlights") or [])[:3]:
            if not isinstance(highlight, dict):
                continue
            highlights.append(
                {
                    **highlight,
                    "sourceLabel": source_label,
                    "title": f"{source_label}-{_pick_text(highlight.get('title'), highlight.get('key'))}",
                }
            )

    latest_context = max(
        contexts,
        key=lambda item: _pick_text(item.get("updatedAt"), item.get("sourceFolder")),
    )
    return {
        "sourceType": "combined",
        "sourceLabel": "综合辅助分析",
        "sourceFolder": "|".join(
            item.get("sourceFolder") or item.get("sourceType") or "" for item in contexts
        ),
        "totalScore": None,
        "totalScoreText": "综合",
        "summary": "；".join(summaries),
        "metricHighlights": highlights,
        "diagnosisContexts": contexts,
        "reportUrl": _pick_text(latest_context.get("reportUrl")),
        "updatedAt": _pick_text(latest_context.get("updatedAt")),
        "contextCount": len(contexts),
        "sourceLabels": labels,
    }


def build_combined_diagnosis_report(
    analysis_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build one report-card payload from multiple assistance reports."""
    reports = [
        build_diagnosis_report(payload)
        for payload in analysis_payloads
        if isinstance(payload, dict)
    ]
    reports = [report for report in reports if report.get("summary")]
    if not reports:
        return {}
    if len(reports) == 1:
        return reports[0]

    context = build_combined_diagnosis_context(analysis_payloads)
    summary_cards: list[dict[str, Any]] = []
    gallery: list[dict[str, Any]] = []
    for report in reports:
        source_label = _pick_text(report.get("sourceLabel"), "辅助分析")
        for card in list(report.get("summaryCards") or []):
            if not isinstance(card, dict):
                continue
            summary_cards.append(
                {
                    **card,
                    "sourceLabel": source_label,
                    "title": f"{source_label}-{_pick_text(card.get('title'), card.get('key'))}",
                }
            )
        gallery.extend([item for item in list(report.get("gallery") or []) if isinstance(item, dict)])

    return {
        "sourceType": "combined",
        "sourceLabel": "综合辅助分析",
        "sourceFolder": context.get("sourceFolder", ""),
        "totalScore": None,
        "totalScoreText": "综合",
        "summary": context.get("summary", ""),
        "generationMode": "combined_assistance",
        "summaryCards": summary_cards,
        "gallery": gallery,
        "diagnosisReports": reports,
        "reportUrl": context.get("reportUrl", ""),
        "updatedAt": context.get("updatedAt", ""),
    }


__all__ = [
    "build_combined_diagnosis_context",
    "build_combined_diagnosis_report",
    "build_diagnosis_context",
    "build_diagnosis_report",
]
