"""LLM-backed skin report generation."""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any

from api.agents.graph import build_skin_report_graph
from api.agents.prompts import (
    METRIC_SYSTEM_PROMPT,
    OVERALL_SYSTEM_PROMPT,
    build_metric_retry_prompt,
    build_metric_user_prompt,
    build_overall_retry_prompt,
    build_overall_user_prompt,
)
from api.agents.schemas import MetricSummaryPayload, OverallAdvice
from pydantic import BaseModel, ValidationError

MetricInput = dict[str, Any]

METRIC_CONFIG: tuple[tuple[str, str], ...] = (
    ("oil_moisture", "水油度"),
    ("sensitivity", "敏感度"),
    ("smoothness", "光滑度"),
    ("wrinkles", "皱纹"),
    ("skin_tone", "肤色"),
)

DEFAULT_SKIN_REPORT_MODEL = os.getenv("SKIN_REPORT_MODEL", "openai:gpt-5.4")


def _pick_number(*values: object) -> float | None:
    """Return the first finite numeric value."""
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _pick_text(*values: object) -> str:
    """Return the first non-empty string."""
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalize_score(*values: object, default: int = 60) -> int:
    """Clamp a score into the 0-100 range."""
    picked = _pick_number(*values)
    if picked is None:
        return default
    return max(0, min(100, int(round(picked))))


def _limit_text(text: str, max_length: int) -> str:
    """Trim text to the requested length and normalize whitespace."""
    normalized = " ".join(text.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[:max_length].rstrip("，,。.;；:：!！?？ ")


def _extract_metric_inputs(analysis_results: dict[str, Any]) -> OrderedDict[str, MetricInput]:
    """Extract the five report dimensions from the analysis payload."""
    oil_section = analysis_results.get("oil_moi", {})
    skin_color_section = analysis_results.get("skin_color", {})
    sensitivity_section = analysis_results.get("sensitivity", {})
    smoothness_section = analysis_results.get("smoothness", {})
    wrinkles_section = analysis_results.get("wrinkles", {})

    metric_inputs: OrderedDict[str, MetricInput] = OrderedDict()
    metric_inputs["oil_moisture"] = {
        "title": "水油度",
        "score": _normalize_score(
            oil_section.get("score"),
            (oil_section.get("oil_analysis") or {}).get("oil_score"),
            (oil_section.get("moisture_analysis") or {}).get("moisture_score"),
        ),
        "context": _pick_text(
            oil_section.get("description"),
            (oil_section.get("oil_analysis") or {}).get("description"),
            (oil_section.get("moisture_analysis") or {}).get("description"),
        ),
    }
    metric_inputs["sensitivity"] = {
        "title": "敏感度",
        "score": _normalize_score(
            sensitivity_section.get("score"),
            (sensitivity_section.get("sensitivity_analysis") or {}).get("sensitivity_score"),
        ),
        "context": _pick_text(
            sensitivity_section.get("description"),
            (sensitivity_section.get("sensitivity_analysis") or {}).get("description"),
        ),
    }
    metric_inputs["smoothness"] = {
        "title": "光滑度",
        "score": _normalize_score(smoothness_section.get("score")),
        "context": _pick_text(
            smoothness_section.get("description"),
            ((smoothness_section.get("smooth") or {}).get("doudou") or {}).get("suggestion"),
        ),
    }
    metric_inputs["wrinkles"] = {
        "title": "皱纹",
        "score": _normalize_score(
            ((wrinkles_section.get("black_eye") or {}).get("position_score") or {}).get("score"),
            wrinkles_section.get("score"),
        ),
        "context": _pick_text(
            (wrinkles_section.get("wrinkles") or {}).get("suggest"),
            ((wrinkles_section.get("black_eye") or {}).get("suggest") or {}).get("talk_suggest"),
        ),
    }
    metric_inputs["skin_tone"] = {
        "title": "肤色",
        "score": _normalize_score(
            skin_color_section.get("score"),
            (skin_color_section.get("hyperpigmentation") or {}).get("se_ban"),
        ),
        "context": _pick_text(
            skin_color_section.get("description"),
            (skin_color_section.get("hyperpigmentation") or {}).get("suggestion"),
            (skin_color_section.get("skin_tone_classification") or {}).get("description"),
        ),
    }
    return metric_inputs


def calculate_total_score(metric_reports: dict[str, dict[str, Any]]) -> int:
    """Average the five metric scores into one total score."""
    scores = [
        int(report.get("score", 0))
        for report in metric_reports.values()
        if isinstance(report, dict)
    ]
    if not scores:
        return 0
    return int(round(sum(scores) / len(scores)))


def _metric_band(score: int) -> str:
    """Return a coarse score band label."""
    if score >= 85:
        return "strong"
    if score >= 70:
        return "steady"
    if score >= 55:
        return "mixed"
    return "weak"


def _fallback_metric_summary(metric_key: str, score: int) -> str:
    """Return a deterministic short summary when the LLM is unavailable."""
    band = _metric_band(score)
    templates: dict[str, dict[str, str]] = {
        "oil_moisture": {
            "strong": "水油状态稳定，继续温和清洁并坚持补水锁水。",
            "steady": "水油轻微波动，注意清洁节奏并加强日常补水。",
            "mixed": "水油平衡一般，少熬夜少过度清洁，搭配保湿乳。",
            "weak": "水油失衡较明显，减少刺激清洁，重点补水修护。",
        },
        "sensitivity": {
            "strong": "肌肤耐受较稳，继续精简护肤并坚持保湿防晒。",
            "steady": "敏感风险轻度存在，避免频繁刷酸，优先修护屏障。",
            "mixed": "屏障状态一般，减少刺激成分，使用舒缓保湿产品。",
            "weak": "敏感倾向较明显，暂停刺激护理，先做修护保湿与防晒。",
        },
        "smoothness": {
            "strong": "肌理较平整，继续规律清洁保湿，维持角质代谢。",
            "steady": "局部粗糙轻度存在，可温和清洁并加强补水。",
            "mixed": "粗糙和堵塞开始明显，可规律清洁并温和疏通角质。",
            "weak": "肌理不够平滑，注意控油清洁，逐步改善毛孔堵塞。",
        },
        "wrinkles": {
            "strong": "细纹控制较稳，坚持防晒并保持充足保湿即可。",
            "steady": "轻度细纹可见，注意防晒并加入抗氧化护理。",
            "mixed": "干纹细纹开始明显，做好保湿并逐步加入抗老护理。",
            "weak": "细纹问题较突出，强化防晒保湿，并坚持抗老护理。",
        },
        "skin_tone": {
            "strong": "肤色整体均匀，继续防晒并避免反复刺激。",
            "steady": "肤色轻度不均，注意防晒并搭配提亮修护。",
            "mixed": "色沉迹象开始明显，减少刺激，做好防晒与修护。",
            "weak": "肤色不均较明显，先稳住屏障，再配合提亮防晒。",
        },
    }
    return templates[metric_key][band]


def _fallback_overall_summary(metric_reports: dict[str, dict[str, Any]], total_score: int) -> str:
    """Return a deterministic overall summary when the LLM is unavailable."""
    reports = list(metric_reports.values())
    if not reports:
        return "整体肤况信息不足，建议先完成一次完整拍照分析后再查看报告。"

    lowest_reports = sorted(reports, key=lambda item: int(item.get("score", 0)))[:2]
    focus_titles = "、".join(str(item.get("title", "")) for item in lowest_reports if item.get("title"))

    if total_score >= 85:
        intro = "整体肤况较稳定"
    elif total_score >= 70:
        intro = "整体肤况中等偏稳"
    elif total_score >= 55:
        intro = "整体肤况有一定波动"
    else:
        intro = "整体肤况需要优先修护"

    summary = (
        f"{intro}，当前应优先关注{focus_titles}。"
        "建议先做好温和清洁、保湿修护与日间防晒，再逐步做针对性改善。"
    )
    return _limit_text(summary, 150)


def _build_fallback_report(metric_inputs: OrderedDict[str, MetricInput], reason: str) -> dict[str, Any]:
    """Build a safe fallback report without calling the LLM."""
    metric_reports: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for key, metric_input in metric_inputs.items():
        score = int(metric_input["score"])
        metric_reports[key] = {
            "title": str(metric_input["title"]),
            "score": score,
            "summary": _limit_text(_fallback_metric_summary(key, score), 50),
        }

    total_score = calculate_total_score(metric_reports)
    return {
        "metric_reports": dict(metric_reports),
        "total_score": total_score,
        "overall_summary": _fallback_overall_summary(metric_reports, total_score),
        "generation_mode": "fallback",
        "generation_reason": reason,
    }


def _build_model():
    """Create the LangChain chat model used for report generation."""
    from langchain.chat_models import init_chat_model

    kwargs: dict[str, Any] = {
        "temperature": 0.2,
    }
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return init_chat_model(DEFAULT_SKIN_REPORT_MODEL, **kwargs)


def _normalize_metric_report(
    metric_key: str,
    metric_input: MetricInput,
    summary_text: str,
) -> dict[str, Any]:
    """Force the model output into the expected frontend contract."""
    fallback_summary = _fallback_metric_summary(metric_key, int(metric_input["score"]))

    summary = _limit_text(summary_text or fallback_summary, 50)
    return {
        "title": str(metric_input["title"]),
        "score": int(metric_input["score"]),
        "summary": summary or fallback_summary,
    }


def _extract_response_text(response: Any) -> str:
    """Extract the best-effort text payload from a chat model response."""
    if isinstance(response, str):
        return response.strip()

    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str) and item.strip():
                chunks.append(item.strip())
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
                    continue
                nested_text = item.get("content")
                if isinstance(nested_text, str) and nested_text.strip():
                    chunks.append(nested_text.strip())
                    continue
            text_attr = getattr(item, "text", None)
            if isinstance(text_attr, str) and text_attr.strip():
                chunks.append(text_attr.strip())
        if chunks:
            return "\n".join(chunks).strip()

    response_text = getattr(response, "text", None)
    if isinstance(response_text, str) and response_text.strip():
        return response_text.strip()
    if callable(response_text):
        text_value = response_text()
        if isinstance(text_value, str) and text_value.strip():
            return text_value.strip()

    return str(response).strip()


def _normalize_json_candidate(raw_text: str) -> list[str]:
    """Return candidate JSON snippets extracted from a model response."""
    stripped = raw_text.strip()
    candidates: list[str] = []

    if stripped:
        candidates.append(stripped)

    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        fenced = "\n".join(lines).strip()
        if fenced:
            candidates.append(fenced)

    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(stripped[first_brace : last_brace + 1].strip())

    unique_candidates: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def _validate_json_text(raw_text: str, schema: type[BaseModel]) -> BaseModel:
    """Validate a raw model response against a JSON schema."""
    last_error = "empty_response"
    for candidate in _normalize_json_candidate(raw_text):
        try:
            return schema.model_validate_json(candidate)
        except ValidationError as exc:
            last_error = str(exc)
    msg = last_error
    raise ValueError(msg)


def _invoke_json_response(
    *,
    model: Any,
    system_prompt: str,
    initial_user_prompt: str,
    retry_prompt_builder: Any,
    schema: type[BaseModel],
    max_attempts: int = 3,
) -> BaseModel:
    """Invoke the model until valid JSON is produced or retries are exhausted."""
    previous_output = ""
    validation_error = ""

    for attempt in range(max_attempts):
        user_prompt = initial_user_prompt
        if attempt > 0:
            user_prompt = retry_prompt_builder(previous_output, validation_error)

        response = model.invoke(
            [
                ("system", system_prompt),
                ("user", user_prompt),
            ]
        )
        raw_text = _extract_response_text(response)
        try:
            return _validate_json_text(raw_text, schema)
        except ValueError as exc:
            previous_output = raw_text
            validation_error = str(exc)

    msg = validation_error or "invalid_json_output"
    raise ValueError(msg)


def _generate_metric_report(model: Any, metric_key: str, metric_input: MetricInput) -> dict[str, Any]:
    """Generate one metric report with JSON validation and repair retries."""
    metric_name = str(metric_input.get("title", ""))
    score = int(metric_input.get("score", 60))
    context = str(metric_input.get("context", ""))
    payload = _invoke_json_response(
        model=model,
        system_prompt=METRIC_SYSTEM_PROMPT,
        initial_user_prompt=build_metric_user_prompt(
            metric_name=metric_name,
            score=score,
            context=context,
        ),
        retry_prompt_builder=lambda previous_output, validation_error: build_metric_retry_prompt(
            metric_name=metric_name,
            score=score,
            context=context,
            previous_output=previous_output,
            validation_error=validation_error,
        ),
        schema=MetricSummaryPayload,
    )
    return _normalize_metric_report(metric_key, metric_input, payload.summary)


def _generate_overall_summary(
    model: Any,
    metric_reports: dict[str, dict[str, Any]],
    total_score: int,
) -> str:
    """Generate the final overall summary with JSON validation and repair retries."""
    payload = _invoke_json_response(
        model=model,
        system_prompt=OVERALL_SYSTEM_PROMPT,
        initial_user_prompt=build_overall_user_prompt(
            total_score=total_score,
            metric_reports=metric_reports,
        ),
        retry_prompt_builder=lambda previous_output, validation_error: build_overall_retry_prompt(
            total_score=total_score,
            metric_reports=metric_reports,
            previous_output=previous_output,
            validation_error=validation_error,
        ),
        schema=OverallAdvice,
    )
    return _limit_text(
        payload.overall_summary or _fallback_overall_summary(metric_reports, total_score),
        150,
    )


def generate_skin_report(analysis_results: dict[str, Any]) -> dict[str, Any]:
    """Generate an LLM-backed report, with a deterministic fallback."""
    metric_inputs = _extract_metric_inputs(analysis_results)
    if not metric_inputs:
        return _build_fallback_report(OrderedDict(), "missing_scores")

    if not os.getenv("OPENAI_API_KEY"):
        return _build_fallback_report(metric_inputs, "missing_openai_api_key")

    try:
        graph = build_skin_report_graph(
            model_factory=_build_model,
            extract_metric_inputs=_extract_metric_inputs,
            calculate_total_score=calculate_total_score,
            generate_metric_report=_generate_metric_report,
            generate_overall_summary=_generate_overall_summary,
        )
        result = graph.invoke({"analysis_results": analysis_results})
        llm_report = result.get("llm_report")
        if isinstance(llm_report, dict):
            return llm_report
    except ImportError as exc:
        return _build_fallback_report(metric_inputs, f"missing_dependency:{exc}")
    except Exception as exc:  # noqa: BLE001
        return _build_fallback_report(metric_inputs, f"llm_generation_failed:{exc}")

    return _build_fallback_report(metric_inputs, "empty_llm_result")


__all__ = ["calculate_total_score", "generate_skin_report"]
