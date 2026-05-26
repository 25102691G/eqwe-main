"""Schemas for the skin report agent."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class MetricAdvice(BaseModel):
    """Structured output for one metric recommendation."""

    title: str = Field(..., max_length=12)
    score: int = Field(..., ge=0, le=100)
    summary: str = Field(..., max_length=50)


class MetricSummaryPayload(BaseModel):
    """Validated JSON payload returned by the model for one metric."""

    summary: str = Field(..., max_length=50)


class OverallAdvice(BaseModel):
    """Structured output for the overall recommendation."""

    overall_summary: str = Field(..., max_length=150)


class SkinReportState(TypedDict, total=False):
    """LangGraph state used while generating the final skin report."""

    analysis_results: dict[str, Any]
    metric_inputs: dict[str, dict[str, Any]]
    oil_moisture_report: dict[str, Any]
    sensitivity_report: dict[str, Any]
    smoothness_report: dict[str, Any]
    wrinkles_report: dict[str, Any]
    skin_tone_report: dict[str, Any]
    metric_reports: dict[str, dict[str, Any]]
    total_score: int
    overall_summary: str
    llm_report: dict[str, Any]
