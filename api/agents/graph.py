"""LangGraph definition for the skin report workflow."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from api.agents.schemas import SkinReportState

MetricInput = dict[str, Any]
MetricInputExtractor = Callable[[dict[str, Any]], OrderedDict[str, MetricInput]]
ModelFactory = Callable[[], Any]
ScoreCalculator = Callable[[dict[str, dict[str, Any]]], int]
MetricReportGenerator = Callable[[Any, str, MetricInput], dict[str, Any]]
OverallSummaryGenerator = Callable[[Any, dict[str, dict[str, Any]], int], str]


def build_skin_report_graph(
    *,
    model_factory: ModelFactory,
    extract_metric_inputs: MetricInputExtractor,
    calculate_total_score: ScoreCalculator,
    generate_metric_report: MetricReportGenerator,
    generate_overall_summary: OverallSummaryGenerator,
) -> Any:
    """Build the LangGraph graph for the structured report workflow."""
    from langgraph.graph import END, START, StateGraph

    model = model_factory()

    def load_scores(state: SkinReportState) -> dict[str, Any]:
        return {"metric_inputs": extract_metric_inputs(state.get("analysis_results") or {})}

    def aggregate(state: SkinReportState) -> dict[str, Any]:
        metric_reports: OrderedDict[str, dict[str, Any]] = OrderedDict(
            [
                ("oil_moisture", dict(state.get("oil_moisture_report") or {})),
                ("sensitivity", dict(state.get("sensitivity_report") or {})),
                ("smoothness", dict(state.get("smoothness_report") or {})),
                ("wrinkles", dict(state.get("wrinkles_report") or {})),
                ("skin_tone", dict(state.get("skin_tone_report") or {})),
            ]
        )
        return {
            "metric_reports": dict(metric_reports),
            "total_score": calculate_total_score(metric_reports),
        }

    def overall_summary(state: SkinReportState) -> dict[str, Any]:
        metric_reports = dict(state.get("metric_reports") or {})
        total_score = int(state.get("total_score", 0))
        return {
            "overall_summary": generate_overall_summary(
                model, metric_reports, total_score
            )
        }

    def finalize(state: SkinReportState) -> dict[str, Any]:
        return {
            "llm_report": {
                "metric_reports": dict(state.get("metric_reports") or {}),
                "total_score": int(state.get("total_score", 0)),
                "overall_summary": str(state.get("overall_summary", "")),
                "generation_mode": "llm",
            }
        }

    graph = StateGraph(SkinReportState)
    graph.add_node("load_scores", load_scores)
    graph.add_node(
        "oil_moisture",
        lambda state: {
            "oil_moisture_report": generate_metric_report(
                model,
                "oil_moisture",
                dict((state.get("metric_inputs") or {}).get("oil_moisture", {})),
            )
        },
    )
    graph.add_node(
        "sensitivity",
        lambda state: {
            "sensitivity_report": generate_metric_report(
                model,
                "sensitivity",
                dict((state.get("metric_inputs") or {}).get("sensitivity", {})),
            )
        },
    )
    graph.add_node(
        "smoothness",
        lambda state: {
            "smoothness_report": generate_metric_report(
                model,
                "smoothness",
                dict((state.get("metric_inputs") or {}).get("smoothness", {})),
            )
        },
    )
    graph.add_node(
        "wrinkles",
        lambda state: {
            "wrinkles_report": generate_metric_report(
                model,
                "wrinkles",
                dict((state.get("metric_inputs") or {}).get("wrinkles", {})),
            )
        },
    )
    graph.add_node(
        "skin_tone",
        lambda state: {
            "skin_tone_report": generate_metric_report(
                model,
                "skin_tone",
                dict((state.get("metric_inputs") or {}).get("skin_tone", {})),
            )
        },
    )
    graph.add_node("aggregate", aggregate)
    graph.add_node("overall_summary", overall_summary)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "load_scores")
    for node_name in (
        "oil_moisture",
        "sensitivity",
        "smoothness",
        "wrinkles",
        "skin_tone",
    ):
        graph.add_edge("load_scores", node_name)
        graph.add_edge(node_name, "aggregate")
    graph.add_edge("aggregate", "overall_summary")
    graph.add_edge("overall_summary", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


__all__ = ["build_skin_report_graph"]
