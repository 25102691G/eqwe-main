"""Prompt templates for the skin report agent."""

from __future__ import annotations

METRIC_SYSTEM_PROMPT = """你是一名护肤报告助手。

你的任务是根据皮肤指标名称、分数和补充上下文，生成一句简短中文建议，并严格返回 JSON。

输出要求：
1. 只输出一个 JSON 对象，不要输出 markdown 代码块，不要输出额外说明
2. JSON 格式必须是 {"summary":"..."}
3. `summary` 必须同时包含“注意什么”和“如何保养”
4. `summary` 控制在50字以内
5. 禁止输出医疗诊断、治疗结论、夸张承诺、恐吓表达
6. 语气专业、克制、适合微信小程序展示
"""

OVERALL_SYSTEM_PROMPT = """你是一名护肤报告助手。

你的任务是根据五个皮肤分项分数和分项建议，输出一段整体概述，并严格返回 JSON。

输出要求：
1. 只输出一个 JSON 对象，不要输出 markdown 代码块，不要输出额外说明
2. JSON 格式必须是 {"overall_summary":"..."}
3. `overall_summary` 控制在150字以内
4. 必须包含整体状态、主要注意点、总体保养方向
5. 禁止输出医疗诊断、治疗结论、夸张承诺、恐吓表达
6. 不要简单重复五条原句，要有总结感
"""


def build_metric_user_prompt(*, metric_name: str, score: int, context: str) -> str:
    """Render the user prompt for one metric report."""
    return (
        f"指标名称：{metric_name}\n"
        f"分数：{score}\n"
        f"补充上下文：{context or '无'}\n\n"
        '请只返回合法 JSON，例如：{"summary":"水油轻微波动，注意清洁节奏并加强日常补水。"}'
    )


def build_metric_retry_prompt(
    *,
    metric_name: str,
    score: int,
    context: str,
    previous_output: str,
    validation_error: str,
) -> str:
    """Render a repair prompt after invalid metric JSON output."""
    return (
        build_metric_user_prompt(metric_name=metric_name, score=score, context=context)
        + "\n\n你上一条回复不符合要求，请修正。"
        + f"\n上一条回复：{previous_output or '空'}"
        + f"\n校验错误：{validation_error}"
        + '\n现在只返回一个合法 JSON 对象，格式必须是 {"summary":"..."}'
    )


def build_overall_user_prompt(
    *,
    total_score: int,
    metric_reports: dict[str, dict[str, object]],
) -> str:
    """Render the user prompt for the overall summary."""
    lines = [f"总分：{total_score}", "分项结果："]
    for report in metric_reports.values():
        title = str(report.get("title", ""))
        score = int(report.get("score", 0))
        summary = str(report.get("summary", ""))
        lines.append(f"- {title}：{score}分；建议：{summary}")
    lines.append("")
    lines.append('请只返回合法 JSON，例如：{"overall_summary":"整体肤况中等偏稳，建议先做好保湿修护与日间防晒。"}')
    return "\n".join(lines)


def build_overall_retry_prompt(
    *,
    total_score: int,
    metric_reports: dict[str, dict[str, object]],
    previous_output: str,
    validation_error: str,
) -> str:
    """Render a repair prompt after invalid overall JSON output."""
    return (
        build_overall_user_prompt(total_score=total_score, metric_reports=metric_reports)
        + "\n\n你上一条回复不符合要求，请修正。"
        + f"\n上一条回复：{previous_output or '空'}"
        + f"\n校验错误：{validation_error}"
        + '\n现在只返回一个合法 JSON 对象，格式必须是 {"overall_summary":"..."}'
    )
