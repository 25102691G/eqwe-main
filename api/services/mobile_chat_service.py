"""Utilities for mobile chat session replies and attachment handling."""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

from openai import OpenAI

from api.configs.storage_config import STORAGE_CONFIG
from api.middleware.storage.cloud_storage_service import CloudStorageService

logger = logging.getLogger(__name__)

MOBILE_CHAT_MODEL = (
    os.getenv("MOBILE_CHAT_MODEL")
    or os.getenv("OPENAI_MODEL")
    or os.getenv("SKIN_REPORT_MODEL")
    or "openai:gpt-5.4"
)
MAX_HISTORY_MESSAGES = 16
MAX_ATTACHMENT_TEXT_LENGTH = 12000
STREAM_FALLBACK_CHUNK_SIZE = 24
CONTEXT_MESSAGE_TYPES = {"diagnosis-summary", "diagnosis-report"}
cloud_storage = CloudStorageService(STORAGE_CONFIG)

SYSTEM_PROMPT = """你是 Peace Skin 小程序里的健康状态辅助问询助手。

你的任务：
1. 基于当前会话中的面象/舌象辅助分析摘要、用户输入和附件内容回答。
2. 输出使用简体中文，优先直接、清楚、可执行。
3. 不要假装做出医学诊断或治疗结论，结果只定位为皮肤状态、舌象特征、体质倾向的辅助分析和护理/调养建议；遇到明显异常时建议线下就医。
4. 如果用户围绕面象、舌象或九大体质结果追问，要显式结合本次摘要回答。
5. 如果附件信息不足，明确说明你看到了什么，还缺什么。
"""


def _utc_now() -> str:
    """Return the current UTC timestamp in ISO format."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _pick_text(*values: object) -> str:
    """Return the first non-empty string value."""
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalize_responses_model_name(model_name: str) -> str:
    """Return a model name accepted by the OpenAI Responses API."""
    return str(model_name or "").removeprefix("openai:") or "gpt-5.4"


def _responses_temperature() -> float:
    """Return the configured temperature for mobile chat responses."""
    try:
        return float(os.getenv("MOBILE_CHAT_TEMPERATURE", "0.3"))
    except ValueError:
        return 0.3


def _build_responses_client() -> OpenAI:
    """Create the OpenAI-compatible Responses API client."""
    kwargs: dict[str, Any] = {"api_key": os.getenv("OPENAI_API_KEY")}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url.rstrip("/")
    return OpenAI(**kwargs)


def _extract_response_output_text(response: Any) -> str:
    """Extract plain text from a Responses API response."""
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = getattr(response, "output", None)
    if not isinstance(output, list):
        return ""

    parts: list[str] = []
    for item in output:
        content = getattr(item, "content", None)
        if not isinstance(content, list):
            continue
        for part in content:
            text = getattr(part, "text", None)
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts).strip()


def _extract_response_stream_delta(event: Any) -> str:
    """Extract text delta from one Responses API stream event."""
    event_type = str(getattr(event, "type", "") or "")
    if event_type != "response.output_text.delta":
        return ""
    delta = getattr(event, "delta", "")
    return delta if isinstance(delta, str) else ""


def _chunk_text(text: str, *, size: int = STREAM_FALLBACK_CHUNK_SIZE) -> list[str]:
    """Split text into user-visible stream chunks."""
    if not text:
        return []
    return [text[index : index + size] for index in range(0, len(text), size)]


def _safe_excerpt(text: str, *, limit: int = MAX_ATTACHMENT_TEXT_LENGTH) -> str:
    """Trim attachment text for prompt use."""
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit]


def _guess_attachment_kind(content_type: str, suffix: str) -> str:
    """Classify an attachment as image or document."""
    if content_type.startswith("image/"):
        return "image"
    if suffix.lower() in {".txt", ".pdf"}:
        return "document"
    return "unknown"


def _extract_text_from_txt(file_path: Path) -> str:
    """Read UTF-8 or best-effort text content from a plain text file."""
    return file_path.read_text(encoding="utf-8", errors="replace")


def _extract_text_from_txt_bytes(file_bytes: bytes) -> str:
    """Read plain-text attachment content from in-memory bytes."""
    return file_bytes.decode("utf-8", errors="replace")


def _extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from a PDF file using `pypdf` when available."""
    from pypdf import PdfReader

    reader = PdfReader(str(file_path))
    chunks = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(chunks)


def _extract_text_from_pdf_bytes(file_bytes: bytes) -> str:
    """Extract text from an in-memory PDF attachment."""
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(file_bytes))
    chunks = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(chunks)


def _load_attachment_bytes(attachment: dict[str, Any]) -> bytes:
    """Load attachment bytes from object storage or local fallback."""
    object_key = _pick_text(attachment.get("object_key"))
    if object_key:
        download_success, payload = cloud_storage.download_to_memory(object_key)
        if download_success and isinstance(payload, (bytes, bytearray)):
            return bytes(payload)
        msg = f"Failed to download attachment object: {object_key}"
        raise FileNotFoundError(msg)

    stored_path = Path(str(attachment.get("stored_path") or ""))
    if stored_path.exists():
        return stored_path.read_bytes()

    msg = "Attachment bytes are unavailable."
    raise FileNotFoundError(msg)


def describe_attachment(
    *,
    session_id: str,
    attachment_id: str,
    stored_path: str | Path | None,
    object_key: str,
    original_name: str,
    size_bytes: int,
    download_url: str,
    file_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Build attachment metadata for storage and downstream model usage.

    Args:
        session_id: Owning session identifier.
        attachment_id: Stable attachment identifier.
        stored_path: Local persisted file path when available.
        object_key: Cloud object key for the persisted file.
        original_name: User-visible file name.
        size_bytes: File size in bytes.
        download_url: Flask endpoint URL for preview/download.
        file_bytes: Optional uploaded file bytes for metadata extraction.

    Returns:
        Attachment metadata used by the mobile chat endpoints.
    """
    file_path = Path(stored_path) if stored_path else None
    content_type = mimetypes.guess_type(original_name)[0] or "application/octet-stream"
    suffix = (file_path.suffix if file_path else Path(original_name).suffix).lower()
    kind = _guess_attachment_kind(content_type, suffix)
    extracted_text = ""
    extraction_error = ""

    if kind == "document":
        try:
            if suffix == ".txt":
                if file_bytes is not None:
                    extracted_text = _safe_excerpt(_extract_text_from_txt_bytes(file_bytes))
                elif file_path is not None:
                    extracted_text = _safe_excerpt(_extract_text_from_txt(file_path))
            elif suffix == ".pdf":
                if file_bytes is not None:
                    extracted_text = _safe_excerpt(_extract_text_from_pdf_bytes(file_bytes))
                elif file_path is not None:
                    extracted_text = _safe_excerpt(_extract_text_from_pdf(file_path))
        except Exception as exc:  # noqa: BLE001
            extraction_error = str(exc)

    return {
        "attachment_id": attachment_id,
        "session_id": session_id,
        "name": original_name,
        "content_type": content_type,
        "kind": kind,
        "size_bytes": size_bytes,
        "stored_path": str(file_path or ""),
        "object_key": object_key,
        "download_url": download_url,
        "text_excerpt": extracted_text,
        "extraction_error": extraction_error,
        "created_at": _utc_now(),
    }


def _diagnosis_system_prompt(diagnosis_context: dict[str, Any] | None) -> str:
    """Build the analysis-assistance system prompt fragment."""
    if not diagnosis_context:
        return "当前会话没有关联面象或舌象辅助分析结果。做完舌象/面向分析后才能给到健康 报告/方案/建议"

    diagnosis_contexts = diagnosis_context.get("diagnosisContexts") or []
    if isinstance(diagnosis_contexts, list) and diagnosis_contexts:
        sections = []
        for item in diagnosis_contexts:
            if not isinstance(item, dict):
                continue
            sections.append(_diagnosis_system_prompt(item))
        if sections:
            return "当前会话已关联多份辅助分析摘要。\n\n" + "\n\n".join(sections)

    source_label = _pick_text(
        diagnosis_context.get("sourceLabel"),
        "舌象一期" if diagnosis_context.get("sourceType") == "tongue" else "",
        "面象肤况" if diagnosis_context.get("sourceType") == "face" else "",
        "辅助分析",
    )
    highlights = diagnosis_context.get("metricHighlights") or []
    highlight_lines = []
    for item in highlights:
        if not isinstance(item, dict):
            continue
        title = _pick_text(item.get("title"), item.get("key"))
        score = item.get("score")
        summary = _pick_text(item.get("summary"))
        highlight_lines.append(f"- {title}: {score}; {summary}")

    joined_highlights = "\n".join(highlight_lines) or "- 无重点指标"
    return (
        f"当前会话已关联一份{source_label}摘要。\n"
        f"评分/质量: {_pick_text(str(diagnosis_context.get('totalScoreText') or ''), str(diagnosis_context.get('totalScore') or ''))}\n"
        f"摘要: {_pick_text(diagnosis_context.get('summary'))}\n"
        "重点特征:\n"
        f"{joined_highlights}"
    )


def _history_message_text(message: dict[str, Any]) -> str:
    """Convert a stored message into prompt text."""
    content = _pick_text(message.get("content"))
    attachments = message.get("attachments") or []
    if not attachments:
        return content

    attachment_names = ", ".join(
        str(attachment.get("name"))
        for attachment in attachments
        if isinstance(attachment, dict) and attachment.get("name")
    )
    if attachment_names:
        return f"{content}\n\n附件: {attachment_names}"
    return content


def _build_latest_user_text(
    text: str,
    attachments: list[dict[str, Any]],
) -> str:
    """Build the latest user text, including text-file summaries and image names."""
    prompt_sections = [_pick_text(text) or "请结合我上传的附件回答。"]
    document_sections: list[str] = []
    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue
        if attachment.get("kind") == "image":
            prompt_sections.append(f"已上传图片: {_pick_text(attachment.get('name'))}")
            continue

        document_text = _pick_text(attachment.get("text_excerpt"))
        if document_text:
            document_sections.append(
                f"文件 {_pick_text(attachment.get('name'))} 内容摘要:\n{document_text}"
            )
        else:
            document_sections.append(
                f"文件 {_pick_text(attachment.get('name'))} 已上传，但暂未成功提取文本。"
            )

    if document_sections:
        prompt_sections.append("\n\n".join(document_sections))

    return "\n\n".join(section for section in prompt_sections if section)


def _build_response_content(
    text: str,
    attachments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build Responses API content blocks for one user message."""
    content: list[dict[str, Any]] = [
        {"type": "input_text", "text": _build_latest_user_text(text, attachments)}
    ]

    for attachment in attachments:
        if not isinstance(attachment, dict) or attachment.get("kind") != "image":
            continue
        content_type = _pick_text(attachment.get("content_type")) or "image/jpeg"
        file_bytes = _load_attachment_bytes(attachment)
        encoded = base64.b64encode(file_bytes).decode("utf-8")
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:{content_type};base64,{encoded}",
            }
        )
    return content


def _build_response_instructions(session: dict[str, Any]) -> str:
    """Build the Responses API instruction string."""
    return "\n\n".join(
        [
            SYSTEM_PROMPT,
            _diagnosis_system_prompt(session.get("diagnosis_context")),
        ]
    )


def _build_response_input(
    session: dict[str, Any],
    user_text: str,
    attachments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build Responses API input items from a stored session and new user turn."""
    input_items: list[dict[str, Any]] = []
    history = list(session.get("messages") or [])[-MAX_HISTORY_MESSAGES:]
    for item in history:
        if not isinstance(item, dict):
            continue
        if str(item.get("message_type") or "") in CONTEXT_MESSAGE_TYPES:
            continue
        role = str(item.get("role") or "")
        content = _history_message_text(item)
        if not content:
            continue
        if role == "assistant":
            input_items.append(
                {
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content}],
                }
            )
        elif role == "user":
            input_items.append(
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": content}],
                }
            )

    input_items.append(
        {
            "role": "user",
            "content": _build_response_content(user_text, attachments),
        }
    )
    return input_items


def _fallback_reply(
    session: dict[str, Any],
    user_text: str,
    attachments: list[dict[str, Any]],
) -> str:
    """Build a deterministic reply when the LLM is unavailable."""
    diagnosis_context = session.get("diagnosis_context") or {}
    summary = _pick_text(diagnosis_context.get("summary"))
    attachment_names = ", ".join(
        _pick_text(item.get("name"))
        for item in attachments
        if isinstance(item, dict)
    )
    base_reply = "我已经收到你的问题。"
    if summary:
        source_label = _pick_text(
            diagnosis_context.get("sourceLabel"),
            "舌象一期" if diagnosis_context.get("sourceType") == "tongue" else "",
            "面象肤况" if diagnosis_context.get("sourceType") == "face" else "",
            "辅助分析",
        )
        base_reply += f" 当前会话中关联的{source_label}摘要是：{summary}"
    if attachment_names:
        base_reply += f" 我也看到了这些附件：{attachment_names}。"
    if _pick_text(user_text):
        base_reply += " 目前服务端没有可用的大模型响应，因此先建议你围绕当前症状、日常护理和作息习惯继续补充信息。"
    else:
        base_reply += " 请继续描述你最关心的问题。"
    return base_reply


def generate_reply(
    session: dict[str, Any],
    *,
    user_text: str,
    attachments: list[dict[str, Any]] | None = None,
) -> str:
    """Generate one non-streaming assistant reply.

    Args:
        session: Current session payload.
        user_text: Latest user input text.
        attachments: Uploaded attachment metadata for this turn.

    Returns:
        Assistant reply text.
    """
    resolved_attachments = list(attachments or [])
    if not os.getenv("OPENAI_API_KEY"):
        return _fallback_reply(session, user_text, resolved_attachments)

    try:
        client = _build_responses_client()
        response = client.responses.create(
            model=_normalize_responses_model_name(MOBILE_CHAT_MODEL),
            instructions=_build_response_instructions(session),
            input=_build_response_input(session, user_text, resolved_attachments),
            temperature=_responses_temperature(),
            store=False,
        )
        reply = _extract_response_output_text(response)
        if reply:
            return reply
    except Exception as exc:  # noqa: BLE001
        logger.exception("Mobile chat Responses API invoke failed: %s", exc)
    return _fallback_reply(session, user_text, resolved_attachments)


def stream_reply_chunks(
    session: dict[str, Any],
    *,
    user_text: str,
    attachments: list[dict[str, Any]] | None = None,
) -> tuple[list[str], str]:
    """Generate streamable reply chunks.

    Args:
        session: Current session payload.
        user_text: Latest user input text.
        attachments: Uploaded attachment metadata for this turn.

    Returns:
        A tuple of stream chunks and the final concatenated reply text.
    """
    resolved_attachments = list(attachments or [])
    if os.getenv("OPENAI_API_KEY"):
        try:
            client = _build_responses_client()
            chunks: list[str] = []
            for event in client.responses.create(
                model=_normalize_responses_model_name(MOBILE_CHAT_MODEL),
                instructions=_build_response_instructions(session),
                input=_build_response_input(session, user_text, resolved_attachments),
                temperature=_responses_temperature(),
                store=False,
                stream=True,
            ):
                text = _extract_response_stream_delta(event)
                if text:
                    chunks.append(text)
            if chunks:
                return chunks, "".join(chunks)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Mobile chat Responses API stream failed: %s", exc)

    reply = _fallback_reply(session, user_text, resolved_attachments)
    return _chunk_text(reply), reply


def create_message(
    *,
    role: str,
    content: str,
    message_type: str = "text",
    attachments: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a normalized stored message payload."""
    return {
        "message_id": uuid.uuid4().hex,
        "role": role,
        "content": content,
        "message_type": message_type,
        "attachments": list(attachments or []),
        "metadata": dict(metadata or {}),
        "created_at": _utc_now(),
    }


def create_diagnosis_message(diagnosis_context: dict[str, Any]) -> dict[str, Any]:
    """Create the UI message payload for an assistance summary card."""
    return create_message(
        role="assistant",
        content=_pick_text(diagnosis_context.get("summary")),
        message_type="diagnosis-summary",
        metadata=dict(diagnosis_context),
    )


def create_diagnosis_report_message(diagnosis_report: dict[str, Any]) -> dict[str, Any]:
    """Create the UI message payload for an expandable skin assistance report card."""
    return create_message(
        role="assistant",
        content=_pick_text(
            diagnosis_report.get("summary"),
            "本次辅助分析详情已生成，点击展开查看。",
        ),
        message_type="diagnosis-report",
        metadata=dict(diagnosis_report),
    )


__all__ = [
    "create_diagnosis_message",
    "create_diagnosis_report_message",
    "create_message",
    "describe_attachment",
    "generate_reply",
    "stream_reply_chunks",
]
