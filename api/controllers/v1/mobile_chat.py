"""Mobile chat endpoints for the WeChat Mini Program."""

from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import tempfile
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from flask import Response, jsonify, request, send_file, stream_with_context, url_for
from werkzeug.utils import secure_filename

from api.configs.storage_config import STORAGE_CONFIG
from api.controllers.v1 import bp
from api.controllers.v1.mobile_contract import build_object_key
from api.middleware.storage.cloud_storage_service import CloudStorageService
from api.services.chat_session_store import ChatSessionStore
from api.services.chat_runtime import build_chat_runtime
from api.services.chat_session_store_factory import build_chat_session_store
from api.services.diagnosis_summary import (
    build_combined_diagnosis_context,
    build_combined_diagnosis_report,
    build_diagnosis_context,
    build_diagnosis_report,
)
from api.services.face_chat_analysis import (
    analyze_face_image_to_context,
    should_run_face_analysis,
)
from api.services.mobile_chat_service import (
    create_diagnosis_message,
    create_diagnosis_report_message,
    create_message,
    describe_attachment,
    generate_reply,
    stream_reply_chunks,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPLOAD_BASE_DIR = PROJECT_ROOT / "upload_files"
CHAT_SESSION_DIR = UPLOAD_BASE_DIR / "chat_sessions"
CHAT_ATTACHMENT_DIR = UPLOAD_BASE_DIR / "chat_attachments"

CHAT_SESSION_DIR.mkdir(parents=True, exist_ok=True)
CHAT_ATTACHMENT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_ATTACHMENT_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".txt",
    ".pdf",
}

CHAT_ATTACHMENT_OBJECT_PREFIX = str(
    os.getenv("CHAT_ATTACHMENT_OBJECT_PREFIX", "chat-attachments")
).strip("/")
CHAT_RATE_LIMIT_MAX_REQUESTS = int(os.getenv("CHAT_RATE_LIMIT_MAX_REQUESTS", "30"))
CHAT_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("CHAT_RATE_LIMIT_WINDOW_SECONDS", "60"))
CHAT_SESSION_LOCK_TTL_SECONDS = int(os.getenv("CHAT_SESSION_LOCK_TTL_SECONDS", "180"))
CHAT_STREAM_STATE_TTL_SECONDS = int(os.getenv("CHAT_STREAM_STATE_TTL_SECONDS", "180"))

session_store = build_chat_session_store(CHAT_SESSION_DIR)
chat_runtime = build_chat_runtime()
cloud_storage = CloudStorageService(STORAGE_CONFIG)


def _sanitize_session_id(session_id: str | None) -> str:
    """Normalize a user supplied session identifier."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "", str(session_id or "").strip())
    return cleaned or uuid.uuid4().hex


def _json_payload() -> dict[str, Any]:
    """Return the JSON request body or an empty object."""
    payload = request.get_json(silent=True)
    if isinstance(payload, dict):
        return payload
    return {}


def _attachment_dir(session_id: str) -> Path:
    """Return the directory used to store attachments for a session."""
    target = CHAT_ATTACHMENT_DIR / session_id
    target.mkdir(parents=True, exist_ok=True)
    return target


def _delete_attachment_dir(session_id: str) -> None:
    """Delete all persisted attachments for one session."""
    target = CHAT_ATTACHMENT_DIR / session_id
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)


def _delete_object_prefix(prefix: str) -> None:
    """Delete all cloud objects stored under one prefix when supported."""
    normalized_prefix = str(prefix or "").strip().strip("/")
    if not normalized_prefix:
        return

    list_files = getattr(cloud_storage, "list_files", None)
    delete_file = getattr(cloud_storage, "delete_file", None)
    if not callable(list_files) or not callable(delete_file):
        return

    try:
        for item in list_files(prefix=normalized_prefix):
            object_key = str((item or {}).get("key") or "").strip()
            if object_key:
                delete_file(object_key)
    except Exception:  # noqa: BLE001
        return


def _delete_session_artifacts(session: dict[str, Any]) -> None:
    """Delete best-effort local and cloud artifacts for one chat session."""
    deleted_object_keys: set[str] = set()
    for attachment in list(session.get("attachments") or []):
        object_key = str((attachment or {}).get("object_key") or "").strip()
        if not object_key or object_key in deleted_object_keys:
            continue
        delete_file = getattr(cloud_storage, "delete_file", None)
        if callable(delete_file):
            try:
                delete_file(object_key)
            except Exception:  # noqa: BLE001
                pass
        deleted_object_keys.add(object_key)

    diagnosis_context = session.get("diagnosis_context") or {}
    if isinstance(diagnosis_context, dict):
        source_folder = str(diagnosis_context.get("sourceFolder") or "").strip()
        if source_folder:
            _delete_object_prefix(source_folder)
            local_dir = UPLOAD_BASE_DIR / source_folder
            if local_dir.exists():
                shutil.rmtree(local_dir, ignore_errors=True)


def _resolve_attachment(session_id: str, attachment_id: str) -> dict[str, Any] | None:
    """Resolve stored attachment metadata from the session store."""
    attachments = session_store.resolve_attachments(session_id, [attachment_id])
    if not attachments:
        return None
    return dict(attachments[0])


def _load_attachment_bytes(attachment: dict[str, Any]) -> bytes:
    """Load attachment bytes from object storage or local fallback."""
    object_key = str(attachment.get("object_key") or "").strip()
    if object_key:
        download_success, payload = cloud_storage.download_to_memory(object_key)
        if download_success and isinstance(payload, (bytes, bytearray)):
            return bytes(payload)
        msg = f"Attachment object download failed: {object_key}"
        raise FileNotFoundError(msg)

    stored_path = str(attachment.get("stored_path") or "").strip()
    path = Path(stored_path) if stored_path else None
    if path is not None and path.is_file():
        return path.read_bytes()

    msg = "Attachment file not found."
    raise FileNotFoundError(msg)


def _attachment_object_key(session_id: str, stored_name: str) -> str:
    """Build the cloud object key for one chat attachment."""
    folder = f"{CHAT_ATTACHMENT_OBJECT_PREFIX}/{session_id}"
    return build_object_key(folder, stored_name)


def _sanitize_client_message_id(client_message_id: str | None) -> str:
    """Normalize a client-supplied idempotency key."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "", str(client_message_id or "").strip())
    return cleaned


def _request_rate_limit_key(session_id: str) -> str:
    """Build the Redis rate-limit key for one request."""
    remote_addr = str(request.headers.get("X-Forwarded-For") or request.remote_addr or "").strip()
    return f"{session_id}:{remote_addr or 'unknown'}"


def _record_command_event(
    session_id: str,
    *,
    event_type: str,
    command_name: str = "",
    payload: dict[str, Any] | None = None,
) -> None:
    """Record one operation event when the store supports it."""
    record_event = getattr(session_store, "record_command_event", None)
    if callable(record_event):
        record_event(
            session_id,
            event_type=event_type,
            command_name=command_name,
            payload=payload,
        )


def _invalidate_recent_sessions_cache() -> None:
    """Invalidate cached recent-session lists."""
    chat_runtime.invalidate_recent_sessions()


def _recent_sessions(limit: int) -> list[dict[str, Any]]:
    """Return recent sessions with Redis caching when available."""
    cached_sessions = chat_runtime.get_recent_sessions(limit=limit)
    if cached_sessions is not None:
        return cached_sessions

    sessions = session_store.list_sessions(limit=limit)
    chat_runtime.set_recent_sessions(limit=limit, sessions=sessions)
    return sessions


def _ensure_request_allowed(session_id: str) -> Response | None:
    """Reject requests that exceed the configured rate limit."""
    allowed = chat_runtime.allow_request(
        _request_rate_limit_key(session_id),
        max_requests=CHAT_RATE_LIMIT_MAX_REQUESTS,
        window_seconds=CHAT_RATE_LIMIT_WINDOW_SECONDS,
    )
    if allowed:
        return None

    return (
        jsonify(
            {
                "error": "rate limit exceeded",
                "message": "Too many chat requests. Please retry shortly.",
            }
        ),
        429,
    )


def _existing_turn(session_id: str, client_message_id: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return an already-completed user/assistant turn by client message id."""
    find_turn_messages = getattr(session_store, "find_turn_messages", None)
    if not callable(find_turn_messages):
        return None, None
    return find_turn_messages(session_id, client_message_id)


def _attachment_temp_file(attachment: dict[str, Any]) -> tuple[Path, Callable[[], None]]:
    """Materialize an attachment to a temporary local file."""
    stored_path = str(attachment.get("stored_path") or "").strip()
    path = Path(stored_path) if stored_path else None
    if path is not None and path.is_file():
        return path, lambda: None

    suffix = _supported_suffix(str(attachment.get("name") or "attachment.bin")) or ".bin"
    temp_dir = Path(tempfile.mkdtemp(prefix="chat-attachment-", dir=CHAT_ATTACHMENT_DIR))
    temp_path = temp_dir / f"materialized{suffix}"
    temp_path.write_bytes(_load_attachment_bytes(attachment))

    def cleanup() -> None:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return temp_path, cleanup


def _supported_suffix(filename: str) -> str:
    """Return the supported lowercase filename suffix."""
    return Path(filename).suffix.lower()


def _serialize_stream_event(payload: dict[str, Any]) -> str:
    """Serialize one NDJSON stream event."""
    return json.dumps(payload, ensure_ascii=False) + "\n"


def _prepare_face_analysis_session(
    *,
    session_id: str,
    session_before_turn: dict[str, Any],
    attachments: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run face analysis on the first image attachment and update session state."""
    image_attachment = next(
        (
            attachment
            for attachment in attachments
            if isinstance(attachment, dict) and attachment.get("kind") == "image"
        ),
        None,
    )
    if image_attachment is None:
        msg = "No image attachment available for face analysis."
        raise ValueError(msg)

    attachment_id = str(image_attachment.get("attachment_id") or uuid.uuid4().hex)
    materialized_path, cleanup = _attachment_temp_file(image_attachment)
    try:
        analysis_payload, diagnosis_context = analyze_face_image_to_context(
            str(materialized_path),
            source_folder=f"chat-face-{session_id}-{attachment_id}",
        )
    finally:
        cleanup()
    diagnosis_report = build_diagnosis_report(analysis_payload)
    updated_session = session_store.set_diagnosis_context(session_id, diagnosis_context)
    diagnosis_message = create_diagnosis_message(diagnosis_context)
    updated_session = session_store.add_or_replace_diagnosis_message(
        session_id,
        diagnosis_message,
    )
    diagnosis_report_message = create_diagnosis_report_message(diagnosis_report)
    updated_session = session_store.add_or_replace_diagnosis_report_message(
        session_id,
        diagnosis_report_message,
    )
    diagnosis_ready_session = {
        **dict(session_before_turn),
        "diagnosis_context": diagnosis_context,
    }
    return diagnosis_ready_session, diagnosis_context, updated_session


@bp.route("/mobile/chat/session", methods=["POST"])
def mobile_chat_session() -> Response:
    """Create or restore a mobile chat session."""
    payload = _json_payload()
    session_id = _sanitize_session_id(payload.get("session_id"))
    session = session_store.create_session(session_id)
    _record_command_event(
        session["session_id"],
        event_type="session.create",
        payload={"restored": bool(payload.get("session_id"))},
    )
    _invalidate_recent_sessions_cache()
    return jsonify({"status": "success", "session": session, "session_id": session["session_id"]})


@bp.route("/mobile/chat/session/<session_id>", methods=["GET"])
def mobile_chat_session_detail(session_id: str) -> Response:
    """Return the current session state."""
    session = session_store.get_session(_sanitize_session_id(session_id))
    if session is None:
        return jsonify({"error": "session not found", "message": "Unknown session id."}), 404
    return jsonify({"status": "success", "session": session, "session_id": session["session_id"]})


@bp.route("/mobile/chat/session/<session_id>", methods=["PATCH"])
def mobile_chat_session_update(session_id: str) -> Response:
    """Update user-editable chat session metadata."""
    normalized_session_id = _sanitize_session_id(session_id)
    payload = _json_payload()
    title = payload.get("title")
    pinned = payload.get("pinned")
    session = session_store.update_session_metadata(
        normalized_session_id,
        title=str(title).strip() if title is not None else None,
        pinned=bool(pinned) if isinstance(pinned, bool) else None,
    )
    _record_command_event(
        normalized_session_id,
        event_type="session.update",
        command_name="session.update",
        payload={"title": title, "pinned": pinned},
    )
    _invalidate_recent_sessions_cache()
    return jsonify({"status": "success", "session": session, "session_id": session["session_id"]})


@bp.route("/mobile/chat/session/<session_id>", methods=["DELETE"])
def mobile_chat_session_delete(session_id: str) -> Response:
    """Delete one persisted mobile chat session."""
    normalized_session_id = _sanitize_session_id(session_id)
    existing_session = session_store.get_session(normalized_session_id)
    if existing_session is None:
        return jsonify({"error": "session not found", "message": "Unknown session id."}), 404

    _delete_session_artifacts(existing_session)
    removed = session_store.delete_session(normalized_session_id)
    _delete_attachment_dir(normalized_session_id)
    if not removed:
        return jsonify({"error": "session not found", "message": "Unknown session id."}), 404
    _invalidate_recent_sessions_cache()
    return jsonify({"status": "success", "session_id": normalized_session_id})


@bp.route("/mobile/chat/sessions", methods=["GET"])
def mobile_chat_sessions() -> Response:
    """Return recent mobile chat sessions for thread switching."""
    limit = request.args.get("limit", default=12, type=int) or 12
    bounded_limit = min(max(limit, 1), 50)
    sessions = _recent_sessions(bounded_limit)
    return jsonify({"status": "success", "sessions": sessions, "count": len(sessions)})


@bp.route("/mobile/chat/attachment", methods=["POST"])
def mobile_chat_attachment() -> Response:
    """Upload one image or document for a chat session."""
    if "file" not in request.files:
        return jsonify({"error": "missing file", "message": "Expected a multipart `file` field."}), 400

    file_storage = request.files["file"]
    if not file_storage or not file_storage.filename:
        return jsonify({"error": "empty file", "message": "The uploaded file is empty."}), 400

    session_id = _sanitize_session_id(request.form.get("session_id"))
    session_store.create_session(session_id)

    original_name = secure_filename(file_storage.filename or "")
    suffix = _supported_suffix(original_name)
    if suffix not in ALLOWED_ATTACHMENT_SUFFIXES:
        return (
            jsonify(
                {
                    "error": "unsupported attachment type",
                    "message": "Only jpg, jpeg, png, webp, txt, and pdf are supported.",
                }
            ),
            400,
        )

    attachment_id = uuid.uuid4().hex
    stored_name = f"{attachment_id}{suffix}"
    object_key = _attachment_object_key(session_id, stored_name)
    file_bytes = file_storage.read()
    if not file_bytes:
        return jsonify({"error": "empty file", "message": "The uploaded file is empty."}), 400

    upload_success, upload_result = cloud_storage.upload_from_memory(
        file_bytes,
        object_key,
        content_type=file_storage.mimetype or "application/octet-stream",
    )
    if not upload_success:
        return jsonify({"error": "upload failed", "message": str(upload_result)}), 500
    size_bytes = len(file_bytes)

    download_url = url_for(
        "v1.mobile_chat_attachment_file",
        session_id=session_id,
        attachment_id=attachment_id,
        _external=True,
    )
    attachment = describe_attachment(
        session_id=session_id,
        attachment_id=attachment_id,
        stored_path=None,
        object_key=object_key,
        original_name=original_name,
        size_bytes=size_bytes,
        download_url=download_url,
        file_bytes=file_bytes,
    )
    session, normalized_attachment = session_store.add_or_replace_attachment(
        session_id,
        attachment,
    )
    _record_command_event(
        session_id,
        event_type="attachment.upload",
        command_name="attachment.upload",
        payload={"attachment_id": attachment_id, "object_key": object_key},
    )
    _invalidate_recent_sessions_cache()
    return jsonify(
        {
            "status": "success",
            "attachment": normalized_attachment,
            "session_id": session_id,
            "session": session,
        }
    )


@bp.route("/mobile/chat/attachment/<session_id>/<attachment_id>", methods=["GET"])
def mobile_chat_attachment_file(session_id: str, attachment_id: str) -> Response:
    """Serve a stored chat attachment file."""
    normalized_session_id = _sanitize_session_id(session_id)
    attachment = _resolve_attachment(normalized_session_id, attachment_id)
    if attachment is None:
        return jsonify({"error": "file not found", "message": "Attachment not found."}), 404

    content_type = (
        str(attachment.get("content_type") or "").strip()
        or mimetypes.guess_type(str(attachment.get("name") or ""))[0]
        or "application/octet-stream"
    )
    file_name = str(attachment.get("name") or attachment_id)

    try:
        file_bytes = _load_attachment_bytes(attachment)
    except FileNotFoundError:
        return jsonify({"error": "file not found", "message": "Attachment not found."}), 404

    return send_file(BytesIO(file_bytes), mimetype=content_type, download_name=file_name)


@bp.route("/mobile/chat/message", methods=["POST"])
def mobile_chat_message() -> Response:
    """Handle one non-streaming mobile chat turn."""
    payload = _json_payload()
    session_id = _sanitize_session_id(payload.get("session_id"))
    rate_limit_response = _ensure_request_allowed(session_id)
    if rate_limit_response is not None:
        return rate_limit_response

    client_message_id = _sanitize_client_message_id(payload.get("client_message_id"))
    existing_user_message, existing_assistant_message = _existing_turn(
        session_id,
        client_message_id,
    )
    if existing_user_message is not None and existing_assistant_message is not None:
        existing_session = session_store.get_session(session_id) or session_store.create_session(session_id)
        return jsonify(
            {
                "status": "success",
                "session_id": session_id,
                "user_message": existing_user_message,
                "assistant_message": existing_assistant_message,
                "session": existing_session,
            }
        )

    lock_owner = chat_runtime.acquire_session_lock(
        session_id,
        ttl_seconds=CHAT_SESSION_LOCK_TTL_SECONDS,
    )
    if lock_owner is None:
        return (
            jsonify(
                {
                    "error": "session busy",
                    "message": "Assistant is already generating a reply for this session.",
                }
            ),
            409,
        )

    text = str(payload.get("text") or "").strip()
    attachment_ids = [
        str(item)
        for item in list(payload.get("attachment_ids") or [])
        if str(item).strip()
    ]
    try:
        session = session_store.create_session(session_id)
        session_before_turn = dict(session)
        attachments = session_store.resolve_attachments(session_id, attachment_ids)
        resumed_turn = existing_user_message is not None and existing_assistant_message is None
        if resumed_turn:
            session_before_turn["messages"] = [
                message
                for message in list(session_before_turn.get("messages") or [])
                if str((message.get("metadata") or {}).get("client_message_id") or "") != client_message_id
            ]

        message_metadata = {"client_message_id": client_message_id} if client_message_id else {}
        if resumed_turn:
            user_message = existing_user_message or {}
        else:
            user_message = create_message(
                role="user",
                content=text,
                attachments=attachments,
                metadata=message_metadata,
            )
            session_store.append_messages(session_id, [user_message])

        if should_run_face_analysis(user_text=text, attachments=attachments):
            analysis_session, diagnosis_context, session = _prepare_face_analysis_session(
                session_id=session_id,
                session_before_turn=session_before_turn,
                attachments=attachments,
            )
            reply = generate_reply(analysis_session, user_text=text, attachments=[])
            assistant_metadata = {
                **message_metadata,
                "analysis_triggered": True,
                "diagnosis_context": diagnosis_context,
            }
            assistant_message = create_message(
                role="assistant",
                content=reply,
                metadata=assistant_metadata,
            )
        else:
            reply = generate_reply(session_before_turn, user_text=text, attachments=attachments)
            assistant_message = create_message(
                role="assistant",
                content=reply,
                metadata=message_metadata,
            )
        session = session_store.append_messages(session_id, [assistant_message])
        _record_command_event(
            session_id,
            event_type="chat.message",
            command_name="chat.message",
            payload={
                "client_message_id": client_message_id,
                "attachment_count": len(attachments),
                "resumed_turn": resumed_turn,
            },
        )
        _invalidate_recent_sessions_cache()
        return jsonify(
            {
                "status": "success",
                "session_id": session_id,
                "user_message": user_message,
                "assistant_message": assistant_message,
                "session": session,
            }
        )
    finally:
        chat_runtime.release_session_lock(session_id, lock_owner)
        chat_runtime.clear_stream_state(session_id)


@bp.route("/mobile/chat/stream", methods=["POST"])
def mobile_chat_stream() -> Response:
    """Handle one streaming mobile chat turn using NDJSON events."""
    payload = _json_payload()
    session_id = _sanitize_session_id(payload.get("session_id"))
    rate_limit_response = _ensure_request_allowed(session_id)
    if rate_limit_response is not None:
        return rate_limit_response

    client_message_id = _sanitize_client_message_id(payload.get("client_message_id"))
    existing_user_message, existing_assistant_message = _existing_turn(
        session_id,
        client_message_id,
    )
    if existing_user_message is not None and existing_assistant_message is not None:
        existing_session = session_store.get_session(session_id) or session_store.create_session(session_id)

        @stream_with_context
        def replay_existing() -> Any:
            yield _serialize_stream_event(
                {
                    "type": "start",
                    "session_id": session_id,
                    "user_message": existing_user_message,
                }
            )
            yield _serialize_stream_event(
                {
                    "type": "done",
                    "assistant_message": existing_assistant_message,
                    "session": existing_session,
                    "session_id": session_id,
                }
            )

        return Response(replay_existing(), mimetype="application/x-ndjson")

    lock_owner = chat_runtime.acquire_session_lock(
        session_id,
        ttl_seconds=CHAT_SESSION_LOCK_TTL_SECONDS,
    )
    if lock_owner is None:
        return (
            jsonify(
                {
                    "error": "session busy",
                    "message": "Assistant is already generating a reply for this session.",
                }
            ),
            409,
        )

    text = str(payload.get("text") or "").strip()
    attachment_ids = [
        str(item)
        for item in list(payload.get("attachment_ids") or [])
        if str(item).strip()
    ]
    session = session_store.create_session(session_id)
    session_before_turn = dict(session)
    attachments = session_store.resolve_attachments(session_id, attachment_ids)
    resumed_turn = existing_user_message is not None and existing_assistant_message is None
    if resumed_turn:
        session_before_turn["messages"] = [
            message
            for message in list(session_before_turn.get("messages") or [])
            if str((message.get("metadata") or {}).get("client_message_id") or "") != client_message_id
        ]
    message_metadata = {"client_message_id": client_message_id} if client_message_id else {}
    if resumed_turn:
        user_message = existing_user_message or {}
    else:
        user_message = create_message(
            role="user",
            content=text,
            attachments=attachments,
            metadata=message_metadata,
        )
        session_store.append_messages(session_id, [user_message])
    should_analyze_face = should_run_face_analysis(user_text=text, attachments=attachments)

    @stream_with_context
    def generate() -> Any:
        chat_runtime.set_stream_state(
            session_id,
            {"client_message_id": client_message_id, "status": "running"},
            ttl_seconds=CHAT_STREAM_STATE_TTL_SECONDS,
        )
        yield _serialize_stream_event(
            {
                "type": "start",
                "session_id": session_id,
                "user_message": user_message,
            }
        )
        try:
            analysis_session = session_before_turn
            diagnosis_context = None
            if should_analyze_face:
                analysis_session, diagnosis_context, _ = _prepare_face_analysis_session(
                    session_id=session_id,
                    session_before_turn=session_before_turn,
                    attachments=attachments,
                )
                yield _serialize_stream_event(
                    {
                        "type": "context",
                        "diagnosis_context": diagnosis_context,
                    }
                )

            chunks, reply = stream_reply_chunks(
                analysis_session,
                user_text=text,
                attachments=[],
            )
            for chunk in chunks:
                yield _serialize_stream_event({"type": "delta", "delta": chunk})

            assistant_message = create_message(
                role="assistant",
                content=reply,
                metadata={
                    **message_metadata,
                    "analysis_triggered": should_analyze_face,
                    "diagnosis_context": diagnosis_context,
                }
                if should_analyze_face
                else message_metadata,
            )
            updated_session = session_store.append_messages(session_id, [assistant_message])
            _record_command_event(
                session_id,
                event_type="chat.stream",
                command_name="chat.stream",
                payload={
                    "client_message_id": client_message_id,
                    "attachment_count": len(attachments),
                    "resumed_turn": resumed_turn,
                },
            )
            _invalidate_recent_sessions_cache()
            yield _serialize_stream_event(
                {
                    "type": "done",
                    "assistant_message": assistant_message,
                    "session": updated_session,
                    "session_id": session_id,
                }
            )
        except Exception as exc:  # noqa: BLE001
            yield _serialize_stream_event(
                {
                    "type": "error",
                    "message": str(exc),
                    "session_id": session_id,
                }
            )
        finally:
            chat_runtime.clear_stream_state(session_id)
            chat_runtime.release_session_lock(session_id, lock_owner)

    return Response(generate(), mimetype="application/x-ndjson")


@bp.route("/mobile/chat/diagnosis-context", methods=["POST"])
def mobile_chat_diagnosis_context() -> Response:
    """Store a skin assistance summary into the current chat session."""
    payload = _json_payload()
    session_id = _sanitize_session_id(payload.get("session_id"))
    session_store.create_session(session_id)

    analysis_result = payload.get("analysis_result")
    analysis_results = payload.get("analysis_results")
    if isinstance(analysis_results, list):
        normalized_analysis_results = [
            item for item in analysis_results if isinstance(item, dict)
        ]
    elif isinstance(analysis_result, dict):
        normalized_analysis_results = [analysis_result]
    else:
        normalized_analysis_results = []

    if not normalized_analysis_results:
        return (
            jsonify(
                {
                    "error": "missing analysis result",
                    "message": "Expected an `analysis_result` object or `analysis_results` array.",
                }
            ),
            400,
        )

    diagnosis_context = build_combined_diagnosis_context(normalized_analysis_results)
    diagnosis_report = build_combined_diagnosis_report(normalized_analysis_results)
    session = session_store.set_diagnosis_context(session_id, diagnosis_context)
    diagnosis_message = create_diagnosis_message(diagnosis_context)
    session = session_store.add_or_replace_diagnosis_message(session_id, diagnosis_message)
    diagnosis_report_message = create_diagnosis_report_message(diagnosis_report)
    session = session_store.add_or_replace_diagnosis_report_message(
        session_id,
        diagnosis_report_message,
    )
    _record_command_event(
        session_id,
        event_type="diagnosis.set",
        command_name="diagnosis.set",
        payload={"source_folder": diagnosis_context.get("sourceFolder")},
    )
    _invalidate_recent_sessions_cache()
    return jsonify(
        {
            "status": "success",
            "session_id": session_id,
            "diagnosis_context": diagnosis_context,
            "diagnosis_message": diagnosis_message,
            "diagnosis_report_message": diagnosis_report_message,
            "session": session,
        }
    )


@bp.route("/mobile/chat/diagnosis-context/<session_id>", methods=["DELETE"])
def mobile_chat_clear_diagnosis_context(session_id: str) -> Response:
    """Remove the skin assistance context from one chat session."""
    normalized_session_id = _sanitize_session_id(session_id)
    existing_session = session_store.get_session(normalized_session_id)
    if existing_session is None:
        return jsonify({"error": "session not found", "message": "Unknown session id."}), 404

    session = session_store.clear_diagnosis_context(normalized_session_id)
    _record_command_event(
        normalized_session_id,
        event_type="diagnosis.clear",
        command_name="diagnosis.clear",
    )
    _invalidate_recent_sessions_cache()
    return jsonify({"status": "success", "session": session, "session_id": normalized_session_id})
