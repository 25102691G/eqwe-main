"""Persist lightweight mobile chat sessions."""

from __future__ import annotations

import json
import uuid
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(UTC).isoformat()


def _message_timestamp(message: dict[str, Any]) -> str:
    """Return the best-effort timestamp for a message object."""
    created_at = message.get("created_at")
    if isinstance(created_at, str) and created_at:
        return created_at
    return _utc_now()


class ChatSessionStore:
    """Store mobile chat sessions in memory and on disk."""

    def __init__(self, root_dir: str | Path) -> None:
        """Initialize the session store.

        Args:
            root_dir: Directory used to persist session JSON files.
        """
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict[str, Any]] = {}

    def _session_path(self, session_id: str) -> Path:
        """Return the JSON file path for a session."""
        return self.root_dir / f"{session_id}.json"

    def _normalize_session(self, session: dict[str, Any]) -> dict[str, Any]:
        """Fill default fields for a session payload."""
        now = _utc_now()
        return {
            "session_id": str(session.get("session_id") or uuid.uuid4().hex),
            "created_at": str(session.get("created_at") or now),
            "updated_at": str(session.get("updated_at") or now),
            "title": str(session.get("title") or ""),
            "pinned": bool(session.get("pinned")),
            "messages": list(session.get("messages") or []),
            "attachments": list(session.get("attachments") or []),
            "diagnosis_context": session.get("diagnosis_context"),
            "command_events": list(session.get("command_events") or []),
        }

    def create_session(self, session_id: str | None = None) -> dict[str, Any]:
        """Create a new session or return the existing one.

        Args:
            session_id: Optional explicit session identifier.

        Returns:
            The normalized session payload.
        """
        normalized_session_id = str(session_id or uuid.uuid4().hex)
        existing = self.get_session(normalized_session_id)
        if existing is not None:
            return existing

        session = self._normalize_session({"session_id": normalized_session_id})
        return self.save_session(session)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Load a session from memory or disk.

        Args:
            session_id: Session identifier.

        Returns:
            The session payload if present, else `None`.
        """
        if session_id in self._cache:
            return deepcopy(self._cache[session_id])

        path = self._session_path(session_id)
        if not path.exists():
            return None

        payload = json.loads(path.read_text(encoding="utf-8"))
        session = self._normalize_session(payload)
        self._cache[session_id] = session
        return deepcopy(session)

    def save_session(self, session: dict[str, Any]) -> dict[str, Any]:
        """Persist a session payload.

        Args:
            session: Session payload to write.

        Returns:
            The normalized saved session.
        """
        normalized = self._normalize_session(session)
        normalized["updated_at"] = _utc_now()

        session_id = str(normalized["session_id"])
        self._cache[session_id] = deepcopy(normalized)
        self._session_path(session_id).write_text(
            json.dumps(normalized, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return deepcopy(normalized)

    def _session_summary(self, session: dict[str, Any]) -> dict[str, Any]:
        """Build compact session metadata for list views."""
        messages = list(session.get("messages") or [])
        last_message = next(
            (
                item
                for item in reversed(messages)
                if isinstance(item, dict) and str(item.get("content") or "").strip()
            ),
            None,
        )
        diagnosis_context = session.get("diagnosis_context")
        diagnosis_summary = (
            str(diagnosis_context.get("summary") or "").strip()
            if isinstance(diagnosis_context, dict)
            else ""
        )
        last_message_preview = (
            str((last_message or {}).get("content") or "").strip()[:120]
            if isinstance(last_message, dict)
            else ""
        )
        return {
            "session_id": str(session.get("session_id") or ""),
            "created_at": str(session.get("created_at") or ""),
            "updated_at": str(session.get("updated_at") or ""),
            "title": str(session.get("title") or ""),
            "pinned": bool(session.get("pinned")),
            "message_count": len(messages),
            "last_message_preview": last_message_preview,
            "last_message_role": str((last_message or {}).get("role") or ""),
            "last_message_type": str((last_message or {}).get("message_type") or ""),
            "has_diagnosis_context": bool(diagnosis_summary),
            "diagnosis_summary": diagnosis_summary,
        }

    def list_sessions(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent sessions ordered by `updated_at` descending.

        Args:
            limit: Maximum number of sessions to return.

        Returns:
            Compact session metadata for recent-session views.
        """
        normalized_limit = max(1, limit)
        sessions: list[dict[str, Any]] = []
        seen_session_ids: set[str] = set()

        for path in self.root_dir.glob("*.json"):
            session_id = path.stem
            session = self.get_session(session_id)
            if session is None:
                continue
            seen_session_ids.add(session_id)
            sessions.append(self._session_summary(session))

        for session_id, cached_session in self._cache.items():
            if session_id in seen_session_ids:
                continue
            sessions.append(self._session_summary(cached_session))

        sessions.sort(
            key=lambda item: (
                bool(item.get("pinned")),
                str(item.get("updated_at") or ""),
                str(item.get("session_id") or ""),
            ),
            reverse=True,
        )
        return sessions[:normalized_limit]

    def update_session_metadata(
        self,
        session_id: str,
        *,
        title: str | None = None,
        pinned: bool | None = None,
    ) -> dict[str, Any]:
        """Update user-editable session metadata.

        Args:
            session_id: Target session identifier.
            title: Optional session title override.
            pinned: Optional pinned-state override.

        Returns:
            The updated session payload.
        """
        session = self.create_session(session_id)
        if title is not None:
            session["title"] = str(title).strip()
        if pinned is not None:
            session["pinned"] = bool(pinned)
        return self.save_session(session)

    def clear_diagnosis_context(self, session_id: str) -> dict[str, Any]:
        """Remove skin assistance context and its derived messages from a session.

        Args:
            session_id: Target session identifier.

        Returns:
            The updated session payload.
        """
        session = self.create_session(session_id)
        session["diagnosis_context"] = None
        session["messages"] = [
            message
            for message in list(session.get("messages") or [])
            if str(message.get("message_type") or "")
            not in {"diagnosis-summary", "diagnosis-report"}
        ]
        return self.save_session(session)

    def delete_session(self, session_id: str) -> bool:
        """Delete one stored session.

        Args:
            session_id: Target session identifier.

        Returns:
            `True` when the session existed and was removed.
        """
        removed = False
        if session_id in self._cache:
            del self._cache[session_id]
            removed = True

        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
            removed = True
        return removed

    def append_messages(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Append messages to a session.

        Args:
            session_id: Target session identifier.
            messages: New message payloads.

        Returns:
            The updated session payload.
        """
        session = self.create_session(session_id)
        normalized_messages = []
        for message in messages:
            normalized_message = dict(message)
            normalized_message["created_at"] = _message_timestamp(normalized_message)
            normalized_messages.append(normalized_message)
        session["messages"].extend(normalized_messages)
        return self.save_session(session)

    def add_or_replace_attachment(
        self,
        session_id: str,
        attachment: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Insert or replace an attachment for a session.

        Args:
            session_id: Target session identifier.
            attachment: Attachment metadata.

        Returns:
            The updated session and the normalized attachment payload.
        """
        session = self.create_session(session_id)
        normalized_attachment = dict(attachment)
        attachment_id = str(normalized_attachment.get("attachment_id") or uuid.uuid4().hex)
        normalized_attachment["attachment_id"] = attachment_id
        normalized_attachment["created_at"] = str(
            normalized_attachment.get("created_at") or _utc_now()
        )

        next_attachments = [
            item
            for item in session["attachments"]
            if str(item.get("attachment_id")) != attachment_id
        ]
        next_attachments.append(normalized_attachment)
        session["attachments"] = next_attachments
        updated_session = self.save_session(session)
        return updated_session, deepcopy(normalized_attachment)

    def record_command_event(
        self,
        session_id: str,
        *,
        event_type: str,
        command_name: str = "",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record one command or operation event for a session.

        Args:
            session_id: Target session identifier.
            event_type: Event category.
            command_name: Optional command name.
            payload: Additional event metadata.

        Returns:
            The updated session payload.
        """
        session = self.create_session(session_id)
        event = {
            "event_id": uuid.uuid4().hex,
            "event_type": str(event_type or "").strip(),
            "command_name": str(command_name or "").strip(),
            "payload": dict(payload or {}),
            "created_at": _utc_now(),
        }
        session["command_events"] = list(session.get("command_events") or [])
        session["command_events"].append(event)
        return self.save_session(session)

    def resolve_attachments(
        self,
        session_id: str,
        attachment_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Resolve attachment metadata by identifier.

        Args:
            session_id: Target session identifier.
            attachment_ids: Attachment identifiers to resolve.

        Returns:
            Resolved attachment payloads in input order.
        """
        if not attachment_ids:
            return []

        session = self.create_session(session_id)
        by_id = {
            str(item.get("attachment_id")): dict(item)
            for item in session["attachments"]
            if item.get("attachment_id")
        }
        return [by_id[attachment_id] for attachment_id in attachment_ids if attachment_id in by_id]

    def set_diagnosis_context(
        self,
        session_id: str,
        diagnosis_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Update the skin assistance context for a session.

        Args:
            session_id: Target session identifier.
            diagnosis_context: Skin assistance summary context.

        Returns:
            The updated session payload.
        """
        session = self.create_session(session_id)
        session["diagnosis_context"] = dict(diagnosis_context)
        return self.save_session(session)

    def find_turn_messages(
        self,
        session_id: str,
        client_message_id: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Return the stored user/assistant pair for one client turn id.

        Args:
            session_id: Target session identifier.
            client_message_id: Client-generated turn identifier.

        Returns:
            The matching user message and assistant message if present.
        """
        normalized_client_message_id = str(client_message_id or "").strip()
        if not normalized_client_message_id:
            return None, None

        session = self.get_session(session_id)
        if session is None:
            return None, None

        user_message: dict[str, Any] | None = None
        assistant_message: dict[str, Any] | None = None
        for message in list(session.get("messages") or []):
            metadata = message.get("metadata") or {}
            if not isinstance(metadata, dict):
                continue
            if str(metadata.get("client_message_id") or "").strip() != normalized_client_message_id:
                continue

            role = str(message.get("role") or "")
            if role == "user" and user_message is None:
                user_message = dict(message)
            elif role == "assistant" and assistant_message is None:
                assistant_message = dict(message)

        return deepcopy(user_message), deepcopy(assistant_message)

    def add_or_replace_diagnosis_message(
        self,
        session_id: str,
        diagnosis_message: dict[str, Any],
    ) -> dict[str, Any]:
        """Add or replace the skin assistance summary message in a session.

        Args:
            session_id: Target session identifier.
            diagnosis_message: Skin assistance card message payload.

        Returns:
            The updated session payload.
        """
        return self._add_or_replace_context_message(
            session_id,
            diagnosis_message,
            message_type="diagnosis-summary",
        )

    def add_or_replace_diagnosis_report_message(
        self,
        session_id: str,
        diagnosis_report_message: dict[str, Any],
    ) -> dict[str, Any]:
        """Add or replace the skin assistance report message in a session.

        Args:
            session_id: Target session identifier.
            diagnosis_report_message: Expandable skin assistance report payload.

        Returns:
            The updated session payload.
        """
        return self._add_or_replace_context_message(
            session_id,
            diagnosis_report_message,
            message_type="diagnosis-report",
        )

    def _add_or_replace_context_message(
        self,
        session_id: str,
        context_message: dict[str, Any],
        *,
        message_type: str,
    ) -> dict[str, Any]:
        """Add or replace a context-derived message using source folder deduping."""
        session = self.create_session(session_id)
        metadata = context_message.get("metadata", {}) or {}
        source_folder = metadata.get("sourceFolder")
        source_type = str(metadata.get("sourceType") or "").strip()
        summary = str(metadata.get("summary") or context_message.get("content") or "").strip()
        next_messages: list[dict[str, Any]] = []
        replaced = False

        for message in session["messages"]:
            current_message_type = str(message.get("message_type") or "")
            current_metadata = message.get("metadata", {}) or {}
            current_folder = current_metadata.get("sourceFolder")
            current_source_type = str(current_metadata.get("sourceType") or "").strip()
            current_summary = str(
                current_metadata.get("summary") or message.get("content") or ""
            ).strip()
            if (
                current_message_type == message_type
                and (
                    (source_type == "combined" and current_source_type == "combined")
                    or
                    (source_folder and current_folder == source_folder)
                    or (
                        not source_folder
                        and source_type
                        and summary
                        and current_source_type == source_type
                        and current_summary == summary
                    )
                )
            ):
                next_messages.append(dict(context_message))
                replaced = True
                continue
            next_messages.append(message)

        if not replaced:
            next_messages.append(dict(context_message))

        session["messages"] = next_messages
        return self.save_session(session)


__all__ = ["ChatSessionStore"]
