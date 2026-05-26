"""PostgreSQL-backed storage for mobile chat sessions."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _message_timestamp(message: dict[str, Any]) -> str:
    """Return the best-effort timestamp for a message object."""
    created_at = message.get("created_at")
    if isinstance(created_at, str) and created_at:
        return created_at
    return _utc_now()


def _isoformat(value: Any) -> str:
    """Convert timestamps from PostgreSQL to API-safe ISO strings."""
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value or "")


class PostgresChatSessionStore:
    """Store mobile chat sessions in PostgreSQL."""

    def __init__(self, dsn: str) -> None:
        """Initialize the PostgreSQL-backed chat store.

        Args:
            dsn: PostgreSQL connection string.
        """
        self.dsn = str(dsn or "").strip()
        if not self.dsn:
            msg = "PostgreSQL DSN is required."
            raise ValueError(msg)
        self._ensure_schema()

    def _connect(self) -> Any:
        """Return one PostgreSQL connection."""
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - depends on runtime env
            msg = "psycopg is required for PostgreSQL chat storage."
            raise RuntimeError(msg) from exc

        return psycopg.connect(self.dsn, connect_timeout=3)

    def _ensure_schema(self) -> None:
        """Create the required chat tables when missing."""
        statements = (
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                pinned BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                message_type TEXT NOT NULL DEFAULT 'text',
                attachments_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
            ON chat_messages (session_id, created_at DESC, message_id DESC)
            """,
            """
            CREATE TABLE IF NOT EXISTS chat_attachments (
                attachment_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                name TEXT NOT NULL DEFAULT '',
                content_type TEXT NOT NULL DEFAULT '',
                kind TEXT NOT NULL DEFAULT '',
                size_bytes BIGINT NOT NULL DEFAULT 0,
                stored_path TEXT NOT NULL DEFAULT '',
                object_key TEXT NOT NULL DEFAULT '',
                download_url TEXT NOT NULL DEFAULT '',
                text_excerpt TEXT NOT NULL DEFAULT '',
                extraction_error TEXT NOT NULL DEFAULT '',
                metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_chat_attachments_session_created
            ON chat_attachments (session_id, created_at DESC, attachment_id DESC)
            """,
            """
            CREATE TABLE IF NOT EXISTS chat_diagnosis_contexts (
                session_id TEXT PRIMARY KEY REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                source_folder TEXT NOT NULL DEFAULT '',
                total_score DOUBLE PRECISION NULL,
                summary TEXT NOT NULL DEFAULT '',
                payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS chat_command_events (
                event_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                event_type TEXT NOT NULL,
                command_name TEXT NOT NULL DEFAULT '',
                payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_chat_command_events_session_created
            ON chat_command_events (session_id, created_at DESC, event_id DESC)
            """,
        )
        with self._connect() as conn:
            with conn.cursor() as cursor:
                for statement in statements:
                    cursor.execute(statement)

    def _ensure_session_row(self, conn: Any, session_id: str) -> None:
        """Insert the session row when it does not yet exist."""
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO chat_sessions (session_id)
                VALUES (%s)
                ON CONFLICT (session_id) DO NOTHING
                """,
                (session_id,),
            )

    def _touch_session(self, conn: Any, session_id: str) -> None:
        """Update the session `updated_at` timestamp."""
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE chat_sessions
                SET updated_at = NOW()
                WHERE session_id = %s
                """,
                (session_id,),
            )

    def _session_row(self, conn: Any, session_id: str) -> tuple[Any, ...] | None:
        """Return the raw session row."""
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT session_id, title, pinned, created_at, updated_at
                FROM chat_sessions
                WHERE session_id = %s
                """,
                (session_id,),
            )
            return cursor.fetchone()

    def _load_messages(self, conn: Any, session_id: str) -> list[dict[str, Any]]:
        """Load all stored messages for one session."""
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    message_id,
                    role,
                    content,
                    message_type,
                    attachments_json::text,
                    metadata_json::text,
                    created_at
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY created_at ASC, message_id ASC
                """,
                (session_id,),
            )
            rows = cursor.fetchall()

        messages: list[dict[str, Any]] = []
        for row in rows:
            messages.append(
                {
                    "message_id": str(row[0]),
                    "role": str(row[1] or ""),
                    "content": str(row[2] or ""),
                    "message_type": str(row[3] or "text"),
                    "attachments": list(json.loads(row[4] or "[]")),
                    "metadata": dict(json.loads(row[5] or "{}")),
                    "created_at": _isoformat(row[6]),
                }
            )
        return messages

    def _load_attachments(self, conn: Any, session_id: str) -> list[dict[str, Any]]:
        """Load all stored attachments for one session."""
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    attachment_id,
                    name,
                    content_type,
                    kind,
                    size_bytes,
                    stored_path,
                    object_key,
                    download_url,
                    text_excerpt,
                    extraction_error,
                    metadata_json::text,
                    created_at
                FROM chat_attachments
                WHERE session_id = %s
                ORDER BY created_at ASC, attachment_id ASC
                """,
                (session_id,),
            )
            rows = cursor.fetchall()

        attachments: list[dict[str, Any]] = []
        for row in rows:
            metadata = dict(json.loads(row[10] or "{}"))
            attachments.append(
                {
                    "attachment_id": str(row[0]),
                    "session_id": session_id,
                    "name": str(row[1] or ""),
                    "content_type": str(row[2] or ""),
                    "kind": str(row[3] or ""),
                    "size_bytes": int(row[4] or 0),
                    "stored_path": str(row[5] or ""),
                    "object_key": str(row[6] or ""),
                    "download_url": str(row[7] or ""),
                    "text_excerpt": str(row[8] or ""),
                    "extraction_error": str(row[9] or ""),
                    "metadata": metadata,
                    "created_at": _isoformat(row[11]),
                }
            )
        return attachments

    def _load_diagnosis_context(self, conn: Any, session_id: str) -> dict[str, Any] | None:
        """Load the skin assistance context for one session."""
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT payload_json::text
                FROM chat_diagnosis_contexts
                WHERE session_id = %s
                """,
                (session_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return dict(json.loads(row[0] or "{}"))

    def _build_session(self, conn: Any, session_id: str) -> dict[str, Any] | None:
        """Build the aggregated session payload."""
        row = self._session_row(conn, session_id)
        if row is None:
            return None

        return {
            "session_id": str(row[0]),
            "title": str(row[1] or ""),
            "pinned": bool(row[2]),
            "created_at": _isoformat(row[3]),
            "updated_at": _isoformat(row[4]),
            "messages": self._load_messages(conn, session_id),
            "attachments": self._load_attachments(conn, session_id),
            "diagnosis_context": self._load_diagnosis_context(conn, session_id),
        }

    def create_session(self, session_id: str | None = None) -> dict[str, Any]:
        """Create a new session or return the existing one."""
        normalized_session_id = str(session_id or uuid.uuid4().hex)
        with self._connect() as conn:
            self._ensure_session_row(conn, normalized_session_id)
            session = self._build_session(conn, normalized_session_id)
        if session is None:
            msg = f"Failed to create chat session: {normalized_session_id}"
            raise RuntimeError(msg)
        return session

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Load a session from PostgreSQL."""
        with self._connect() as conn:
            return self._build_session(conn, session_id)

    def save_session(self, session: dict[str, Any]) -> dict[str, Any]:
        """Persist high-level session fields from an aggregated payload."""
        session_id = str(session.get("session_id") or uuid.uuid4().hex)
        title = str(session.get("title") or "")
        pinned = bool(session.get("pinned"))

        with self._connect() as conn:
            self._ensure_session_row(conn, session_id)
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE chat_sessions
                    SET title = %s,
                        pinned = %s,
                        updated_at = NOW()
                    WHERE session_id = %s
                    """,
                    (title, pinned, session_id),
                )

                diagnosis_context = session.get("diagnosis_context")
                if isinstance(diagnosis_context, dict):
                    cursor.execute(
                        """
                        INSERT INTO chat_diagnosis_contexts (
                            session_id,
                            source_folder,
                            total_score,
                            summary,
                            payload_json,
                            updated_at
                        )
                        VALUES (%s, %s, %s, %s, %s::jsonb, NOW())
                        ON CONFLICT (session_id) DO UPDATE SET
                            source_folder = EXCLUDED.source_folder,
                            total_score = EXCLUDED.total_score,
                            summary = EXCLUDED.summary,
                            payload_json = EXCLUDED.payload_json,
                            updated_at = NOW()
                        """,
                        (
                            session_id,
                            str(diagnosis_context.get("sourceFolder") or ""),
                            diagnosis_context.get("totalScore"),
                            str(diagnosis_context.get("summary") or ""),
                            json.dumps(diagnosis_context, ensure_ascii=False),
                        ),
                    )
                else:
                    cursor.execute(
                        "DELETE FROM chat_diagnosis_contexts WHERE session_id = %s",
                        (session_id,),
                    )
            return self._build_session(conn, session_id) or {}

    def _session_summary(self, conn: Any, session_id: str) -> dict[str, Any]:
        """Build compact session metadata for list views."""
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    s.session_id,
                    s.created_at,
                    s.updated_at,
                    s.title,
                    s.pinned,
                    COALESCE(msg_counts.message_count, 0),
                    COALESCE(LEFT(last_msg.content, 120), ''),
                    COALESCE(last_msg.role, ''),
                    COALESCE(last_msg.message_type, ''),
                    COALESCE(dc.summary, '')
                FROM chat_sessions AS s
                LEFT JOIN chat_diagnosis_contexts AS dc
                    ON dc.session_id = s.session_id
                LEFT JOIN LATERAL (
                    SELECT COUNT(*)::INT AS message_count
                    FROM chat_messages
                    WHERE session_id = s.session_id
                ) AS msg_counts ON TRUE
                LEFT JOIN LATERAL (
                    SELECT content, role, message_type
                    FROM chat_messages
                    WHERE session_id = s.session_id
                      AND BTRIM(COALESCE(content, '')) <> ''
                    ORDER BY created_at DESC, message_id DESC
                    LIMIT 1
                ) AS last_msg ON TRUE
                WHERE s.session_id = %s
                """,
                (session_id,),
            )
            row = cursor.fetchone()
        if row is None:
            msg = f"Session does not exist: {session_id}"
            raise ValueError(msg)

        diagnosis_summary = str(row[9] or "")
        return {
            "session_id": str(row[0]),
            "created_at": _isoformat(row[1]),
            "updated_at": _isoformat(row[2]),
            "title": str(row[3] or ""),
            "pinned": bool(row[4]),
            "message_count": int(row[5] or 0),
            "last_message_preview": str(row[6] or ""),
            "last_message_role": str(row[7] or ""),
            "last_message_type": str(row[8] or ""),
            "has_diagnosis_context": bool(diagnosis_summary),
            "diagnosis_summary": diagnosis_summary,
        }

    def list_sessions(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent sessions ordered by `updated_at` descending."""
        normalized_limit = max(1, int(limit))
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT session_id
                    FROM chat_sessions
                    ORDER BY pinned DESC, updated_at DESC, session_id DESC
                    LIMIT %s
                    """,
                    (normalized_limit,),
                )
                session_ids = [str(row[0]) for row in cursor.fetchall()]
            return [self._session_summary(conn, session_id) for session_id in session_ids]

    def update_session_metadata(
        self,
        session_id: str,
        *,
        title: str | None = None,
        pinned: bool | None = None,
    ) -> dict[str, Any]:
        """Update user-editable session metadata."""
        with self._connect() as conn:
            self._ensure_session_row(conn, session_id)
            with conn.cursor() as cursor:
                if title is not None:
                    cursor.execute(
                        """
                        UPDATE chat_sessions
                        SET title = %s,
                            updated_at = NOW()
                        WHERE session_id = %s
                        """,
                        (str(title).strip(), session_id),
                    )
                if pinned is not None:
                    cursor.execute(
                        """
                        UPDATE chat_sessions
                        SET pinned = %s,
                            updated_at = NOW()
                        WHERE session_id = %s
                        """,
                        (bool(pinned), session_id),
                    )
            return self._build_session(conn, session_id) or {}

    def clear_diagnosis_context(self, session_id: str) -> dict[str, Any]:
        """Remove skin assistance context and its derived messages from a session."""
        with self._connect() as conn:
            self._ensure_session_row(conn, session_id)
            with conn.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM chat_diagnosis_contexts WHERE session_id = %s",
                    (session_id,),
                )
                cursor.execute(
                    """
                    DELETE FROM chat_messages
                    WHERE session_id = %s
                      AND message_type IN ('diagnosis-summary', 'diagnosis-report')
                    """,
                    (session_id,),
                )
            self._touch_session(conn, session_id)
            return self._build_session(conn, session_id) or {}

    def delete_session(self, session_id: str) -> bool:
        """Delete one stored session."""
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM chat_sessions WHERE session_id = %s",
                    (session_id,),
                )
                return bool(cursor.rowcount)

    def append_messages(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Append messages to a session."""
        with self._connect() as conn:
            self._ensure_session_row(conn, session_id)
            with conn.cursor() as cursor:
                for message in messages:
                    normalized_message = dict(message)
                    normalized_message["created_at"] = _message_timestamp(normalized_message)
                    cursor.execute(
                        """
                        INSERT INTO chat_messages (
                            message_id,
                            session_id,
                            role,
                            content,
                            message_type,
                            attachments_json,
                            metadata_json,
                            created_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::timestamptz)
                        ON CONFLICT (message_id) DO UPDATE SET
                            role = EXCLUDED.role,
                            content = EXCLUDED.content,
                            message_type = EXCLUDED.message_type,
                            attachments_json = EXCLUDED.attachments_json,
                            metadata_json = EXCLUDED.metadata_json,
                            created_at = EXCLUDED.created_at
                        """,
                        (
                            str(normalized_message.get("message_id") or uuid.uuid4().hex),
                            session_id,
                            str(normalized_message.get("role") or ""),
                            str(normalized_message.get("content") or ""),
                            str(normalized_message.get("message_type") or "text"),
                            json.dumps(
                                list(normalized_message.get("attachments") or []),
                                ensure_ascii=False,
                            ),
                            json.dumps(
                                dict(normalized_message.get("metadata") or {}),
                                ensure_ascii=False,
                            ),
                            str(normalized_message["created_at"]),
                        ),
                    )
            self._touch_session(conn, session_id)
            return self._build_session(conn, session_id) or {}

    def add_or_replace_attachment(
        self,
        session_id: str,
        attachment: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Insert or replace an attachment for a session."""
        normalized_attachment = dict(attachment)
        attachment_id = str(normalized_attachment.get("attachment_id") or uuid.uuid4().hex)
        normalized_attachment["attachment_id"] = attachment_id
        normalized_attachment["created_at"] = str(
            normalized_attachment.get("created_at") or _utc_now()
        )
        metadata = dict(normalized_attachment.get("metadata") or {})

        with self._connect() as conn:
            self._ensure_session_row(conn, session_id)
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO chat_attachments (
                        attachment_id,
                        session_id,
                        name,
                        content_type,
                        kind,
                        size_bytes,
                        stored_path,
                        object_key,
                        download_url,
                        text_excerpt,
                        extraction_error,
                        metadata_json,
                        created_at
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::timestamptz
                    )
                    ON CONFLICT (attachment_id) DO UPDATE SET
                        session_id = EXCLUDED.session_id,
                        name = EXCLUDED.name,
                        content_type = EXCLUDED.content_type,
                        kind = EXCLUDED.kind,
                        size_bytes = EXCLUDED.size_bytes,
                        stored_path = EXCLUDED.stored_path,
                        object_key = EXCLUDED.object_key,
                        download_url = EXCLUDED.download_url,
                        text_excerpt = EXCLUDED.text_excerpt,
                        extraction_error = EXCLUDED.extraction_error,
                        metadata_json = EXCLUDED.metadata_json,
                        created_at = EXCLUDED.created_at
                    """,
                    (
                        attachment_id,
                        session_id,
                        str(normalized_attachment.get("name") or ""),
                        str(normalized_attachment.get("content_type") or ""),
                        str(normalized_attachment.get("kind") or ""),
                        int(normalized_attachment.get("size_bytes") or 0),
                        str(normalized_attachment.get("stored_path") or ""),
                        str(normalized_attachment.get("object_key") or ""),
                        str(normalized_attachment.get("download_url") or ""),
                        str(normalized_attachment.get("text_excerpt") or ""),
                        str(normalized_attachment.get("extraction_error") or ""),
                        json.dumps(metadata, ensure_ascii=False),
                        str(normalized_attachment["created_at"]),
                    ),
                )
            self._touch_session(conn, session_id)
            updated_session = self._build_session(conn, session_id) or {}
            attachment_by_id = {
                str(item.get("attachment_id") or ""): item
                for item in updated_session.get("attachments") or []
            }
            return updated_session, dict(attachment_by_id.get(attachment_id) or normalized_attachment)

    def resolve_attachments(
        self,
        session_id: str,
        attachment_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Resolve attachment metadata by identifier."""
        normalized_ids = [str(item).strip() for item in attachment_ids if str(item).strip()]
        if not normalized_ids:
            return []

        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        attachment_id,
                        name,
                        content_type,
                        kind,
                        size_bytes,
                        stored_path,
                        object_key,
                        download_url,
                        text_excerpt,
                        extraction_error,
                        metadata_json::text,
                        created_at
                    FROM chat_attachments
                    WHERE session_id = %s
                      AND attachment_id = ANY(%s)
                    """,
                    (session_id, normalized_ids),
                )
                rows = cursor.fetchall()

        by_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            by_id[str(row[0])] = {
                "attachment_id": str(row[0]),
                "session_id": session_id,
                "name": str(row[1] or ""),
                "content_type": str(row[2] or ""),
                "kind": str(row[3] or ""),
                "size_bytes": int(row[4] or 0),
                "stored_path": str(row[5] or ""),
                "object_key": str(row[6] or ""),
                "download_url": str(row[7] or ""),
                "text_excerpt": str(row[8] or ""),
                "extraction_error": str(row[9] or ""),
                "metadata": dict(json.loads(row[10] or "{}")),
                "created_at": _isoformat(row[11]),
            }
        return [by_id[attachment_id] for attachment_id in normalized_ids if attachment_id in by_id]

    def set_diagnosis_context(
        self,
        session_id: str,
        diagnosis_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Update the skin assistance context for a session."""
        payload = dict(diagnosis_context)
        with self._connect() as conn:
            self._ensure_session_row(conn, session_id)
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO chat_diagnosis_contexts (
                        session_id,
                        source_folder,
                        total_score,
                        summary,
                        payload_json,
                        updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s::jsonb, NOW())
                    ON CONFLICT (session_id) DO UPDATE SET
                        source_folder = EXCLUDED.source_folder,
                        total_score = EXCLUDED.total_score,
                        summary = EXCLUDED.summary,
                        payload_json = EXCLUDED.payload_json,
                        updated_at = NOW()
                    """,
                    (
                        session_id,
                        str(payload.get("sourceFolder") or ""),
                        payload.get("totalScore"),
                        str(payload.get("summary") or ""),
                        json.dumps(payload, ensure_ascii=False),
                    ),
                )
            self._touch_session(conn, session_id)
            return self._build_session(conn, session_id) or {}

    def find_turn_messages(
        self,
        session_id: str,
        client_message_id: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Return the stored user/assistant pair for one client turn id."""
        normalized_client_message_id = str(client_message_id or "").strip()
        if not normalized_client_message_id:
            return None, None

        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        message_id,
                        role,
                        content,
                        message_type,
                        attachments_json::text,
                        metadata_json::text,
                        created_at
                    FROM chat_messages
                    WHERE session_id = %s
                      AND metadata_json ->> 'client_message_id' = %s
                    ORDER BY created_at ASC, message_id ASC
                    """,
                    (session_id, normalized_client_message_id),
                )
                rows = cursor.fetchall()

        user_message: dict[str, Any] | None = None
        assistant_message: dict[str, Any] | None = None
        for row in rows:
            message = {
                "message_id": str(row[0]),
                "role": str(row[1] or ""),
                "content": str(row[2] or ""),
                "message_type": str(row[3] or "text"),
                "attachments": list(json.loads(row[4] or "[]")),
                "metadata": dict(json.loads(row[5] or "{}")),
                "created_at": _isoformat(row[6]),
            }
            if message["role"] == "user" and user_message is None:
                user_message = message
            elif message["role"] == "assistant" and assistant_message is None:
                assistant_message = message
        return user_message, assistant_message

    def add_or_replace_diagnosis_message(
        self,
        session_id: str,
        diagnosis_message: dict[str, Any],
    ) -> dict[str, Any]:
        """Add or replace the skin assistance summary message in a session."""
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
        """Add or replace the skin assistance report message in a session."""
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
        """Add or replace a context-derived message using source-folder deduping."""
        source_folder = (
            (context_message.get("metadata") or {}).get("sourceFolder")
            if isinstance(context_message.get("metadata") or {}, dict)
            else None
        )
        metadata = (
            context_message.get("metadata") or {}
            if isinstance(context_message.get("metadata") or {}, dict)
            else {}
        )
        source_type = str(metadata.get("sourceType") or "").strip()
        summary = str(metadata.get("summary") or context_message.get("content") or "").strip()
        with self._connect() as conn:
            self._ensure_session_row(conn, session_id)
            with conn.cursor() as cursor:
                if source_type == "combined":
                    cursor.execute(
                        """
                        DELETE FROM chat_messages
                        WHERE session_id = %s
                          AND message_type = %s
                          AND metadata_json ->> 'sourceType' = 'combined'
                        """,
                        (session_id, message_type),
                    )
                elif source_folder:
                    cursor.execute(
                        """
                        DELETE FROM chat_messages
                        WHERE session_id = %s
                          AND message_type = %s
                          AND metadata_json ->> 'sourceFolder' = %s
                        """,
                        (session_id, message_type, str(source_folder)),
                    )
                elif source_type and summary:
                    cursor.execute(
                        """
                        DELETE FROM chat_messages
                        WHERE session_id = %s
                          AND message_type = %s
                          AND metadata_json ->> 'sourceType' = %s
                          AND COALESCE(metadata_json ->> 'summary', content) = %s
                        """,
                        (session_id, message_type, source_type, summary),
                    )
                normalized_message = dict(context_message)
                normalized_message["created_at"] = _message_timestamp(normalized_message)
                cursor.execute(
                    """
                    INSERT INTO chat_messages (
                        message_id,
                        session_id,
                        role,
                        content,
                        message_type,
                        attachments_json,
                        metadata_json,
                        created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::timestamptz)
                    ON CONFLICT (message_id) DO UPDATE SET
                        role = EXCLUDED.role,
                        content = EXCLUDED.content,
                        message_type = EXCLUDED.message_type,
                        attachments_json = EXCLUDED.attachments_json,
                        metadata_json = EXCLUDED.metadata_json,
                        created_at = EXCLUDED.created_at
                    """,
                    (
                        str(normalized_message.get("message_id") or uuid.uuid4().hex),
                        session_id,
                        str(normalized_message.get("role") or ""),
                        str(normalized_message.get("content") or ""),
                        str(normalized_message.get("message_type") or message_type),
                        json.dumps(
                            list(normalized_message.get("attachments") or []),
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            dict(normalized_message.get("metadata") or {}),
                            ensure_ascii=False,
                        ),
                        str(normalized_message["created_at"]),
                    ),
                )
            self._touch_session(conn, session_id)
            return self._build_session(conn, session_id) or {}

    def record_command_event(
        self,
        session_id: str,
        *,
        event_type: str,
        command_name: str = "",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record one command or operation event for a session."""
        with self._connect() as conn:
            self._ensure_session_row(conn, session_id)
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO chat_command_events (
                        event_id,
                        session_id,
                        event_type,
                        command_name,
                        payload_json,
                        created_at
                    )
                    VALUES (%s, %s, %s, %s, %s::jsonb, NOW())
                    """,
                    (
                        uuid.uuid4().hex,
                        session_id,
                        str(event_type or "").strip(),
                        str(command_name or "").strip(),
                        json.dumps(dict(payload or {}), ensure_ascii=False),
                    ),
                )
            return self._build_session(conn, session_id) or {}


__all__ = ["PostgresChatSessionStore"]
