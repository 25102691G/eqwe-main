"""Factory helpers for selecting the mobile chat session store backend."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from api.services.chat_session_store import ChatSessionStore


def build_chat_session_store(root_dir: str | Path) -> Any:
    """Build the configured chat session store implementation.

    Args:
        root_dir: Local JSON fallback directory.

    Returns:
        The configured chat session store instance.
    """
    backend = str(os.getenv("CHAT_STORAGE_BACKEND", "auto")).strip().lower()
    strict_postgres = str(os.getenv("CHAT_STORAGE_STRICT_POSTGRES", "false")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    database_url = str(
        os.getenv("CHAT_DATABASE_URL") or os.getenv("DATABASE_URL") or ""
    ).strip()

    if backend in {"postgres", "postgresql"} or (backend == "auto" and database_url):
        from api.services.postgres_chat_session_store import PostgresChatSessionStore

        try:
            return PostgresChatSessionStore(database_url)
        except Exception as exc:  # noqa: BLE001
            if strict_postgres:
                raise
            print(f"PostgreSQL chat store unavailable, falling back to local JSON store: {exc}")

    return ChatSessionStore(root_dir)


__all__ = ["build_chat_session_store"]
