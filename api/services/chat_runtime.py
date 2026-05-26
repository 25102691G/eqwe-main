"""Redis-backed runtime helpers for mobile chat."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any


def _bool_env(name: str, default: bool) -> bool:
    """Parse one boolean environment flag."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class NoopChatRuntime:
    """Fallback runtime that disables Redis-enhanced behavior."""

    def get_recent_sessions(self, *, limit: int) -> list[dict[str, Any]] | None:
        return None

    def set_recent_sessions(self, *, limit: int, sessions: list[dict[str, Any]]) -> None:
        return None

    def invalidate_recent_sessions(self) -> None:
        return None

    def acquire_session_lock(self, session_id: str, *, ttl_seconds: int) -> str | None:
        return uuid.uuid4().hex

    def release_session_lock(self, session_id: str, owner_token: str | None) -> None:
        return None

    def allow_request(
        self,
        key: str,
        *,
        max_requests: int,
        window_seconds: int,
    ) -> bool:
        return True

    def set_stream_state(
        self,
        session_id: str,
        payload: dict[str, Any],
        *,
        ttl_seconds: int,
    ) -> None:
        return None

    def clear_stream_state(self, session_id: str) -> None:
        return None


class RedisChatRuntime:
    """Redis-backed runtime state for locks, caches, and rate limits."""

    def __init__(self, client: Any) -> None:
        """Initialize the runtime with a Redis client."""
        self.client = client

    def _recent_sessions_key(self, limit: int) -> str:
        return f"chat:recent_sessions:{max(1, int(limit))}"

    def _session_lock_key(self, session_id: str) -> str:
        return f"chat:session_lock:{session_id}"

    def _stream_state_key(self, session_id: str) -> str:
        return f"chat:stream_state:{session_id}"

    def _rate_limit_key(self, key: str) -> str:
        return f"chat:rate_limit:{key}"

    def get_recent_sessions(self, *, limit: int) -> list[dict[str, Any]] | None:
        cached = self.client.get(self._recent_sessions_key(limit))
        if not cached:
            return None
        return list(json.loads(cached))

    def set_recent_sessions(self, *, limit: int, sessions: list[dict[str, Any]]) -> None:
        ttl_seconds = int(os.getenv("CHAT_RECENT_SESSIONS_CACHE_TTL", "20"))
        self.client.set(
            self._recent_sessions_key(limit),
            json.dumps(list(sessions), ensure_ascii=False),
            ex=max(1, ttl_seconds),
        )

    def invalidate_recent_sessions(self) -> None:
        keys = list(self.client.scan_iter(match="chat:recent_sessions:*"))
        if keys:
            self.client.delete(*keys)

    def acquire_session_lock(self, session_id: str, *, ttl_seconds: int) -> str | None:
        owner_token = uuid.uuid4().hex
        acquired = self.client.set(
            self._session_lock_key(session_id),
            owner_token,
            nx=True,
            ex=max(1, ttl_seconds),
        )
        if not acquired:
            return None
        return owner_token

    def release_session_lock(self, session_id: str, owner_token: str | None) -> None:
        if not owner_token:
            return
        key = self._session_lock_key(session_id)
        lock_value = self.client.get(key)
        if lock_value and str(lock_value) == str(owner_token):
            self.client.delete(key)

    def allow_request(
        self,
        key: str,
        *,
        max_requests: int,
        window_seconds: int,
    ) -> bool:
        normalized_key = self._rate_limit_key(key)
        current_count = self.client.incr(normalized_key)
        if current_count == 1:
            self.client.expire(normalized_key, max(1, window_seconds))
        return int(current_count) <= max(1, max_requests)

    def set_stream_state(
        self,
        session_id: str,
        payload: dict[str, Any],
        *,
        ttl_seconds: int,
    ) -> None:
        self.client.set(
            self._stream_state_key(session_id),
            json.dumps(dict(payload), ensure_ascii=False),
            ex=max(1, ttl_seconds),
        )

    def clear_stream_state(self, session_id: str) -> None:
        self.client.delete(self._stream_state_key(session_id))


def build_chat_runtime() -> NoopChatRuntime | RedisChatRuntime:
    """Build the configured chat runtime helper."""
    if not _bool_env("CHAT_RUNTIME_USE_REDIS", True):
        return NoopChatRuntime()

    try:
        import redis
    except ImportError:
        return NoopChatRuntime()

    redis_url = str(os.getenv("REDIS_URL") or "").strip()
    try:
        if redis_url:
            client = redis.Redis.from_url(redis_url, decode_responses=True)
        else:
            client = redis.Redis(
                host=os.getenv("REDIS_HOST", "127.0.0.1"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                db=int(os.getenv("REDIS_DB", "0")),
                username=os.getenv("REDIS_USERNAME") or None,
                password=os.getenv("REDIS_PASSWORD") or None,
                ssl=_bool_env("REDIS_USE_SSL", False),
                decode_responses=True,
            )
        client.ping()
    except Exception:
        return NoopChatRuntime()

    return RedisChatRuntime(client)


__all__ = ["NoopChatRuntime", "RedisChatRuntime", "build_chat_runtime"]
