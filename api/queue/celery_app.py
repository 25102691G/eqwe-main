"""Celery application factory for asynchronous analysis tasks."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

try:
    from celery import Celery
except ModuleNotFoundError:  # pragma: no cover - exercised only in incomplete envs
    Celery = None  # type: ignore[assignment]


def _first_env(*names: str, default: str) -> str:
    """Return the first non-empty environment variable value."""

    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _create_celery_app():
    """Create the shared Celery app, or None when Celery is not installed."""

    if Celery is None:
        return None

    broker_url = _first_env(
        "CELERY_BROKER_URL",
        "REDIS_URL",
        default="redis://:chatredis@127.0.0.1:16379/0",
    )
    result_backend = _first_env(
        "CELERY_RESULT_BACKEND",
        "REDIS_URL",
        default=broker_url,
    )

    app = Celery(
        "skin_alporithm",
        broker=broker_url,
        backend=result_backend,
        include=["api.tasks.analysis_tasks"],
    )
    app.conf.update(
        task_track_started=True,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone=os.getenv("TZ", "Asia/Shanghai"),
        enable_utc=False,
        broker_connection_retry_on_startup=True,
        task_routes={
            "api.tasks.face_workflow": {"queue": "face_analysis"},
            "api.tasks.face_analysis": {"queue": "face_analysis"},
            "api.tasks.tongue_analysis": {"queue": "tongue_analysis"},
        },
    )
    return app


celery_app = _create_celery_app()

__all__ = ["celery_app"]
