"""HTTP endpoints for queued face and tongue analysis."""

from __future__ import annotations

from typing import Any

from flask import request, url_for

from api.controllers.v1 import bp
from api.queue.celery_app import celery_app
from api.tasks.analysis_tasks import run_face_analysis, run_face_workflow, run_tongue_analysis

try:
    from celery.result import AsyncResult
except ModuleNotFoundError:  # pragma: no cover - exercised only in incomplete envs
    AsyncResult = None  # type: ignore[assignment]


def _require_celery() -> tuple[dict[str, str], int] | None:
    """Return an error response when Celery is not installed/configured."""

    if celery_app is None or AsyncResult is None:
        return {
            "error": "Celery is not available",
            "message": "Install celery[redis] in the active environment and configure REDIS_URL.",
        }, 503
    return None


def _decode_task_payload() -> dict[str, Any] | tuple[dict[str, str], int]:
    """Validate a queued analysis payload."""

    if not request.is_json:
        return {"error": "request body must be JSON"}, 400

    payload = request.get_json(silent=True) or {}
    file_path = str(payload.get("file_path", "")).strip()
    file_name = str(payload.get("file_name", "")).strip()
    if not file_path or not file_name:
        return {"error": "file_path and file_name are required"}, 400

    normalized = dict(payload)
    normalized["file_path"] = file_path
    normalized["file_name"] = file_name
    return normalized


def _build_submit_response(async_result: Any, *, analysis_type: str) -> tuple[dict[str, Any], int]:
    """Build the queue submission response."""

    task_id = async_result.id
    return {
        "status": "queued",
        "analysis_type": analysis_type,
        "task_id": task_id,
        "state": async_result.state,
        "status_url": url_for("v1.analysis_task_status", task_id=task_id, _external=True),
    }, 202


def _submit_task(task, *, payload: dict[str, Any], analysis_type: str) -> tuple[dict[str, Any], int]:
    """Submit one Celery task and return a consistent response."""

    celery_error = _require_celery()
    if celery_error is not None:
        return celery_error
    if task is None:
        return {"error": f"{analysis_type} task is not registered"}, 503
    return _build_submit_response(task.delay(payload), analysis_type=analysis_type)


@bp.route("/analysis-tasks/face", methods=["POST"])
def submit_face_workflow_task() -> tuple[dict[str, Any], int]:
    """Queue the full face workflow: alignment plus face analysis."""

    payload_result = _decode_task_payload()
    if isinstance(payload_result, tuple):
        return payload_result
    return _submit_task(run_face_workflow, payload=payload_result, analysis_type="face")


@bp.route("/analysis-tasks/face-analysis", methods=["POST"])
def submit_face_analysis_task() -> tuple[dict[str, Any], int]:
    """Queue face analysis for an already uploaded or aligned image."""

    payload_result = _decode_task_payload()
    if isinstance(payload_result, tuple):
        return payload_result
    return _submit_task(run_face_analysis, payload=payload_result, analysis_type="face_analysis")


@bp.route("/analysis-tasks/tongue", methods=["POST"])
def submit_tongue_analysis_task() -> tuple[dict[str, Any], int]:
    """Queue complete phase-1 tongue analysis."""

    payload_result = _decode_task_payload()
    if isinstance(payload_result, tuple):
        return payload_result
    return _submit_task(run_tongue_analysis, payload=payload_result, analysis_type="tongue")


@bp.route("/analysis-tasks/<task_id>", methods=["GET"])
def analysis_task_status(task_id: str) -> tuple[dict[str, Any], int]:
    """Return the current Celery status/result for one queued analysis task."""

    celery_error = _require_celery()
    if celery_error is not None:
        return celery_error

    result = AsyncResult(task_id, app=celery_app)
    payload: dict[str, Any] = {
        "task_id": task_id,
        "state": result.state,
        "ready": result.ready(),
        "successful": result.successful(),
    }

    if result.state == "SUCCESS":
        payload["result"] = result.result
    elif result.state == "FAILURE":
        payload["error"] = str(result.info)
        if result.traceback:
            payload["traceback"] = result.traceback
    elif isinstance(result.info, dict):
        payload["meta"] = result.info

    return payload, 200


__all__ = [
    "analysis_task_status",
    "submit_face_analysis_task",
    "submit_face_workflow_task",
    "submit_tongue_analysis_task",
]
