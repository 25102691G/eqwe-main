"""Celery tasks for face and tongue analysis."""

from __future__ import annotations

import json
from typing import Any

from flask import Response
from flask import Flask

from api.queue.celery_app import celery_app


class AnalysisTaskError(RuntimeError):
    """Raised when one queued analysis endpoint returns an error response."""


_task_flask_app: Flask | None = None


def _get_task_flask_app() -> Flask:
    """Build a minimal Flask app used only to provide request/url contexts."""

    global _task_flask_app
    if _task_flask_app is not None:
        return _task_flask_app

    app = Flask("analysis_task_runner")

    def mobile_result_image(folder_path: str, filename: str) -> str:
        return f"{folder_path}/{filename}"

    app.add_url_rule(
        "/v1/mobile/result-image/<path:folder_path>/<filename>",
        endpoint="v1.mobile_result_image",
        view_func=mobile_result_image,
    )
    _task_flask_app = app
    return app


def _resolve_view(path: str):
    """Import and return the existing view function for one queued endpoint."""

    if path == "/v1/face-align":
        from api.controllers.v1.face_align import face_align_api

        return face_align_api
    if path == "/v1/analyze-face":
        from api.controllers.v1.infer import analyze_face

        return analyze_face
    if path == "/v1/tongue-segment":
        from api.controllers.v1.tongue_segment import tongue_segment

        return tongue_segment

    msg = f"unsupported queued endpoint: {path}"
    raise ValueError(msg)


def _normalize_view_result(result: Any) -> tuple[dict[str, Any], int]:
    """Normalize Flask view return values into JSON body and status code."""

    status_code = 200
    response_body = result

    if isinstance(result, tuple):
        response_body = result[0]
        if len(result) > 1 and isinstance(result[1], int):
            status_code = result[1]

    if isinstance(response_body, Response):
        status_code = response_body.status_code
        body = response_body.get_json(silent=True)
        if body is None:
            body = {"raw_response": response_body.get_data(as_text=True)}
        return body, status_code

    if isinstance(response_body, dict):
        return response_body, status_code

    return {"raw_response": str(response_body)}, status_code


def _post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Call one existing Flask endpoint in-process and return its JSON body."""

    app = _get_task_flask_app()
    view_func = _resolve_view(path)
    with app.test_request_context(path, method="POST", json=payload):
        body, status_code = _normalize_view_result(view_func())

    if status_code >= 400:
        error_payload = {
            "endpoint": path,
            "http_status": status_code,
            "response": body,
        }
        raise AnalysisTaskError(json.dumps(error_payload, ensure_ascii=False))
    return body


if celery_app is not None:

    @celery_app.task(bind=True, name="api.tasks.face_workflow")
    def run_face_workflow(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run the current face workflow: align first, then analyze aligned image."""

        self.update_state(state="PROGRESS", meta={"stage": "face_align"})
        align_result = _post_json("/v1/face-align", payload)

        analyze_payload = {
            "file_path": payload["file_path"],
            "file_name": align_result.get("aligned_file_name") or "align.jpg",
        }
        self.update_state(state="PROGRESS", meta={"stage": "face_analysis"})
        analysis_result = _post_json("/v1/analyze-face", analyze_payload)

        return {
            "analysis_type": "face",
            "status": "success",
            "align_result": align_result,
            "analysis_result": analysis_result,
        }

    @celery_app.task(bind=True, name="api.tasks.face_analysis")
    def run_face_analysis(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run face analysis for an already uploaded/aligned image."""

        self.update_state(state="PROGRESS", meta={"stage": "face_analysis"})
        return {
            "analysis_type": "face_analysis",
            "status": "success",
            "analysis_result": _post_json("/v1/analyze-face", payload),
        }

    @celery_app.task(bind=True, name="api.tasks.tongue_analysis")
    def run_tongue_analysis(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run complete phase-1 tongue analysis."""

        self.update_state(state="PROGRESS", meta={"stage": "tongue_phase1"})
        return {
            "analysis_type": "tongue",
            "status": "success",
            "analysis_result": _post_json("/v1/tongue-segment", payload),
        }

else:  # pragma: no cover - only used when Celery is missing
    run_face_workflow = None
    run_face_analysis = None
    run_tongue_analysis = None


__all__ = [
    "AnalysisTaskError",
    "run_face_analysis",
    "run_face_workflow",
    "run_tongue_analysis",
]
