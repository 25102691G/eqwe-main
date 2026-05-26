from flask import jsonify

from api.controllers.v1 import bp


@bp.route("/", methods=["GET"])
def index_api():
    return jsonify(
        {
            "welcome": "SKIN_DET OpenAPI",
            "api_version": "V1",
            "endpoints": [
                "GET /v1/ - API info",
                "POST /v1/mobile/upload-image - mobile upload bridge",
                "GET /v1/mobile/result-image/<folder>/<filename> - mobile asset proxy",
                "POST /v1/mobile/chat/session - create or restore mobile chat session",
                "GET /v1/mobile/chat/session/<session_id> - get mobile chat session",
                "PATCH /v1/mobile/chat/session/<session_id> - update mobile chat session metadata",
                "DELETE /v1/mobile/chat/session/<session_id> - delete one mobile chat session",
                "GET /v1/mobile/chat/sessions - list recent mobile chat sessions",
                "POST /v1/mobile/chat/attachment - upload mobile chat attachment",
                "GET /v1/mobile/chat/attachment/<session_id>/<attachment_id> - get mobile chat attachment",
                "POST /v1/mobile/chat/message - non-stream mobile chat reply",
                "POST /v1/mobile/chat/stream - stream mobile chat reply",
                "POST /v1/mobile/chat/diagnosis-context - save skin assistance context into chat",
                "DELETE /v1/mobile/chat/diagnosis-context/<session_id> - clear skin assistance context from chat",
                "POST /v1/face-align - face alignment",
                "POST /v1/analyze-face - face analysis",
                "POST /v1/tongue-quality-check - tongue image quality gate",
                "POST /v1/tongue-segment - tongue phase-1 analysis",
                "POST /v1/analysis-tasks/face - queue face alignment plus analysis",
                "POST /v1/analysis-tasks/face-analysis - queue face analysis only",
                "POST /v1/analysis-tasks/tongue - queue tongue phase-1 analysis",
                "GET /v1/analysis-tasks/<task_id> - queued analysis task status",
                "GET /v1/face-align/<uuid>/<filename> - legacy aligned file route",
            ],
        }
    )
