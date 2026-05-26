"""Flask entry point for the skin analysis service."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS


def _configure_console_encoding() -> None:
    """Use UTF-8 for logs on Windows consoles when possible."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


_configure_console_encoding()
load_dotenv()

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from api.queue.celery_app import celery_app as celery

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = os.path.join(project_root, "upload_files")
Path(app.config["UPLOAD_FOLDER"]).mkdir(exist_ok=True)

print("Preloading models into memory...")
from api.services.model_manager import get_model_manager

model_manager = get_model_manager()
print(f"Loaded models: {', '.join(model_manager.list_models())}")

from api.controllers.v1 import bp as v1_bp
from api.controllers.v1 import register_routes

register_routes()

app.register_blueprint(v1_bp)


if __name__ == "__main__":
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "5000"))

    print("Starting skin analysis service...")
    print(f"Project root: {project_root}")
    print(f"Listening on: http://{host}:{port}")
    print("Endpoints:")
    print("  GET /v1/ - API info")
    print("  POST /v1/mobile/upload-image - mobile upload bridge")
    print("  GET /v1/mobile/result-image/<folder>/<filename> - mobile asset proxy")
    print("  POST /v1/mobile/chat/session - create or restore mobile chat session")
    print("  POST /v1/mobile/chat/attachment - upload mobile chat attachment")
    print("  POST /v1/mobile/chat/message - non-stream mobile chat reply")
    print("  POST /v1/mobile/chat/stream - stream mobile chat reply")
    print("  POST /v1/mobile/chat/diagnosis-context - save skin assistance summary into chat")
    print("  POST /v1/face-align - face alignment")
    print("  POST /v1/analyze-face - face analysis")
    print("  POST /v1/tongue-quality-check - tongue image quality gate")
    print("  POST /v1/tongue-segment - tongue phase-1 analysis")
    print("  POST /v1/analysis-tasks/face - queue face alignment plus analysis")
    print("  POST /v1/analysis-tasks/face-analysis - queue face analysis only")
    print("  POST /v1/analysis-tasks/tongue - queue tongue phase-1 analysis")
    print("  GET /v1/analysis-tasks/<task_id> - queued analysis task status")

    app.run(debug=True, host=host, port=port)
