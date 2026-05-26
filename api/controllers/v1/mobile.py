"""Mobile-friendly bridge endpoints for the WeChat Mini Program."""

import mimetypes
import os
import re
import uuid
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from flask import jsonify, request, send_file, url_for
from werkzeug.utils import secure_filename

from api.configs.storage_config import STORAGE_CONFIG
from api.controllers.v1 import bp
from api.controllers.v1.mobile_contract import build_browser_file_path
from api.middleware.storage.cloud_storage_service import CloudStorageService

load_dotenv()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
UPLOAD_BASE_DIR = os.path.join(project_root, "upload_files")
Path(UPLOAD_BASE_DIR).mkdir(exist_ok=True)

cloud_storage = CloudStorageService(STORAGE_CONFIG)
SAVE_LOCAL_IMAGES = os.getenv("SAVE_LOCAL_IMAGES", "false").lower() == "true"


def _sanitize_folder_name(folder_name):
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", folder_name or "").strip("-")
    if cleaned:
        return cleaned
    return f"miniapp-{uuid.uuid4().hex}"


def _pick_upload_filename(file_storage):
    original_name = secure_filename(file_storage.filename or "")
    _, extension = os.path.splitext(original_name)

    if not extension:
        guessed_extension = mimetypes.guess_extension(file_storage.mimetype or "")
        extension = guessed_extension or ".jpg"

    return f"original{extension.lower()}"


def _guess_content_type(filename):
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


def _resolve_local_file(folder_path, filename):
    candidate = os.path.abspath(os.path.join(UPLOAD_BASE_DIR, folder_path, filename))
    base_dir = os.path.abspath(UPLOAD_BASE_DIR)

    if candidate == base_dir or not candidate.startswith(base_dir + os.sep):
        return None

    if os.path.exists(candidate):
        return candidate

    return None


@bp.route("/mobile/upload-image", methods=["POST"])
def mobile_upload_image():
    """Accept a phone image upload and store it in MinIO for downstream analysis."""
    if "file" not in request.files:
        return {
            "error": "missing file",
            "message": "Expected multipart/form-data with a file field named `file`.",
        }, 400

    file_storage = request.files["file"]
    if not file_storage or not file_storage.filename:
        return {
            "error": "empty file",
            "message": "The uploaded file is empty or has no filename.",
        }, 400

    folder_path = _sanitize_folder_name(request.form.get("folder", ""))
    file_name = _pick_upload_filename(file_storage)
    bucket_name = STORAGE_CONFIG["bucket_name"]
    object_key = f"{folder_path}/{file_name}"

    file_bytes = file_storage.read()
    if not file_bytes:
        return {
            "error": "empty file",
            "message": "The uploaded file has no content.",
        }, 400

    upload_success, upload_result = cloud_storage.upload_from_memory(
        file_bytes,
        object_key,
        content_type=file_storage.mimetype or _guess_content_type(file_name),
    )
    if not upload_success:
        return {
            "error": "upload failed",
            "message": str(upload_result),
        }, 500

    if SAVE_LOCAL_IMAGES:
        task_dir = os.path.join(UPLOAD_BASE_DIR, folder_path)
        Path(task_dir).mkdir(parents=True, exist_ok=True)
        local_path = os.path.join(task_dir, file_name)
        with open(local_path, "wb") as file_handle:
            file_handle.write(file_bytes)

    preview_url = url_for(
        "v1.mobile_result_image",
        folder_path=folder_path,
        filename=file_name,
        _external=True,
    )

    return jsonify(
        {
            "status": "success",
            "bucket": bucket_name,
            "folder": folder_path,
            "file_name": file_name,
            "object_key": object_key,
            "file_path": build_browser_file_path(
                bucket_name,
                folder_path,
                STORAGE_CONFIG.get("endpoint_url", ""),
            ),
            "preview_url": preview_url,
        }
    )


@bp.route("/mobile/result-image/<path:folder_path>/<filename>", methods=["GET"])
def mobile_result_image(folder_path, filename):
    """Proxy result assets from MinIO so the mini program only talks to the Flask server."""
    object_key = f"{folder_path}/{filename}"
    content_type = _guess_content_type(filename)

    download_success, payload = cloud_storage.download_to_memory(object_key)
    if download_success:
        return send_file(
            BytesIO(payload),
            mimetype=content_type,
            download_name=filename,
        )

    local_file = _resolve_local_file(folder_path, filename)
    if local_file is not None:
        return send_file(local_file, mimetype=content_type, download_name=filename)

    return {
        "error": "file not found",
        "object_key": object_key,
        "message": str(payload),
    }, 404
