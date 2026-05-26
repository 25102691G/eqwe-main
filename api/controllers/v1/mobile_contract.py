"""Helpers for the mobile-facing API contract."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

from flask import url_for

RESULT_IMAGE_PRIORITY: tuple[str, ...] = (
    "align.jpg",
    "oil_mask_fixed.jpg",
    "moisture_analysis.jpg",
    "hyperpigmentation.jpg",
    "imgnasolabial.jpg",
    "black_eye.jpg",
    "overlay_on_white.jpg",
    "acne.jpg",
    "blackheads.jpg",
    "pores.jpg",
    "landmarks.jpg",
)

_DISPLAY_NAME_BY_FILENAME: dict[str, str] = {
    "align.jpg": "Aligned Face",
    "oil_mask_fixed.jpg": "Oil Analysis",
    "moisture_analysis.jpg": "Moisture Analysis",
    "hyperpigmentation.jpg": "Hyperpigmentation",
    "imgnasolabial.jpg": "Wrinkles",
    "black_eye.jpg": "Dark Circles",
    "overlay_on_white.jpg": "Sensitivity Overlay",
    "acne.jpg": "Acne",
    "blackheads.jpg": "Blackheads",
    "pores.jpg": "Pores",
    "landmarks.jpg": "Landmarks",
}
_BROWSER_PATH_PATTERN = re.compile(r"/browser/([^/]+)/(.+)$")
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_browser_file_path(file_path: str) -> tuple[str, str]:
    """Extract bucket and folder path from a MinIO browser URL."""
    match = _BROWSER_PATH_PATTERN.search(file_path.strip())
    if match is None:
        msg = "invalid MinIO browser path"
        raise ValueError(msg)
    return match.group(1), match.group(2).strip("/")


def build_browser_file_path(bucket_name: str, folder_path: str, endpoint_url: str) -> str:
    """Build the browser-style MinIO path expected by the legacy endpoints."""
    parsed = urlparse(endpoint_url.strip() or "http://127.0.0.1:9000")
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "127.0.0.1"
    console_port = 9001 if parsed.port in (None, 9000) else parsed.port
    normalized_folder = folder_path.strip("/")
    return f"{scheme}://{host}:{console_port}/browser/{bucket_name}/{normalized_folder}"


def build_object_key(folder_path: str, filename: str) -> str:
    """Build a storage object key from the folder and file name."""
    return f"{folder_path.strip('/')}/{filename}"


def build_mobile_asset_url(folder_path: str, filename: str) -> str:
    """Build the Flask proxy URL for a mobile-facing asset."""
    return url_for(
        "v1.mobile_result_image",
        folder_path=folder_path.strip("/"),
        filename=filename,
        _external=True,
    )


def is_image_filename(filename: str) -> bool:
    """Return whether the file should be rendered in the image gallery."""
    return os.path.splitext(filename.lower())[1] in _IMAGE_EXTENSIONS


def humanize_result_filename(filename: str) -> str:
    """Convert a result file name into a short UI label."""
    if filename in _DISPLAY_NAME_BY_FILENAME:
        return _DISPLAY_NAME_BY_FILENAME[filename]
    name, _ = os.path.splitext(filename)
    return name.replace("_", " ").title()


def ordered_result_filenames(
    uploaded_files: list[dict[str, Any]],
    *,
    preferred: Iterable[str] = (),
) -> list[str]:
    """Return gallery filenames in the preferred UX order."""
    filenames = {
        item.get("filename", "")
        for item in uploaded_files
        if isinstance(item, dict) and item.get("filename")
    }
    ordered: list[str] = []

    for filename in preferred:
        if filename and filename not in ordered:
            ordered.append(filename)
            filenames.add(filename)

    for filename in RESULT_IMAGE_PRIORITY:
        if filename in filenames and filename not in ordered:
            ordered.append(filename)

    remainder = sorted(filename for filename in filenames if filename and filename not in ordered)
    ordered.extend(remainder)
    return ordered


def build_result_images(
    folder_path: str,
    uploaded_files: list[dict[str, Any]],
    *,
    preferred: Iterable[str] = (),
) -> list[dict[str, str]]:
    """Build gallery metadata for mobile clients."""
    images: list[dict[str, str]] = []
    for filename in ordered_result_filenames(uploaded_files, preferred=preferred):
        if not is_image_filename(filename):
            continue
        images.append(
            {
                "filename": filename,
                "label": humanize_result_filename(filename),
                "object_key": build_object_key(folder_path, filename),
                "url": build_mobile_asset_url(folder_path, filename),
            }
        )
    return images


def build_analysis_response(
    *,
    task_uuid: str,
    file_path: str,
    input_file_name: str,
    source_object_key: str,
    bucket_name: str,
    folder_path: str,
    analysis_results: dict[str, Any],
    metadata: dict[str, Any],
    uploaded_files: list[dict[str, Any]],
    report_object_key: str,
    llm_report: dict[str, Any] | None = None,
    local_files: dict[str, str] | None = None,
    message: str = "analysis complete",
) -> dict[str, Any]:
    """Build the mobile-facing analysis response payload."""
    report_filename = os.path.basename(report_object_key)
    image_info = dict(metadata.get("image_info", {}))
    image_info["file_name"] = input_file_name

    normalized_metadata = dict(metadata)
    normalized_metadata["image_info"] = image_info
    normalized_metadata.update(
        {
            "bucket": bucket_name,
            "folder": folder_path,
            "input_image_url": build_mobile_asset_url(folder_path, input_file_name),
            "analysis_report_url": build_mobile_asset_url(folder_path, report_filename),
        }
    )

    report_url = build_mobile_asset_url(folder_path, report_filename)
    return {
        "status": "success",
        "uuid": task_uuid,
        "message": message,
        "analysis_results": analysis_results,
        "metadata": normalized_metadata,
        "file_info": {
            "path": file_path,
            "name": input_file_name,
            "object_key": source_object_key,
        },
        "uploaded_files": uploaded_files,
        "storage": {
            "type": "cloud",
            "bucket": bucket_name,
            "folder": folder_path,
            "source_object_key": source_object_key,
            "analysis_results_object_key": report_object_key,
        },
        "source_image_url": build_mobile_asset_url(folder_path, input_file_name),
        "analysis_report_url": report_url,
        "llm_report": llm_report or {},
        "result_images": build_result_images(
            folder_path,
            uploaded_files,
            preferred=("align.jpg",),
        ),
        "local_files": local_files or {},
    }
