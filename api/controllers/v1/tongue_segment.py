"""Tongue phase-1 analysis and quality-control endpoints."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import request

from api.configs.storage_config import STORAGE_CONFIG
from api.controllers.v1 import bp
from api.controllers.v1.image_decode import decode_image_bytes_to_bgr
from api.controllers.v1.mobile_contract import build_object_key, parse_browser_file_path
from api.middleware.storage.cloud_storage_service import CloudStorageService
from api.services.tongue_phase1_service import DEFAULT_MODEL_PATH
from api.services.tongue_phase1_service import TonguePhase1Analysis
from api.services.tongue_phase1_service import TonguePhase1Service
from api.services.tongue_phase1_service import build_full_mask_image
from api.services.tongue_quality_control import TongueQualityControlService

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPLOAD_BASE_DIR = PROJECT_ROOT / "upload_files"
UPLOAD_BASE_DIR.mkdir(exist_ok=True)

TONGUE_MODEL_PATH = DEFAULT_MODEL_PATH
SAVE_LOCAL_IMAGES = os.getenv("SAVE_LOCAL_IMAGES", "false").lower() == "true"

cloud_storage = CloudStorageService(STORAGE_CONFIG)
quality_control_service = TongueQualityControlService()
tongue_phase1_service = TonguePhase1Service(model_path=TONGUE_MODEL_PATH, output_dir=UPLOAD_BASE_DIR)


@dataclass(frozen=True)
class TongueRequestPayload:
    """Validated tongue endpoint request payload."""

    file_path: str
    file_name: str
    include_images: bool
    include_visualizations: bool
    upload_visualizations: bool


@dataclass(frozen=True)
class DownloadedTongueImage:
    """Downloaded tongue image request context."""

    bucket_name: str
    folder_path: str
    object_key: str
    task_uuid: str
    file_name: str
    file_path: str
    image_bytes: bytes
    image: np.ndarray


@dataclass(frozen=True)
class TongueQualityArtifacts:
    """Minimal segmentation data needed by the quality-control service."""

    bounding_box: dict[str, int]
    full_mask_image: np.ndarray


def _parse_bool(value: Any, *, default: bool = False) -> bool:
    """Parse a JSON boolean flag while accepting common string forms."""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _decode_request_payload() -> TongueRequestPayload | tuple[dict[str, str], int]:
    """Validate the legacy JSON payload."""

    if not request.is_json:
        return {"error": "request body must be JSON"}, 400

    payload = request.get_json(silent=True) or {}
    file_path = str(payload.get("file_path", "")).strip()
    file_name = str(payload.get("file_name", "")).strip()
    if not file_path or not file_name:
        return {"error": "file_path and file_name are required"}, 400

    return TongueRequestPayload(
        file_path=file_path,
        file_name=file_name,
        include_images=_parse_bool(payload.get("include_images"), default=False),
        include_visualizations=_parse_bool(payload.get("include_visualizations"), default=False),
        upload_visualizations=_parse_bool(payload.get("upload_visualizations"), default=True),
    )


def _download_requested_image(file_path: str, file_name: str) -> DownloadedTongueImage:
    """Download one input image from object storage."""

    bucket_name, folder_path = parse_browser_file_path(file_path)
    object_key = build_object_key(folder_path, file_name)
    task_uuid = folder_path

    download_success, image_bytes = cloud_storage.download_to_memory(object_key)
    if not download_success:
        msg = str(image_bytes)
        raise ValueError(msg)

    image = decode_image_bytes_to_bgr(image_bytes)
    if image is None:
        msg = "failed to decode the downloaded image"
        raise ValueError(msg)

    return DownloadedTongueImage(
        bucket_name=bucket_name,
        folder_path=folder_path,
        object_key=object_key,
        task_uuid=task_uuid,
        file_name=file_name,
        file_path=file_path,
        image_bytes=image_bytes,
        image=image,
    )


def _encode_jpeg(image: np.ndarray) -> bytes:
    """Encode one image as JPEG bytes."""

    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        msg = "failed to encode image as JPEG"
        raise ValueError(msg)
    return encoded.tobytes()


def _encode_json(payload: Any) -> bytes:
    """Encode one JSON payload as UTF-8 bytes."""

    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _upload_bytes(
    *,
    folder_path: str,
    filename: str,
    file_bytes: bytes,
    content_type: str,
) -> dict[str, str] | None:
    """Upload one in-memory file to object storage."""

    object_key = build_object_key(folder_path, filename)
    upload_success, upload_result = cloud_storage.upload_from_memory(
        file_bytes,
        object_key,
        content_type=content_type,
    )
    if not upload_success:
        print(f"Failed to upload {filename}: {upload_result}")
        return None
    return {"filename": filename, "object_key": object_key}


def _upload_phase1_outputs(
    *,
    folder_path: str,
    analysis: TonguePhase1Analysis,
    upload_visualizations: bool,
) -> list[dict[str, str]]:
    """Upload phase-1 tongue outputs back to object storage."""

    upload_items: list[tuple[str, bytes, str]] = [
        ("tongue_segmented.jpg", _encode_jpeg(analysis.artifacts.segmented_image), "image/jpeg"),
        ("tongue_mask.jpg", _encode_jpeg(analysis.artifacts.mask_image), "image/jpeg"),
        ("tongue_analysis_result.json", _encode_json(analysis.analysis_payload), "application/json"),
        ("expert_label_template.json", _encode_json(analysis.expert_label_template), "application/json"),
    ]

    visualizations = analysis.artifacts.moisture_visualizations
    if upload_visualizations and visualizations is not None:
        upload_items.extend(
            [
                ("tongue_gloss_mask.jpg", _encode_jpeg(visualizations.gloss_mask), "image/jpeg"),
                ("tongue_gloss_overlay.jpg", _encode_jpeg(visualizations.gloss_overlay), "image/jpeg"),
                ("tongue_crack_mask.jpg", _encode_jpeg(visualizations.crack_mask), "image/jpeg"),
                ("tongue_crack_skeleton.jpg", _encode_jpeg(visualizations.crack_skeleton), "image/jpeg"),
                ("tongue_crack_overlay.jpg", _encode_jpeg(visualizations.crack_overlay), "image/jpeg"),
                ("tongue_overexposed_mask.jpg", _encode_jpeg(visualizations.overexposed_mask), "image/jpeg"),
                ("tongue_overexposed_overlay.jpg", _encode_jpeg(visualizations.overexposed_overlay), "image/jpeg"),
                ("tongue_moisture_heatmap.jpg", _encode_jpeg(visualizations.moisture_heatmap), "image/jpeg"),
                (
                    "tongue_moisture_score_breakdown.jpg",
                    _encode_jpeg(visualizations.score_breakdown_image),
                    "image/jpeg",
                ),
                (
                    "tongue_moisture_score_breakdown.json",
                    _encode_json(asdict(visualizations.score_breakdown)),
                    "application/json",
                ),
            ]
        )

    uploaded_files: list[dict[str, str]] = []
    for filename, file_bytes, content_type in upload_items:
        uploaded_file = _upload_bytes(
            folder_path=folder_path,
            filename=filename,
            file_bytes=file_bytes,
            content_type=content_type,
        )
        if uploaded_file is not None:
            uploaded_files.append(uploaded_file)
    return uploaded_files


def _maybe_save_local_outputs(
    task_dir: Path,
    *,
    analysis: TonguePhase1Analysis,
    save_visualizations: bool,
) -> None:
    """Persist debugging outputs locally when the feature flag is enabled."""

    if not SAVE_LOCAL_IMAGES:
        return

    task_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(task_dir / "tongue_segmented.jpg"), analysis.artifacts.segmented_image)
    cv2.imwrite(str(task_dir / "tongue_mask.jpg"), analysis.artifacts.mask_image)
    (task_dir / "tongue_analysis_result.json").write_bytes(_encode_json(analysis.analysis_payload))
    (task_dir / "expert_label_template.json").write_bytes(_encode_json(analysis.expert_label_template))

    visualizations = analysis.artifacts.moisture_visualizations
    if not save_visualizations or visualizations is None:
        return

    cv2.imwrite(str(task_dir / "tongue_gloss_mask.jpg"), visualizations.gloss_mask)
    cv2.imwrite(str(task_dir / "tongue_gloss_overlay.jpg"), visualizations.gloss_overlay)
    cv2.imwrite(str(task_dir / "tongue_crack_mask.jpg"), visualizations.crack_mask)
    cv2.imwrite(str(task_dir / "tongue_crack_skeleton.jpg"), visualizations.crack_skeleton)
    cv2.imwrite(str(task_dir / "tongue_crack_overlay.jpg"), visualizations.crack_overlay)
    cv2.imwrite(str(task_dir / "tongue_overexposed_mask.jpg"), visualizations.overexposed_mask)
    cv2.imwrite(str(task_dir / "tongue_overexposed_overlay.jpg"), visualizations.overexposed_overlay)
    cv2.imwrite(str(task_dir / "tongue_moisture_heatmap.jpg"), visualizations.moisture_heatmap)
    cv2.imwrite(str(task_dir / "tongue_moisture_score_breakdown.jpg"), visualizations.score_breakdown_image)
    (task_dir / "tongue_moisture_score_breakdown.json").write_bytes(
        _encode_json(asdict(visualizations.score_breakdown))
    )


def _segment_for_quality(image: np.ndarray) -> TongueQualityArtifacts:
    """Run the shared phase-1 segmenter and keep the full-coordinate mask."""

    artifacts = tongue_phase1_service.segmenter.segment_image(image)
    return TongueQualityArtifacts(
        bounding_box=artifacts.bounding_box,
        full_mask_image=build_full_mask_image(image.shape, artifacts),
    )


def _build_quality_control_result(
    image: np.ndarray,
    *,
    artifacts: TongueQualityArtifacts | TonguePhase1Analysis | None,
) -> dict[str, Any]:
    """Run the five-state quality-control assessment."""

    if isinstance(artifacts, TonguePhase1Analysis):
        bounding_box = artifacts.artifacts.bounding_box
        full_mask_image = artifacts.full_mask_image
    elif artifacts is not None:
        bounding_box = artifacts.bounding_box
        full_mask_image = artifacts.full_mask_image
    else:
        bounding_box = None
        full_mask_image = None

    return quality_control_service.assess(
        image,
        tongue_bbox=bounding_box,
        tongue_mask=full_mask_image,
    )


def _build_quality_response(
    downloaded: DownloadedTongueImage,
    *,
    artifacts: TongueQualityArtifacts | None,
) -> dict[str, Any]:
    """Build the standalone tongue quality-control response payload."""

    quality_control = _build_quality_control_result(downloaded.image, artifacts=artifacts)
    return {
        "task_uuid": downloaded.task_uuid,
        "status": "success",
        "results": {
            "quality_control": quality_control,
            "image_size": {
                "width": int(downloaded.image.shape[1]),
                "height": int(downloaded.image.shape[0]),
            },
            "bounding_box": None if artifacts is None else artifacts.bounding_box,
            "segmentation_used": artifacts is not None,
        },
        "metadata": {
            "analysis_timestamp": datetime.now().isoformat(),
            "model_path": str(TONGUE_MODEL_PATH),
            "original_image": {
                "file_name": downloaded.file_name,
                "width": int(downloaded.image.shape[1]),
                "height": int(downloaded.image.shape[0]),
            },
        },
        "storage": {
            "type": "cloud",
            "bucket": downloaded.bucket_name,
            "folder": downloaded.folder_path,
            "source_object_key": downloaded.object_key,
        },
        "uploaded_files": [],
    }


def _build_phase1_storage_response(
    downloaded: DownloadedTongueImage,
    *,
    analysis: TonguePhase1Analysis,
    uploaded_files: list[dict[str, str]],
    quality_control: dict[str, Any],
) -> dict[str, Any]:
    """Add Flask/MinIO compatibility fields to the phase-1 response contract."""

    response = dict(analysis.response)
    response.update(
        {
            "task_uuid": downloaded.task_uuid,
            "results": {
                "segmented_image": "tongue_segmented.jpg",
                "mask_image": "tongue_mask.jpg",
                "analysis_result": "tongue_analysis_result.json",
                "expert_label_template": "expert_label_template.json",
                "image_size": response.get("image_size"),
                "bounding_box": response.get("bounding_box"),
                "region_division": response.get("region_division"),
                "quality_control": quality_control,
            },
            "metadata": {
                "analysis_timestamp": datetime.now().isoformat(),
                "model_path": str(TONGUE_MODEL_PATH),
                "original_image": {
                    "file_name": downloaded.file_name,
                    "width": int(downloaded.image.shape[1]),
                    "height": int(downloaded.image.shape[0]),
                },
            },
            "storage": {
                "type": "cloud",
                "bucket": downloaded.bucket_name,
                "folder": downloaded.folder_path,
                "source_object_key": downloaded.object_key,
                "analysis_results_object_key": build_object_key(
                    downloaded.folder_path,
                    "tongue_analysis_result.json",
                ),
            },
            "uploaded_files": uploaded_files,
        }
    )
    return response


def process_tongue_image(
    image: np.ndarray,
    onnx_path: str | os.PathLike[str],
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, int] | None]:
    """Preserve the legacy helper signature used by older callers."""

    if Path(onnx_path) != TONGUE_MODEL_PATH:
        msg = "Custom segmentation paths are no longer supported."
        raise ValueError(msg)

    analysis = tongue_phase1_service.analyze(
        image,
        file_name="legacy_tongue_image.jpg",
        include_images=False,
        include_visualizations=False,
        generate_visualizations=False,
    )
    return (
        analysis.artifacts.segmented_image,
        analysis.artifacts.mask_image,
        analysis.artifacts.bounding_box,
    )


@bp.route("/tongue-quality-check", methods=["POST"])
def tongue_quality_check() -> tuple[dict[str, Any], int] | dict[str, Any]:
    """Assess whether one tongue image passes the paper-compatible QC gate."""

    payload_result = _decode_request_payload()
    if isinstance(payload_result, tuple):
        return payload_result

    try:
        downloaded = _download_requested_image(payload_result.file_path, payload_result.file_name)
    except ValueError as exc:
        return {"error": str(exc)}, 400

    artifacts = None
    if TONGUE_MODEL_PATH.exists():
        try:
            artifacts = _segment_for_quality(downloaded.image)
        except Exception:
            artifacts = None

    return _build_quality_response(downloaded, artifacts=artifacts)


@bp.route("/tongue-segment", methods=["POST"])
def tongue_segment() -> tuple[dict[str, Any], int] | dict[str, Any]:
    """Run complete phase-1 tongue analysis from a MinIO image."""

    payload_result = _decode_request_payload()
    if isinstance(payload_result, tuple):
        return payload_result

    try:
        downloaded = _download_requested_image(payload_result.file_path, payload_result.file_name)
    except ValueError as exc:
        return {"error": str(exc)}, 400

    if not tongue_phase1_service.is_model_available():
        return {"error": f"segmentation model not found: {TONGUE_MODEL_PATH}"}, 500

    try:
        analysis = tongue_phase1_service.analyze(
            downloaded.image,
            file_name=downloaded.file_name,
            include_images=payload_result.include_images,
            include_visualizations=payload_result.include_visualizations,
            generate_visualizations=payload_result.upload_visualizations,
        )
    except FileNotFoundError as exc:
        return {"error": str(exc)}, 503
    except ValueError as exc:
        return {"error": f"tongue phase-1 analysis failed: {exc}"}, 400
    except RuntimeError as exc:
        return {"error": f"tongue phase-1 analysis failed: {exc}"}, 500
    except Exception as exc:
        return {"error": f"tongue phase-1 analysis failed: {exc}"}, 500

    task_dir = UPLOAD_BASE_DIR / downloaded.folder_path
    _maybe_save_local_outputs(
        task_dir,
        analysis=analysis,
        save_visualizations=payload_result.upload_visualizations,
    )
    uploaded_files = _upload_phase1_outputs(
        folder_path=downloaded.folder_path,
        analysis=analysis,
        upload_visualizations=payload_result.upload_visualizations,
    )
    quality_control = _build_quality_control_result(downloaded.image, artifacts=analysis)
    return _build_phase1_storage_response(
        downloaded,
        analysis=analysis,
        uploaded_files=uploaded_files,
        quality_control=quality_control,
    )


@bp.route("/tongue-segment/health", methods=["GET"])
def tongue_health_check() -> dict[str, Any]:
    """Return the health status of the tongue phase-1 service."""

    return {
        "status": "healthy" if tongue_phase1_service.is_model_available() else "unhealthy",
        "service": "tongue_phase1",
        "model_loaded": tongue_phase1_service.is_model_available(),
        "model_path": str(TONGUE_MODEL_PATH),
        "phase1_enabled": True,
        "quality_control_enabled": True,
        "quality_control_route": "/v1/tongue-quality-check",
    }


__all__ = ["process_tongue_image", "tongue_segment", "tongue_quality_check", "tongue_health_check"]
