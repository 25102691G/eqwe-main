"""Helpers for routing chat image questions through the face analysis pipeline."""

from __future__ import annotations

import json
import mimetypes
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from api.agents import generate_skin_report
from api.configs.storage_config import STORAGE_CONFIG
from api.controllers.v1.image_decode import decode_image_file_to_bgr
from api.controllers.v1.mobile_contract import (
    build_mobile_asset_url,
    build_object_key,
    build_result_images,
)
from api.middleware.storage.cloud_storage_service import CloudStorageService
from api.services.diagnosis_summary import build_diagnosis_context
from api.services.report_scores import (
    calculate_skin_tone_score,
    calculate_smoothness_score,
)

FACE_REQUEST_KEYWORDS = (
    "肤况",
    "肤况分析",
    "辅助分析",
    "面诊",
    "面部",
    "脸",
    "皮肤",
    "肌肤",
    "护肤",
    "护理",
    "建议",
    "分析",
    "检测",
    "得分",
    "评分",
    "毛孔",
    "痘",
    "痘痘",
    "斑",
    "肤色",
    "敏感",
    "出油",
    "黑眼圈",
    "细纹",
    "皱纹",
    "水油",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_BASE_DIR = PROJECT_ROOT / "upload_files"
UPLOAD_BASE_DIR.mkdir(parents=True, exist_ok=True)

cloud_storage = CloudStorageService(STORAGE_CONFIG)


def _classify_skin_type(oiliness: float, moisture: float) -> dict[str, str]:
    """Classify skin condition into short oil and moisture suggestions."""
    if oiliness > 60:
        oil_suggestion = "当前出油偏多，建议加强温和清洁和控油护理。"
    elif oiliness > 40:
        oil_suggestion = "当前水油有一定波动，建议保持温和清洁并减少刺激。"
    else:
        oil_suggestion = "当前出油相对稳定，继续保持基础清洁和保湿即可。"

    if moisture > 50:
        moisture_suggestion = "当前保湿状态尚可，建议继续做好日常保湿和防晒。"
    else:
        moisture_suggestion = "当前存在缺水倾向，建议优先加强补水和屏障修护。"

    return {
        "oil_suggestion": oil_suggestion,
        "moisture_suggestion": moisture_suggestion,
    }


def _guess_content_type(filename: str) -> str:
    """Guess the MIME type for one saved artifact."""
    content_type, _ = mimetypes.guess_type(filename)
    return content_type or "application/octet-stream"


def _encode_jpeg(image: np.ndarray | None) -> bytes | None:
    """Encode one OpenCV image to JPEG bytes."""
    if image is None or not getattr(image, "size", 0):
        return None
    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        return None
    return encoded.tobytes()


def _persist_artifact(folder_path: str, filename: str, payload: bytes) -> dict[str, str]:
    """Persist one generated artifact locally and upload it best-effort to storage."""
    target_dir = UPLOAD_BASE_DIR / folder_path
    target_dir.mkdir(parents=True, exist_ok=True)
    local_path = target_dir / filename
    local_path.write_bytes(payload)

    object_key = build_object_key(folder_path, filename)
    cloud_storage.upload_from_memory(
        payload,
        object_key,
        content_type=_guess_content_type(filename),
    )
    return {
        "filename": filename,
        "object_key": object_key,
    }


def _build_analysis_results(
    image: np.ndarray,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Run the face-analysis services and collect visualization artifacts."""
    from api.controllers.v1.infer import (
        face_service,
        sensitivity_service,
        skin_class_service,
        skin_service,
    )
    from api.services.detector_strict import run_pipeline
    from api.services.nasolabial import nasolabial_process
    from api.services.nevus import process_image

    landmarks = face_service.detect_mediapipe_landmarks(image)
    if landmarks is None:
        msg = "未检测到可用人脸"
        raise ValueError(msg)

    masks = face_service.create_comprehensive_masks(image, landmarks)
    if masks is None:
        msg = "面部区域分割失败"
        raise ValueError(msg)

    analysis_masks = face_service.create_skin_analysis_masks(image, landmarks)
    if analysis_masks is None:
        msg = "皮肤分析区域分割失败"
        raise ValueError(msg)

    landmarks_image = face_service.draw_landmarks(image.copy(), landmarks)

    def task_oil_analysis() -> tuple[float, float, bytes | None, bytes | None]:
        hsv_oil_mask, oil_score = skin_service.detect_oil_regions_hsv(
            image,
            analysis_masks["combined_analysis_regions"],
        )
        oil_visualization = skin_service.apply_orange_mask(image, hsv_oil_mask, alpha=0.5)
        moisture_map, moisture_score = skin_service.calculate_moisture_enhanced(
            image,
            masks["face_contour"],
        )
        moisture_visualization = skin_service.create_moisture_visualization_enhanced(
            image,
            moisture_map,
            masks,
        )
        return (
            float(oil_score),
            float(moisture_score),
            _encode_jpeg(oil_visualization),
            _encode_jpeg(moisture_visualization),
        )

    def task_skin_tone() -> tuple[dict[str, Any], bytes | None]:
        ita_score, skin_tone_category, visualization_image = skin_class_service.classify_skin_tone(
            image,
            save_visualization=True,
            save_path=None,
        )
        return (
            skin_class_service.create_classification_result(ita_score, skin_tone_category),
            _encode_jpeg(visualization_image),
        )

    def task_hyperpigmentation() -> tuple[int, int, bytes | None]:
        visualization_image, zhi_count, se_ban = process_image(image)
        return int(zhi_count), int(se_ban), _encode_jpeg(visualization_image)

    def task_sensitivity() -> tuple[dict[str, Any], bytes | None]:
        sensitivity_analysis = sensitivity_service.analyze_skin_sensitivity(
            image,
            alpha=0.65,
            wr=0.75,
            wt=0.25,
            smooth=11,
            use_mediapipe=True,
            save_debug_images=False,
            save_dir=None,
            mask=masks["skin_only"],
        )
        return (
            sensitivity_service.create_sensitivity_result(sensitivity_analysis),
            dict(sensitivity_analysis.get("image_bytes") or {}).get("overlay_on_white"),
        )

    def task_smoothness() -> dict[str, Any]:
        return run_pipeline(image, None, masks)

    def task_wrinkles() -> tuple[dict[str, Any], bytes | None, bytes | None]:
        nasolabial_image, black_eye_image, wrinkle_result = nasolabial_process(image)
        return wrinkle_result, _encode_jpeg(nasolabial_image), _encode_jpeg(black_eye_image)

    with ThreadPoolExecutor(max_workers=6) as executor:
        future_oil = executor.submit(task_oil_analysis)
        future_skin_tone = executor.submit(task_skin_tone)
        future_hyperpigmentation = executor.submit(task_hyperpigmentation)
        future_sensitivity = executor.submit(task_sensitivity)
        future_smoothness = executor.submit(task_smoothness)
        future_wrinkles = executor.submit(task_wrinkles)

        oil_score, moisture_score, oil_bytes, moisture_bytes = future_oil.result()
        skin_tone_result, skin_tone_visualization_bytes = future_skin_tone.result()
        zhi_count, se_ban, hyperpigmentation_bytes = future_hyperpigmentation.result()
        sensitivity_result, sensitivity_bytes = future_sensitivity.result()
        smooth_json = future_smoothness.result()
        wrinkles_result, nasolabial_bytes, black_eye_bytes = future_wrinkles.result()

    skin_type = _classify_skin_type(oil_score, moisture_score)
    skin_area_pixels = int(np.count_nonzero(masks["skin_only"]))
    smoothness_counts = dict(smooth_json.get("counts") or {})
    acne_group_total = int(smoothness_counts.get("acne_group_total", 0))
    blackheads_count = int((smoothness_counts.get("blackheads") or {}).get("counts", 0))
    pores_count = int((smoothness_counts.get("pores") or {}).get("counts", 0))

    skin_tone_score = calculate_skin_tone_score(
        stain_count=int(se_ban),
        skin_area_pixels=skin_area_pixels,
    )
    smoothness_score = calculate_smoothness_score(
        acne_group_total=acne_group_total,
        blackheads_count=blackheads_count,
        pores_count=pores_count,
        skin_area_pixels=skin_area_pixels,
    )

    analysis_results = {
        "oil_moi": {
            "oil_analysis": {
                "oil_score": round(oil_score, 2),
                "description": skin_type["oil_suggestion"],
            },
            "moisture_analysis": {
                "moisture_score": round(moisture_score, 2),
                "description": skin_type["moisture_suggestion"],
            },
            "score": round((oil_score + moisture_score) / 2, 2),
            "description": "反映皮肤水油平衡状态。",
        },
        "skin_color": {
            "skin_tone_classification": skin_tone_result,
            "score": skin_tone_score,
            "description": "根据色沉与肤色均匀度估算肤色状态。",
            "score_detail": {
                "skin_area_pixels": skin_area_pixels,
                "stain_count": se_ban,
                "stain_density_per_10k": round((se_ban * 10000.0) / max(skin_area_pixels, 1), 2),
            },
            "hyperpigmentation": {
                "zhi_count": zhi_count,
                "se_ban": se_ban,
                "description": "色沉点位统计。",
                "suggestion": "建议持续防晒，并避免反复刺激。",
            },
        },
        "sensitivity": {
            "sensitivity_analysis": sensitivity_result,
            "score": sensitivity_result["sensitivity_score"],
            "description": "根据红度与纹理稳定度估算敏感倾向。",
        },
        "smoothness": {
            "smooth": smooth_json["counts"],
            "score": smoothness_score,
            "description": "根据痘痘、黑头和毛孔密度估算平滑度。",
            "score_detail": {
                "skin_area_pixels": skin_area_pixels,
                "acne_group_total": acne_group_total,
                "blackheads_count": blackheads_count,
                "pores_count": pores_count,
            },
        },
        "wrinkles": wrinkles_result,
    }

    artifacts: dict[str, bytes] = {}
    for filename, payload in (
        ("landmarks.jpg", _encode_jpeg(landmarks_image)),
        ("oil_mask_fixed.jpg", oil_bytes),
        ("moisture_analysis.jpg", moisture_bytes),
        ("skin_tone_classification.jpg", skin_tone_visualization_bytes),
        ("hyperpigmentation.jpg", hyperpigmentation_bytes),
        ("overlay_on_white.jpg", sensitivity_bytes),
        ("imgnasolabial.jpg", nasolabial_bytes),
        ("black_eye.jpg", black_eye_bytes),
    ):
        if payload:
            artifacts[filename] = payload

    for filename, payload in dict(smooth_json.get("image_bytes") or {}).items():
        if isinstance(payload, (bytes, bytearray)):
            artifacts[filename] = bytes(payload)

    return analysis_results, artifacts


def should_run_face_analysis(
    *,
    user_text: str,
    attachments: list[dict[str, Any]] | None,
) -> bool:
    """Return whether the current turn should trigger face analysis.

    Args:
        user_text: User input text.
        attachments: Resolved attachment payloads for the turn.

    Returns:
        `True` when the turn includes an image and the prompt looks like a
        face-analysis or skincare request.
    """
    if not attachments:
        return False

    has_image = any(
        isinstance(attachment, dict) and attachment.get("kind") == "image"
        for attachment in attachments
    )
    if not has_image:
        return False

    normalized_text = str(user_text or "").strip().lower()
    if not normalized_text:
        return False

    return any(keyword in normalized_text for keyword in FACE_REQUEST_KEYWORDS)


def analyze_face_image_to_context(
    image_path: str | Path,
    *,
    source_folder: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Analyze one uploaded face image and build a skin assistance context.

    Args:
        image_path: Local stored image path.
        source_folder: Synthetic source identifier for chat storage.

    Returns:
        A tuple of the full analysis payload and the chat skin assistance context.
    """
    from api.controllers.v1.face_align import face_align_service

    local_path = Path(image_path)
    if not local_path.exists():
        msg = f"Image file does not exist: {local_path}"
        raise FileNotFoundError(msg)

    image = decode_image_file_to_bgr(local_path)
    if image is None:
        msg = "无法读取上传的图片"
        raise ValueError(msg)

    aligned_image, message = face_align_service.process_image(image, 0.3)
    if aligned_image is None:
        raise ValueError(message)

    analysis_results, artifacts = _build_analysis_results(aligned_image)
    aligned_bytes = _encode_jpeg(aligned_image)
    if aligned_bytes:
        artifacts["align.jpg"] = aligned_bytes

    llm_report = generate_skin_report(analysis_results)
    report_filename = "analysis_results.json"
    analysis_report_url = build_mobile_asset_url(source_folder, report_filename)

    artifact_entries = [
        {
            "filename": filename,
            "object_key": build_object_key(source_folder, filename),
        }
        for filename in artifacts
    ]
    uploaded_files = [
        *artifact_entries,
        {
            "filename": report_filename,
            "object_key": build_object_key(source_folder, report_filename),
        },
    ]

    metadata = {
        "analysis_timestamp": datetime.now(UTC).isoformat(),
        "image_info": {
            "original_size": {
                "width": int(aligned_image.shape[1]),
                "height": int(aligned_image.shape[0]),
            },
            "file_name": local_path.name,
        },
        "folder": source_folder,
        "analysis_report_url": analysis_report_url,
    }

    analysis_payload = {
        "status": "success",
        "analysis_results": analysis_results,
        "llm_report": llm_report,
        "metadata": metadata,
        "uploaded_files": uploaded_files,
        "storage": {
            "folder": source_folder,
        },
        "analysis_report_url": analysis_report_url,
        "result_images": build_result_images(
            source_folder,
            artifact_entries,
            preferred=("align.jpg",),
        ),
    }

    for filename, payload in artifacts.items():
        _persist_artifact(source_folder, filename, payload)

    report_bytes = json.dumps(analysis_payload, ensure_ascii=False, indent=2).encode("utf-8")
    _persist_artifact(source_folder, report_filename, report_bytes)

    diagnosis_context = build_diagnosis_context(analysis_payload)
    return analysis_payload, diagnosis_context


__all__ = [
    "analyze_face_image_to_context",
    "should_run_face_analysis",
]
