"""Reusable ONNX tongue segmentation helpers for the FastAPI service."""

from __future__ import annotations

import base64
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import is_dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import threading

import cv2
import numpy as np

from tongue_diagnosis.app.demo_report import build_demo_report_payload
from tongue_diagnosis.app.tongue_color import RegionColorMeasurement, analyze_region_colors, analyze_tongue_color
from tongue_diagnosis.app.tongue_coat import (
    RegionCoatMeasurement,
    TongueCoatMeasurement,
    analyze_region_coat,
    analyze_tongue_coat,
)
from tongue_diagnosis.app.tongue_moisture import (
    RegionMoistureMeasurement,
    TongueMoistureMeasurement,
    analyze_region_moisture,
    analyze_tongue_moisture,
)
from tongue_diagnosis.app.tongue_moisture_visualization import MoistureVisualizationArtifacts

REGION_LABELS: dict[str, str] = {
    "1": "left region",
    "2": "top region",
    "3": "right region",
    "4": "bottom region",
    "center": "center region",
}
_REGION_CODES: dict[str, int] = {
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "center": 5,
}
_REGION_FILL_ORDER: tuple[str, ...] = ("center", "1", "2", "3", "4")
_REGION_OUTPUT_ORDER: tuple[str, ...] = ("1", "2", "center", "3", "4")


@dataclass(frozen=True)
class SegmentationQualityMetrics:
    """Image-level segmentation stability metrics."""

    tongue_area_ratio: float
    bbox_touches_edge: bool
    region_coverage_ratios: dict[str, float]


@dataclass(frozen=True)
class SegmentationArtifacts:
    """Outputs produced by one tongue segmentation run."""

    segmented_image: np.ndarray
    mask_image: np.ndarray
    bounding_box: dict[str, int]
    cropped_image: np.ndarray | None = None
    region_masks: dict[str, np.ndarray] | None = None
    tongue_color: RegionColorMeasurement | None = None
    region_colors: tuple[RegionColorMeasurement, ...] = ()
    tongue_coat: TongueCoatMeasurement | None = None
    region_coat: tuple[RegionCoatMeasurement, ...] = ()
    region_moisture: tuple[RegionMoistureMeasurement, ...] = ()
    tongue_moisture: TongueMoistureMeasurement | None = None
    moisture_visualizations: MoistureVisualizationArtifacts | None = None
    segmentation_quality: SegmentationQualityMetrics | None = None


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Return a numerically stable softmax over the class axis."""

    shifted = logits - np.max(logits, axis=2, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=2, keepdims=True)


def build_region_polygons(bbox_rect: tuple[int, int, int, int]) -> dict[str, np.ndarray]:
    """Return the five polygon regions defined inside one bounding box.

    Args:
        bbox_rect: Bounding box expressed as `(x, y, width, height)`.

    Returns:
        A mapping from region identifier to polygon vertices.
    """

    x, y, width, height = bbox_rect
    top_left = (x, y)
    top_right = (x + width, y)
    bottom_left = (x, y + height)
    bottom_right = (x + width, y + height)

    center_x = x + width // 2
    center_y = y + height // 2
    rect_width = width // 5
    rect_height = height // 5

    center_rect_top_left = (center_x - rect_width, center_y - rect_height)
    center_rect_top_right = (center_x + rect_width, center_y - rect_height)
    center_rect_bottom_left = (center_x - rect_width, center_y + rect_height)
    center_rect_bottom_right = (center_x + rect_width, center_y + rect_height)

    return {
        "1": np.array(
            [top_left, center_rect_top_left, center_rect_bottom_left, bottom_left],
            dtype=np.int32,
        ),
        "2": np.array(
            [top_left, top_right, center_rect_top_right, center_rect_top_left],
            dtype=np.int32,
        ),
        "3": np.array(
            [center_rect_top_right, top_right, bottom_right, center_rect_bottom_right],
            dtype=np.int32,
        ),
        "4": np.array(
            [center_rect_bottom_left, center_rect_bottom_right, bottom_right, bottom_left],
            dtype=np.int32,
        ),
        "center": np.array(
            [
                center_rect_top_left,
                center_rect_top_right,
                center_rect_bottom_right,
                center_rect_bottom_left,
            ],
            dtype=np.int32,
        ),
    }


def _build_region_masks(
    tongue_mask: np.ndarray,
    bbox_rect: tuple[int, int, int, int],
) -> dict[str, np.ndarray]:
    """Assign every tongue pixel to exactly one of the five regions."""

    label_map = np.zeros(tongue_mask.shape, dtype=np.uint8)
    polygons = build_region_polygons(bbox_rect)

    for region_id in _REGION_FILL_ORDER:
        polygon_mask = np.zeros(tongue_mask.shape, dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [polygons[region_id]], 1)
        unassigned_pixels = (label_map == 0) & (polygon_mask > 0) & tongue_mask
        label_map[unassigned_pixels] = _REGION_CODES[region_id]

    leftover_pixels = tongue_mask & (label_map == 0)
    label_map[leftover_pixels] = _REGION_CODES["center"]

    return {
        region_id: label_map == region_code
        for region_id, region_code in _REGION_CODES.items()
    }


def _bbox_touches_image_edge(
    bbox: dict[str, int],
    image_shape: tuple[int, ...],
    *,
    margin_px: int = 3,
) -> bool:
    """Return whether one bbox is too close to the original image edge."""

    image_height, image_width = image_shape[:2]
    x_value = bbox["x"]
    y_value = bbox["y"]
    width = bbox["w"]
    height = bbox["h"]
    return (
        x_value <= margin_px
        or y_value <= margin_px
        or x_value + width >= image_width - margin_px
        or y_value + height >= image_height - margin_px
    )


def _build_segmentation_quality(
    *,
    full_mask: np.ndarray,
    image_shape: tuple[int, ...],
    bbox: dict[str, int],
    region_masks: dict[str, np.ndarray],
    total_tongue_pixels: int,
) -> SegmentationQualityMetrics:
    """Build segmentation stability metrics for API consumers."""

    image_area = max(int(image_shape[0]) * int(image_shape[1]), 1)
    tongue_area_ratio = round(float(np.count_nonzero(full_mask > 0) / image_area), 4)
    region_coverage_ratios = {
        region_id: round(int(np.count_nonzero(region_mask)) / max(total_tongue_pixels, 1), 4)
        for region_id, region_mask in region_masks.items()
    }
    return SegmentationQualityMetrics(
        tongue_area_ratio=tongue_area_ratio,
        bbox_touches_edge=_bbox_touches_image_edge(bbox, image_shape),
        region_coverage_ratios=region_coverage_ratios,
    )


def calculate_region_colors(
    image: np.ndarray,
    mask: np.ndarray,
    bbox_rect: tuple[int, int, int, int],
) -> tuple[RegionColorMeasurement, ...]:
    """Calculate representative colors for the five tongue regions.

    Args:
        image: Cropped BGR tongue image without overlays.
        mask: Binary tongue mask aligned with `image`.
        bbox_rect: Bounding box expressed as `(x, y, width, height)` within `image`.

    Returns:
        One color summary per tongue region. Background pixels are excluded.
    """

    tongue_mask = mask > 0
    total_tongue_pixels = int(np.count_nonzero(tongue_mask))
    region_masks = _build_region_masks(tongue_mask, bbox_rect)
    return analyze_region_colors(
        image,
        region_masks,
        region_output_order=_REGION_OUTPUT_ORDER,
        region_labels=REGION_LABELS,
        total_tongue_pixels=total_tongue_pixels,
    )


def calculate_region_moisture(
    image: np.ndarray,
    mask: np.ndarray,
    bbox_rect: tuple[int, int, int, int],
) -> tuple[RegionMoistureMeasurement, ...]:
    """Calculate moisture-related metrics for the five tongue regions.

    Args:
        image: Cropped BGR tongue image without overlays.
        mask: Binary tongue mask aligned with `image`.
        bbox_rect: Bounding box expressed as `(x, y, width, height)` within `image`.

    Returns:
        One moisture summary per tongue region. Background pixels are excluded.
    """

    tongue_mask = mask > 0
    total_tongue_pixels = int(np.count_nonzero(tongue_mask))
    region_masks = _build_region_masks(tongue_mask, bbox_rect)
    return analyze_region_moisture(
        image,
        region_masks,
        region_output_order=_REGION_OUTPUT_ORDER,
        region_labels=REGION_LABELS,
        total_tongue_pixels=total_tongue_pixels,
    )


def calculate_tongue_moisture(
    image: np.ndarray,
    mask: np.ndarray,
    bbox_rect: tuple[int, int, int, int],
) -> TongueMoistureMeasurement | None:
    """Calculate image-level tongue moisture metrics.

    Args:
        image: Cropped BGR tongue image without overlays.
        mask: Binary tongue mask aligned with `image`.
        bbox_rect: Bounding box expressed as `(x, y, width, height)` within `image`.

    Returns:
        One image-level moisture summary, or `None` when no tongue pixels exist.
    """

    tongue_mask = mask > 0
    return analyze_tongue_moisture(image, tongue_mask, bbox_rect)


def draw_dashed_line(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
    dash_length: int = 10,
) -> None:
    """Draw a dashed line on one image."""

    distance = float(np.hypot(end[0] - start[0], end[1] - start[1]))
    dashes = max(int(distance / max(dash_length, 1)), 1)

    for index in range(dashes):
        segment_start = (
            int(start[0] + (end[0] - start[0]) * index / dashes),
            int(start[1] + (end[1] - start[1]) * index / dashes),
        )
        segment_end = (
            int(start[0] + (end[0] - start[0]) * (index + 0.5) / dashes),
            int(start[1] + (end[1] - start[1]) * (index + 0.5) / dashes),
        )
        cv2.line(image, segment_start, segment_end, color, thickness)


def draw_region_division(image: np.ndarray, bbox_rect: tuple[int, int, int, int]) -> np.ndarray:
    """Overlay the five-region tongue partition on the segmented image."""

    result = image.copy()
    polygons = build_region_polygons(bbox_rect)
    color = (255, 255, 0)

    top_left = tuple(int(value) for value in polygons["1"][0])
    top_right = tuple(int(value) for value in polygons["2"][1])
    bottom_left = tuple(int(value) for value in polygons["4"][3])
    bottom_right = tuple(int(value) for value in polygons["3"][2])
    center_rect_top_left = tuple(int(value) for value in polygons["center"][0])
    center_rect_top_right = tuple(int(value) for value in polygons["center"][1])
    center_rect_bottom_right = tuple(int(value) for value in polygons["center"][2])
    center_rect_bottom_left = tuple(int(value) for value in polygons["center"][3])

    draw_dashed_line(result, center_rect_top_left, center_rect_top_right, color)
    draw_dashed_line(result, center_rect_top_right, center_rect_bottom_right, color)
    draw_dashed_line(result, center_rect_bottom_right, center_rect_bottom_left, color)
    draw_dashed_line(result, center_rect_bottom_left, center_rect_top_left, color)

    draw_dashed_line(result, center_rect_top_left, top_left, color)
    draw_dashed_line(result, center_rect_top_right, top_right, color)
    draw_dashed_line(result, center_rect_bottom_left, bottom_left, color)
    draw_dashed_line(result, center_rect_bottom_right, bottom_right, color)
    return result


class TongueSegmenter:
    """Load one ONNX model and run tongue segmentation on uploaded images."""

    def __init__(self, model_path: str | os.PathLike[str], output_dir: str | os.PathLike[str]) -> None:
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._session = None
        self._session_lock = threading.Lock()

    def is_model_available(self) -> bool:
        """Return whether the configured ONNX model exists locally."""

        return self.model_path.exists()

    def segment_image(self, image: np.ndarray) -> SegmentationArtifacts:
        """Segment one BGR image and return the rendered outputs."""

        if not self.is_model_available():
            msg = f"segmentation model not found: {self.model_path}"
            raise FileNotFoundError(msg)

        session = self._get_session()
        input_name = session.get_inputs()[0].name

        original_height, original_width = image.shape[:2]
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_image, (512, 512)).astype(np.float32) / 255.0
        model_input = np.expand_dims(np.transpose(resized, (2, 0, 1)), 0)

        output = session.run(None, {input_name: model_input.reshape((1, 3, 512, 512))})[0]
        probabilities = _stable_softmax(output[0].transpose((1, 2, 0)))
        prediction = cv2.resize(
            probabilities,
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR,
        ).argmax(axis=-1)

        full_mask = np.where(prediction != 0, 255, 0).astype(np.uint8)
        full_mask = cv2.dilate(full_mask, np.ones((3, 3), np.uint8), iterations=2)

        contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            msg = "no tongue region found"
            raise ValueError(msg)

        contour = max(contours, key=cv2.contourArea)
        x, y, width, height = cv2.boundingRect(contour)
        bbox = {"x": int(x), "y": int(y), "w": int(width), "h": int(height)}

        y_start = max(0, y - 30)
        y_end = min(image.shape[0], y + height + 30)
        x_start = max(0, x - 30)
        x_end = min(image.shape[1], x + width + 30)

        cropped_mask = full_mask[y_start:y_end, x_start:x_end]
        cropped_image = image[y_start:y_end, x_start:x_end]
        zero_mask = cropped_mask == 0

        segmented = cv2.bitwise_and(cropped_image, cropped_image, mask=cropped_mask)
        segmented[zero_mask] = (255, 255, 255)

        adjusted_bbox = (x - x_start, y - y_start, width, height)
        tongue_mask = cropped_mask > 0
        total_tongue_pixels = int(np.count_nonzero(tongue_mask))
        region_masks = _build_region_masks(tongue_mask, adjusted_bbox)
        region_colors = analyze_region_colors(
            cropped_image,
            region_masks,
            region_output_order=_REGION_OUTPUT_ORDER,
            region_labels=REGION_LABELS,
            total_tongue_pixels=total_tongue_pixels,
        )
        tongue_color = analyze_tongue_color(cropped_image, tongue_mask)
        tongue_coat = analyze_tongue_coat(cropped_image, tongue_mask)
        region_coat = analyze_region_coat(
            cropped_image,
            region_masks,
            region_output_order=_REGION_OUTPUT_ORDER,
            region_labels=REGION_LABELS,
            total_tongue_pixels=total_tongue_pixels,
        )
        region_moisture = analyze_region_moisture(
            cropped_image,
            region_masks,
            region_output_order=_REGION_OUTPUT_ORDER,
            region_labels=REGION_LABELS,
            total_tongue_pixels=total_tongue_pixels,
        )
        tongue_moisture = analyze_tongue_moisture(cropped_image, tongue_mask, adjusted_bbox)
        segmentation_quality = _build_segmentation_quality(
            full_mask=full_mask,
            image_shape=image.shape,
            bbox=bbox,
            region_masks=region_masks,
            total_tongue_pixels=total_tongue_pixels,
        )
        rendered = draw_region_division(segmented, adjusted_bbox)

        center_x = adjusted_bbox[0] + adjusted_bbox[2] // 2
        center_y = adjusted_bbox[1] + adjusted_bbox[3] // 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_color = (0, 0, 255)

        cv2.putText(rendered, "1", (max(5, adjusted_bbox[0] - 25), center_y), font, 1.5, font_color, 3)
        cv2.putText(rendered, "2", (center_x - 10, max(25, adjusted_bbox[1] - 5)), font, 1.5, font_color, 3)
        cv2.putText(
            rendered,
            "3",
            (min(rendered.shape[1] - 30, adjusted_bbox[0] + adjusted_bbox[2] + 10), center_y),
            font,
            1.5,
            font_color,
            3,
        )
        cv2.putText(
            rendered,
            "4",
            (center_x - 10, min(rendered.shape[0] - 5, adjusted_bbox[1] + adjusted_bbox[3] + 40)),
            font,
            1.5,
            font_color,
            3,
        )

        return SegmentationArtifacts(
            segmented_image=rendered,
            mask_image=cropped_mask,
            bounding_box=bbox,
            cropped_image=cropped_image,
            region_masks=region_masks,
            tongue_color=tongue_color,
            region_colors=region_colors,
            tongue_coat=tongue_coat,
            region_coat=region_coat,
            region_moisture=region_moisture,
            tongue_moisture=tongue_moisture,
            segmentation_quality=segmentation_quality,
        )

    def save_outputs(
        self,
        *,
        file_stem: str,
        artifacts: SegmentationArtifacts,
    ) -> list[dict[str, str]]:
        """Save the rendered segmentation outputs under the configured output directory."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = self.output_dir / f"{file_stem}_{timestamp}"
        output_path.mkdir(parents=True, exist_ok=True)

        segmented_path = output_path / "tongue_segmented.jpg"
        mask_path = output_path / "tongue_mask.jpg"
        cv2.imwrite(str(segmented_path), artifacts.segmented_image)
        cv2.imwrite(str(mask_path), artifacts.mask_image)

        saved_files = [
            {"filename": segmented_path.name, "path": str(segmented_path)},
            {"filename": mask_path.name, "path": str(mask_path)},
        ]
        saved_files.extend(self._save_analysis_json(output_path, artifacts))
        saved_files.extend(self._save_moisture_visualizations(output_path, artifacts.moisture_visualizations))
        return saved_files

    @staticmethod
    def encode_image_to_base64(image: np.ndarray) -> str:
        """Encode one image to a base64 JPEG string."""

        success, encoded = cv2.imencode(".jpg", image)
        if not success:
            msg = "failed to encode image as JPEG"
            raise ValueError(msg)
        return base64.b64encode(encoded.tobytes()).decode("utf-8")

    def _save_moisture_visualizations(
        self,
        output_path: Path,
        visualizations: MoistureVisualizationArtifacts | None,
    ) -> list[dict[str, str]]:
        """Persist optional moisture debug visualizations to one output directory."""

        if visualizations is None:
            return []

        image_outputs = {
            "tongue_gloss_mask.jpg": visualizations.gloss_mask,
            "tongue_gloss_overlay.jpg": visualizations.gloss_overlay,
            "tongue_crack_mask.jpg": visualizations.crack_mask,
            "tongue_crack_skeleton.jpg": visualizations.crack_skeleton,
            "tongue_crack_overlay.jpg": visualizations.crack_overlay,
            "tongue_overexposed_mask.jpg": visualizations.overexposed_mask,
            "tongue_overexposed_overlay.jpg": visualizations.overexposed_overlay,
            "tongue_moisture_heatmap.jpg": visualizations.moisture_heatmap,
            "tongue_moisture_score_breakdown.jpg": visualizations.score_breakdown_image,
        }

        saved_files: list[dict[str, str]] = []
        for filename, image in image_outputs.items():
            image_path = output_path / filename
            cv2.imwrite(str(image_path), image)
            saved_files.append({"filename": image_path.name, "path": str(image_path)})

        score_breakdown_path = output_path / "tongue_moisture_score_breakdown.json"
        score_breakdown_path.write_text(
            json.dumps(asdict(visualizations.score_breakdown), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        saved_files.append({"filename": score_breakdown_path.name, "path": str(score_breakdown_path)})
        return saved_files

    def _save_analysis_json(
        self,
        output_path: Path,
        artifacts: SegmentationArtifacts,
    ) -> list[dict[str, str]]:
        """Persist structured algorithm outputs and an expert-label template."""

        analysis_path = output_path / "tongue_analysis_result.json"
        analysis_payload = {
            "bounding_box": artifacts.bounding_box,
            "segmentation_quality": _to_jsonable_dataclass(artifacts.segmentation_quality),
            "tongue_color": _to_jsonable_dataclass(artifacts.tongue_color),
            "region_colors": _to_jsonable_dataclass(artifacts.region_colors),
            "tongue_coat": _to_jsonable_dataclass(artifacts.tongue_coat),
            "region_coat": _to_jsonable_dataclass(artifacts.region_coat),
            "tongue_moisture": _to_jsonable_dataclass(artifacts.tongue_moisture),
            "region_moisture": _to_jsonable_dataclass(artifacts.region_moisture),
            "demo_report": build_demo_report_payload(
                tongue_color=artifacts.tongue_color,
                region_colors=artifacts.region_colors,
                tongue_coat=artifacts.tongue_coat,
                region_coat=artifacts.region_coat,
                tongue_moisture=artifacts.tongue_moisture,
                region_moisture=artifacts.region_moisture,
                segmentation_quality=artifacts.segmentation_quality,
            ),
        }
        analysis_path.write_text(
            json.dumps(analysis_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        template_path = output_path / "expert_label_template.json"
        template_path.write_text(
            json.dumps(_build_expert_label_template(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return [
            {"filename": analysis_path.name, "path": str(analysis_path)},
            {"filename": template_path.name, "path": str(template_path)},
        ]

    def _get_session(self):
        """Create the ONNX runtime session lazily."""

        if self._session is None:
            with self._session_lock:
                if self._session is None:
                    self._session = self._create_session()
        return self._session

    def _create_session(self):
        """Instantiate one ONNX runtime session."""

        try:
            import onnxruntime
        except ModuleNotFoundError as exc:
            msg = "onnxruntime is not installed. Run: pip install -r tongue_diagnosis/requirements.txt"
            raise RuntimeError(msg) from exc

        return onnxruntime.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )


def decode_image_bytes_to_bgr(image_bytes: bytes) -> np.ndarray | None:
    """Decode raw image bytes into an OpenCV BGR image."""

    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def _to_jsonable_dataclass(value):
    """Convert dataclass values, tuples, and lists into JSON-serializable objects."""

    if value is None:
        return None
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (tuple, list)):
        return [_to_jsonable_dataclass(item) for item in value]
    return value


def _build_expert_label_template() -> dict[str, object]:
    """Return the default expert-label template saved beside algorithm outputs."""

    return {
        "image_quality": {
            "usable": "",
            "reasons": [],
            "notes": "",
        },
        "segmentation_quality": {
            "label": "",
            "notes": "",
        },
        "tongue_color": {
            "overall": "",
            "local_observations": [],
            "confidence": "",
        },
        "moisture_tendency": {
            "label": "",
            "confidence": "",
        },
        "tongue_coat": {
            "visibility": "",
            "color_tendency": "",
            "thickness_tendency": "",
            "confidence": "",
        },
        "crack_observation": {
            "label": "",
            "confidence": "",
        },
        "influencing_factors": [],
        "expert_notes": "",
    }


def default_model_path() -> Path:
    """Return the default ONNX model path used by the standalone service."""

    return Path(__file__).resolve().parents[2] / "skin_alporithm" / "api" / "models" / "best_epoch_weights.pth.onnx"


def default_output_dir() -> Path:
    """Return the default output directory for saved segmentation files."""

    return Path(__file__).resolve().parents[1] / "outputs"
