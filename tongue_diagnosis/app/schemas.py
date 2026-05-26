"""Pydantic response models for the tongue diagnosis API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ImageSize(BaseModel):
    """Image width and height."""

    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)


class BoundingBox(BaseModel):
    """Bounding box for the detected tongue region."""

    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    w: int = Field(..., ge=1)
    h: int = Field(..., ge=1)


class SavedFile(BaseModel):
    """Metadata for one locally saved output file."""

    filename: str
    path: str


class RegionDivision(BaseModel):
    """Overlay region description for the segmented tongue image."""

    description: str
    labels: dict[str, str]


class SegmentationQuality(BaseModel):
    """Segmentation stability metrics for one tongue image."""

    tongue_area_ratio: float = Field(..., ge=0.0, le=1.0)
    bbox_touches_edge: bool
    region_coverage_ratios: dict[str, float] = Field(default_factory=dict)


class RgbColor(BaseModel):
    """One RGB color triplet."""

    r: int = Field(..., ge=0, le=255)
    g: int = Field(..., ge=0, le=255)
    b: int = Field(..., ge=0, le=255)


class HsvColor(BaseModel):
    """One HSV color triplet in OpenCV channel ranges."""

    h: int = Field(..., ge=0, le=179)
    s: int = Field(..., ge=0, le=255)
    v: int = Field(..., ge=0, le=255)


class LabColor(BaseModel):
    """One CIELAB color triplet."""

    l: float = Field(..., ge=0.0, le=100.0)
    a: float
    b: float


class MoistureScoreBreakdown(BaseModel):
    """Rule-based score breakdown for one moisture result."""

    gloss_area_score: float = Field(..., ge=0.0, le=100.0)
    highlight_blob_score: float = Field(..., ge=0.0, le=100.0)
    crack_score_inverse: float = Field(..., ge=0.0, le=100.0)
    moisture_score: float = Field(..., ge=0.0, le=100.0)


class RegionColor(BaseModel):
    """Representative color metrics for one tongue region."""

    region_id: str
    region_name: str
    pixel_count: int = Field(..., ge=0)
    coverage_ratio: float = Field(..., ge=0.0, le=1.0)
    representative_rgb: RgbColor
    representative_hsv: HsvColor
    mean_lab: LabColor
    representative_hex: str
    color_name: str


class TongueCoat(BaseModel):
    """Whole-tongue coat candidate observation."""

    coat_visibility: str
    coat_coverage_ratio: float = Field(..., ge=0.0, le=1.0)
    coat_color_tendency: str
    coat_thickness_tendency: str
    white_coat_ratio: float = Field(..., ge=0.0, le=1.0)
    yellow_coat_ratio: float = Field(..., ge=0.0, le=1.0)
    gray_black_coat_ratio: float = Field(..., ge=0.0, le=1.0)


class RegionCoat(BaseModel):
    """Region-level coat candidate observation."""

    region_id: str
    region_name: str
    pixel_count: int = Field(..., ge=0)
    coverage_ratio: float = Field(..., ge=0.0, le=1.0)
    coat_coverage_ratio: float = Field(..., ge=0.0, le=1.0)
    coat_color_tendency: str
    coat_thickness_tendency: str


class CrackObservation(BaseModel):
    """Standalone crack candidate observation."""

    crack_level: str
    crack_length_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    crack_area_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: str


class RegionMoisture(BaseModel):
    """Representative moisture metrics for one tongue region."""

    region_id: str
    region_name: str
    pixel_count: int = Field(..., ge=0)
    coverage_ratio: float = Field(..., ge=0.0, le=1.0)
    gloss_area_ratio: float = Field(..., ge=0.0, le=1.0)
    highlight_blob_count: int = Field(..., ge=0)
    highlight_blob_max_ratio: float = Field(..., ge=0.0, le=1.0)
    highlight_blob_mean_area: float = Field(..., ge=0.0)
    crack_length_ratio: float = Field(..., ge=0.0, le=1.0)
    crack_area_ratio: float = Field(..., ge=0.0, le=1.0)
    overexposed_ratio: float = Field(..., ge=0.0, le=1.0)
    moisture_score: float = Field(..., ge=0.0, le=100.0)
    moisture_label: str
    moisture_tendency: str
    moisture_explanation: str
    score_breakdown: MoistureScoreBreakdown


class TongueMoisture(BaseModel):
    """Image-level tongue moisture assessment."""

    moisture_score: float = Field(..., ge=0.0, le=100.0)
    moisture_label: str
    moisture_tendency: str
    moisture_explanation: str
    quality_passed: bool
    quality_reasons: list[str] = Field(default_factory=list)
    quality_reasons_zh: list[str] = Field(default_factory=list)
    focus_score: float = Field(..., ge=0.0)
    overexposed_ratio: float = Field(..., ge=0.0, le=1.0)
    segmentation_coverage: float = Field(..., ge=0.0, le=1.0)
    gloss_area_ratio: float = Field(..., ge=0.0, le=1.0)
    highlight_blob_count: int = Field(..., ge=0)
    highlight_blob_max_ratio: float = Field(..., ge=0.0, le=1.0)
    highlight_blob_mean_area: float = Field(..., ge=0.0)
    crack_length_ratio: float = Field(..., ge=0.0, le=1.0)
    crack_area_ratio: float = Field(..., ge=0.0, le=1.0)
    score_breakdown: MoistureScoreBreakdown


class MoistureVisualizations(BaseModel):
    """Optional base64-encoded moisture debug visualizations."""

    gloss_mask_base64: str
    gloss_overlay_base64: str
    crack_mask_base64: str
    crack_skeleton_base64: str
    crack_overlay_base64: str
    overexposed_mask_base64: str
    overexposed_overlay_base64: str
    moisture_heatmap_base64: str
    score_breakdown_image_base64: str


class AssistanceQuality(BaseModel):
    """Quality summary for product-facing tongue image assistance."""

    passed: bool
    level: str
    reasons: list[str] = Field(default_factory=list)
    suggestion: str


class AssistanceSummary(BaseModel):
    """Product-facing observation summary for one tongue image."""

    color_tendency: str
    moisture_tendency: str
    coat_visibility: str
    main_observations: list[str] = Field(default_factory=list)


class AssistanceEvidence(BaseModel):
    """Evidence metrics used by the product-facing assistance summary."""

    tongue_area_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    bbox_touches_edge: bool | None = None
    focus_score: float | None = Field(default=None, ge=0.0)
    overexposed_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    segmentation_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    gloss_area_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    crack_length_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    coat_coverage_ratio: float | None = Field(default=None, ge=0.0, le=1.0)


class TongueImageAssistance(BaseModel):
    """Product-facing health-assistance summary for one tongue image."""

    positioning: str
    quality: AssistanceQuality
    summary: AssistanceSummary
    evidence: AssistanceEvidence
    disclaimer: str


class DemoQualityGate(BaseModel):
    """Quality gate used by the front-end demo report."""

    passed: bool
    level: str
    reasons: list[str] = Field(default_factory=list)
    suggestion: str


class DemoFeatureCard(BaseModel):
    """One front-end display metric for the demo report."""

    key: str
    title: str
    value: str
    level: str
    evidence: list[str] = Field(default_factory=list)


class DemoConstitutionTendency(BaseModel):
    """One nine-constitution tendency candidate for front-end display."""

    constitution: str
    score: float = Field(..., ge=0.0, le=100.0)
    level: str
    evidence: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    confidence: str
    note: str


class DemoFrontendSection(BaseModel):
    """Suggested section metadata for the front-end demo page."""

    key: str
    title: str
    items: list[str] = Field(default_factory=list)


class DemoReport(BaseModel):
    """Front-end-oriented demo report built from current image features."""

    report_type: str
    quality_gate: DemoQualityGate
    feature_cards: list[DemoFeatureCard] = Field(default_factory=list)
    constitution_tendencies: list[DemoConstitutionTendency] = Field(default_factory=list)
    primary_tendencies: list[str] = Field(default_factory=list)
    analysis_summary: str
    frontend_sections: list[DemoFrontendSection] = Field(default_factory=list)
    disclaimer: str


class SegmentResponse(BaseModel):
    """Response returned by the tongue segmentation endpoint."""

    status: str
    message: str
    file_name: str
    image_size: ImageSize
    bounding_box: BoundingBox
    region_division: RegionDivision
    segmentation_quality: SegmentationQuality | None = None
    tongue_color: RegionColor | None = None
    region_colors: list[RegionColor] = Field(default_factory=list)
    tongue_coat: TongueCoat | None = None
    region_coat: list[RegionCoat] = Field(default_factory=list)
    region_moisture: list[RegionMoisture] = Field(default_factory=list)
    tongue_moisture: TongueMoisture | None = None
    crack_observation: CrackObservation | None = None
    tongue_image_assistance: TongueImageAssistance | None = None
    demo_report: DemoReport | None = None
    moisture_visualizations: MoistureVisualizations | None = None
    segmented_image_base64: str | None = None
    mask_image_base64: str | None = None
    saved_files: list[SavedFile] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Health-check response for the FastAPI service."""

    status: str
    service: str
    model_loaded: bool
    model_path: str
