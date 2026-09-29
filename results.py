"""Результат одного DXA-кадра."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from .labels import HipSide, RegionName, RotationState

QualityClass = Literal[0, 1]
ProcessingStatus = Literal["Success", "Failure"]
Point = tuple[float, float]
BBox = tuple[int, int, int, int]

TZ_COLUMNS: tuple[str, ...] = (
    "path_to_study",
    "study_uid",
    "image_uid",
    "anatomical_region",
    "quality_class",
    "violation_type",
    "processing_status",
    "time_of_processing",
    "quality_prob",
)


class SpineColumnAxis(BaseModel, extra="forbid"):
    """Ось по центрам позвонков и её сдвиг от вертикальной середины кадра."""

    proximal: Point
    distal: Point
    offset_mm: float
    centers: list[Point] = Field(default_factory=list)


class XlsxClfPrediction(BaseModel, extra="forbid"):
    """Выход xlsx-классификатора. В решения QC пока не участвует."""

    quality_prob: float
    violation_probs: dict[str, float] = Field(default_factory=dict)


class SpineVisualization(BaseModel, extra="forbid", arbitrary_types_allowed=True):
    image_uint8: np.ndarray
    vertebra_masks: dict[str, np.ndarray]
    iliac_mask: np.ndarray
    foreign_mask: np.ndarray
    l1_xy: Point | None = None
    l5_xy: Point | None = None
    axis_angle_deg: float | None = None
    axis_angle_threshold_deg: float = 5.0
    t12_height_ratio: float | None = None
    t12_height_ratio_threshold: float = 0.5
    column_axis: SpineColumnAxis | None = None
    column_center_threshold_mm: float = 15.0
    quality_class: QualityClass
    violation_reasons: list[str] = Field(default_factory=list)
    xlsx_clf: XlsxClfPrediction | None = None


class HipFovRay(BaseModel, extra="forbid"):
    origin: Point
    edge: Point
    distance_mm: float


class HipShaftAxis(BaseModel, extra="forbid"):
    """Ось диафиза: концы аппроксимации и центры горизонтальных хорд."""

    proximal: Point
    distal: Point
    angle_deg: float
    centers: list[Point] = Field(default_factory=list)


class HipVisualization(BaseModel, extra="forbid", arbitrary_types_allowed=True):
    image_uint8: np.ndarray
    part_masks: dict[str, np.ndarray]
    bone_mask: np.ndarray
    prothesis_mask: np.ndarray
    lt_tip: Point | None = None
    shaft_medial_a: Point | None = None
    shaft_medial_b: Point | None = None
    anatomy_bbox: BBox | None = None
    protrusion_mm: float | None = None
    rotation_state: RotationState = "ok"
    hip_side: HipSide | None = None
    fov_ischium: HipFovRay | None = None
    fov_gt_up: HipFovRay | None = None
    fov_gt: HipFovRay | None = None
    shaft_axis: HipShaftAxis | None = None
    shaft_axis_threshold_deg: float = 5.0
    fov_vertical_margin_mm: float = 30.0
    fov_lateral_margin_mm: float = 20.0
    quality_class: QualityClass
    violation_reasons: list[str] = Field(default_factory=list)
    xlsx_clf: XlsxClfPrediction | None = None


class DXATzResult(BaseModel, extra="forbid"):
    """Одна строка итоговой таблицы: колонки ТЗ 2.5 + quality_prob (разъяснение, вопрос 8)."""

    path_to_study: str
    study_uid: str
    image_uid: str
    anatomical_region: str
    quality_class: QualityClass
    violation_type: str
    processing_status: ProcessingStatus
    time_of_processing: float
    quality_prob: float

    def to_row(self) -> dict[str, str | int | float]:
        return {name: getattr(self, name) for name in TZ_COLUMNS}


class DXAImageResult(BaseModel, arbitrary_types_allowed=True):
    anatomical_region: str
    quality_class: QualityClass
    violation_type: str
    quality_prob: float
    region_name: RegionName | None
    region_confidence: float
    path_to_study: str = ""
    study_uid: str = ""
    image_uid: str = ""
    processing_status: ProcessingStatus = "Success"
    time_of_processing: float = 0.0
    violation_reasons: list[str] = Field(default_factory=list)
    xlsx_clf: XlsxClfPrediction | None = None
    visualization: SpineVisualization | HipVisualization | None = Field(default=None, exclude=True)


class SpineSubPipelineResult(BaseModel, arbitrary_types_allowed=True):
    quality_class: QualityClass
    violation_type: str
    quality_prob: float
    violation_reasons: list[str] = Field(default_factory=list)
    missing_vertebrae: list[str] = Field(default_factory=list)
    iliac_present: bool | None = None
    axis_angle_deg: float | None = None
    t12_height_ratio: float | None = None
    column_offset_mm: float | None = None
    foreign_area_px: int | None = None
    l1_xy: Point | None = None
    l5_xy: Point | None = None
    xlsx_clf: XlsxClfPrediction | None = None
    visualization: SpineVisualization | None = Field(default=None, exclude=True)


class HipSubPipelineResult(BaseModel, arbitrary_types_allowed=True):
    quality_class: QualityClass
    violation_type: str
    quality_prob: float
    violation_reasons: list[str] = Field(default_factory=list)
    missing_parts: list[str] = Field(default_factory=list)
    rotation_state: RotationState = "ok"
    protrusion_mm: float | None = None
    hip_side: HipSide | None = None
    fov_ischium_mm: float | None = None
    fov_gt_up_mm: float | None = None
    fov_gt_mm: float | None = None
    shaft_axis_angle_deg: float | None = None
    roi_ok: bool = True
    roi_margins_px: tuple[float, float, float, float] | None = None
    prothesis_present: bool = False
    lt_tip: Point | None = None
    shaft_medial_a: Point | None = None
    shaft_medial_b: Point | None = None
    xlsx_clf: XlsxClfPrediction | None = None
    visualization: HipVisualization | None = Field(default=None, exclude=True)
