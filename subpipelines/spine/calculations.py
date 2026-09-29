"""Геометрические правила QC позвоночника. Все чеки гоняются всегда."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ...labels import (
    L14_VERTEBRAE,
    REASON_AXIS,
    REASON_FOREIGN,
    REASON_ILIAC,
    REASON_MISSING_VERTEBRAE,
    REASON_SPINE_CENTER,
    REASON_T12_HEIGHT,
    REQUIRED_VERTEBRAE,
    closed_violations_from_reasons,
)
from ...results import SpineColumnAxis


BBox = tuple[int, int, int, int]
Point = tuple[float, float]


def _bbox_height(bbox: BBox) -> int:
    return bbox[3] - bbox[1]


COLUMN_AXIS_MIN_POINTS = 3


def vertebra_centroids(vertebra_masks: dict[str, np.ndarray]) -> list[Point]:
    """Центры масс позвонков сверху вниз, в порядке Th12…L5."""
    centers: list[Point] = []
    for name in REQUIRED_VERTEBRAE:
        mask = vertebra_masks.get(name)
        if mask is None or int(mask.sum()) == 0:
            continue
        ys, xs = np.where(mask > 0)
        centers.append((float(xs.mean()), float(ys.mean())))
    return centers


def column_axis_from_masks(
    vertebra_masks: dict[str, np.ndarray],
    *,
    image_width: int,
    pixel_size_x_mm: float,
    pixel_size_y_mm: float,
) -> SpineColumnAxis | None:
    """Прямая через центры позвонков. offset — сдвиг от вертикальной середины кадра, мм."""
    centers = vertebra_centroids(vertebra_masks)
    if len(centers) < COLUMN_AXIS_MIN_POINTS:
        return None
    y_px = np.array([point[1] for point in centers], dtype=np.float64)
    x_px = np.array([point[0] for point in centers], dtype=np.float64)
    y_mm = y_px * pixel_size_y_mm
    x_mm = x_px * pixel_size_x_mm
    y_centered = y_mm - y_mm.mean()
    denom = float(np.dot(y_centered, y_centered))
    if denom < 1e-6:
        return None
    slope = float(np.dot(y_centered, x_mm - x_mm.mean()) / denom)
    x_mean = float(x_mm.mean())
    y_mean = float(y_mm.mean())

    def x_at(y_row: float) -> float:
        x_m = x_mean + slope * (y_row * pixel_size_y_mm - y_mean)
        return x_m / pixel_size_x_mm

    y_mid = float(y_px.mean())
    offset_mm = abs(x_at(y_mid) - image_width / 2.0) * pixel_size_x_mm
    y_prox = float(y_px.min())
    y_dist = float(y_px.max())
    return SpineColumnAxis(
        proximal=(x_at(y_prox), y_prox),
        distal=(x_at(y_dist), y_dist),
        offset_mm=offset_mm,
        centers=centers,
    )


def axis_angle_deg(l1: Point, l5: Point) -> float:
    """Угол хорды L1–L5 к вертикали кадра, [0, 90]."""
    dx = l5[0] - l1[0]
    dy = l5[1] - l1[1]
    angle = abs(math.degrees(math.atan2(dx, dy)))
    if angle > 90.0:
        angle = 180.0 - angle
    return angle


@dataclass
class SpineRuleInput:
    vertebra_bboxes: dict[str, BBox | None]
    iliac_area_px: int
    foreign_area_px: int
    vertebra_masks: dict[str, np.ndarray]
    image_width: int
    l1_xy: Point | None
    l5_xy: Point | None
    axis_angle_threshold_deg: float
    t12_height_ratio_threshold: float
    center_offset_threshold_mm: float
    pixel_size_x_mm: float
    pixel_size_y_mm: float
    iliac_min_area_px: int
    foreign_min_area_px: int


@dataclass
class SpineRuleOutput:
    reasons: list[str] = field(default_factory=list)
    closed_violations: list[str] = field(default_factory=list)
    missing_vertebrae: list[str] = field(default_factory=list)
    iliac_present: bool = False
    axis_angle_deg: float | None = None
    t12_height_ratio: float | None = None
    column_axis: SpineColumnAxis | None = None
    foreign_area_px: int = 0


def apply_spine_rules(data: SpineRuleInput) -> SpineRuleOutput:
    reasons: list[str] = []

    missing = [name for name in REQUIRED_VERTEBRAE if data.vertebra_bboxes.get(name) is None]
    if missing:
        reasons.append(REASON_MISSING_VERTEBRAE)

    iliac_present = data.iliac_area_px >= data.iliac_min_area_px
    if not iliac_present:
        reasons.append(REASON_ILIAC)

    t12_bbox = data.vertebra_bboxes.get("Th12")
    l14_bboxes = [data.vertebra_bboxes.get(name) for name in L14_VERTEBRAE]
    t12_ratio: float | None = None
    if t12_bbox is not None and all(box is not None for box in l14_bboxes):
        mean_l14 = sum(_bbox_height(box) for box in l14_bboxes) / len(L14_VERTEBRAE)
        if mean_l14 > 0:
            t12_ratio = _bbox_height(t12_bbox) / mean_l14
            if t12_ratio < data.t12_height_ratio_threshold:
                reasons.append(REASON_T12_HEIGHT)

    angle: float | None = None
    if data.l1_xy is not None and data.l5_xy is not None:
        angle = axis_angle_deg(data.l1_xy, data.l5_xy)
        if angle > data.axis_angle_threshold_deg:
            reasons.append(REASON_AXIS)

    present_masks = {
        name: data.vertebra_masks[name]
        for name in REQUIRED_VERTEBRAE
        if data.vertebra_bboxes.get(name) is not None and name in data.vertebra_masks
    }
    column_axis = column_axis_from_masks(
        present_masks,
        image_width=data.image_width,
        pixel_size_x_mm=data.pixel_size_x_mm,
        pixel_size_y_mm=data.pixel_size_y_mm,
    )
    if column_axis is not None and column_axis.offset_mm > data.center_offset_threshold_mm:
        reasons.append(REASON_SPINE_CENTER)

    if data.foreign_area_px >= data.foreign_min_area_px:
        reasons.append(REASON_FOREIGN)

    closed = closed_violations_from_reasons(reasons, "spine")

    return SpineRuleOutput(
        reasons=reasons,
        closed_violations=closed,
        missing_vertebrae=missing,
        iliac_present=iliac_present,
        axis_angle_deg=angle,
        t12_height_ratio=t12_ratio,
        column_axis=column_axis,
        foreign_area_px=data.foreign_area_px,
    )
