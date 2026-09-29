"""Геометрические правила QC ТБС. Все чеки гоняются всегда."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ...labels import (
    HIP_POSITIONING_PARTS,
    REASON_FOV_GT,
    REASON_FOV_GT_UP,
    REASON_FOV_ISCHIUM,
    REASON_MISSING_HIP_PARTS,
    REASON_OVER_ROTATION,
    REASON_PROTHESIS,
    REASON_SHAFT_AXIS,
    REASON_UNDER_ROTATION,
    HipSide,
    RotationState,
    closed_violations_from_reasons,
)
from ...results import HipFovRay, HipShaftAxis

BBox = tuple[int, int, int, int]
Point = tuple[float, float]
ExtremeMode = Literal["top", "bottom", "left", "right"]


def point_line_distance_mm(
    tip: Point,
    shaft_a: Point,
    shaft_b: Point,
    pixel_size_x_mm: float,
    pixel_size_y_mm: float,
) -> float:
    """Расстояние lt_tip → прямая (a, b) в мм, анизотропный пиксель Lunar."""
    ax, ay = shaft_a[0] * pixel_size_x_mm, shaft_a[1] * pixel_size_y_mm
    bx, by = shaft_b[0] * pixel_size_x_mm, shaft_b[1] * pixel_size_y_mm
    tx, ty = tip[0] * pixel_size_x_mm, tip[1] * pixel_size_y_mm
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return 0.0
    return abs((tx - ax) * dy - (ty - ay) * dx) / length


def mask_centroid_x(mask: np.ndarray) -> float | None:
    _ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return float(xs.mean())


def mask_extreme_point(mask: np.ndarray, mode: ExtremeMode) -> Point | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    match mode:
        case "bottom":
            y = int(ys.max())
            return float(np.median(xs[ys == y])), float(y)
        case "top":
            y = int(ys.min())
            return float(np.median(xs[ys == y])), float(y)
        case "left":
            x = int(xs.min())
            return float(x), float(np.median(ys[xs == x]))
        case "right":
            x = int(xs.max())
            return float(x), float(np.median(ys[xs == x]))


def infer_hip_side(
    gt_mask: np.ndarray,
    head_mask: np.ndarray,
    ischium_mask: np.ndarray,
    image_width: int,
) -> HipSide | None:
    """GT латерален относительно головки/седалищной. Нет медиального якоря → центр кадра."""
    gt_x = mask_centroid_x(gt_mask)
    if gt_x is None:
        return None
    medial_x = mask_centroid_x(head_mask)
    if medial_x is None:
        medial_x = mask_centroid_x(ischium_mask)
    if medial_x is None:
        return "left" if gt_x < image_width / 2 else "right"
    return "left" if gt_x < medial_x else "right"


SnapMode = Literal["left", "right", "nearest"]


def snap_x_to_mask_row(
    point: Point | None,
    mask: np.ndarray,
    mode: SnapMode,
) -> Point | None:
    """Горизонтальный snap. None, если на этой Y-строке маски нет."""
    if point is None:
        return None
    height, _width = mask.shape[:2]
    y = int(round(point[1]))
    if y < 0 or y >= height:
        return None
    xs = np.flatnonzero(mask[y] > 0)
    if xs.size == 0:
        return None
    match mode:
        case "left":
            x = float(xs.min())
        case "right":
            x = float(xs.max())
        case "nearest":
            x_left, x_right = float(xs.min()), float(xs.max())
            x0 = float(point[0])
            x = x_left if abs(x0 - x_left) <= abs(x0 - x_right) else x_right
    return x, float(y)


def snap_lt_tip_to_edge(
    point: Point | None,
    lt_mask: np.ndarray,
    bone_mask: np.ndarray,
    hip_side: HipSide | None,
) -> Point | None:
    """Клюв: сначала медиальный край LT на строке, нет пересечения → ближайший край bone."""
    match hip_side:
        case "left":
            lt_mode: SnapMode = "right"
        case "right":
            lt_mode = "left"
        case None:
            lt_mode = "nearest"
    snapped = snap_x_to_mask_row(point, lt_mask, lt_mode)
    if snapped is not None:
        return snapped
    bone = snap_x_to_mask_row(point, bone_mask, "nearest")
    return bone if bone is not None else point


def snap_point_to_bone_edge(
    point: Point | None,
    bone_mask: np.ndarray,
) -> Point | None:
    """Горизонтально к ближайшему из двух краёв bone на той же строке. Нет кости — как была."""
    snapped = snap_x_to_mask_row(point, bone_mask, "nearest")
    return snapped if snapped is not None else point


def measure_fov_rays(
    *,
    ischium_mask: np.ndarray,
    head_mask: np.ndarray,
    gt_mask: np.ndarray,
    image_size: tuple[int, int],
    pixel_size_x_mm: float,
    pixel_size_y_mm: float,
) -> tuple[HipSide | None, HipFovRay | None, HipFovRay | None, HipFovRay | None]:
    height, width = image_size
    hip_side = infer_hip_side(gt_mask, head_mask, ischium_mask, width)

    ischium_ray: HipFovRay | None = None
    origin = mask_extreme_point(ischium_mask, "bottom")
    if origin is not None:
        edge = (origin[0], float(height - 1))
        ischium_ray = HipFovRay(
            origin=origin,
            edge=edge,
            distance_mm=(edge[1] - origin[1]) * pixel_size_y_mm,
        )

    gt_up_ray: HipFovRay | None = None
    origin = mask_extreme_point(gt_mask, "top")
    if origin is not None:
        edge = (origin[0], 0.0)
        gt_up_ray = HipFovRay(
            origin=origin,
            edge=edge,
            distance_mm=origin[1] * pixel_size_y_mm,
        )

    gt_ray: HipFovRay | None = None
    match hip_side:
        case "left":
            origin = mask_extreme_point(gt_mask, "left")
            if origin is not None:
                edge = (0.0, origin[1])
                gt_ray = HipFovRay(
                    origin=origin,
                    edge=edge,
                    distance_mm=origin[0] * pixel_size_x_mm,
                )
        case "right":
            origin = mask_extreme_point(gt_mask, "right")
            if origin is not None:
                edge = (float(width - 1), origin[1])
                gt_ray = HipFovRay(
                    origin=origin,
                    edge=edge,
                    distance_mm=(width - 1 - origin[0]) * pixel_size_x_mm,
                )
        case None:
            pass

    return hip_side, ischium_ray, gt_up_ray, gt_ray


SHAFT_AXIS_MIN_POINTS = 3
SHAFT_AXIS_MIN_RUN_PX = 4


def _largest_run_center_x(row: np.ndarray, min_width: int) -> float | None:
    """Центр самого длинного непрерывного пересечения строки с маской."""
    xs = np.flatnonzero(row > 0)
    if xs.size == 0:
        return None
    runs = np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1)
    run = max(runs, key=lambda item: int(item.size))
    if int(run.size) < min_width:
        return None
    return float(run[0] + run[-1]) / 2.0


def shaft_axis_from_mask(
    mask: np.ndarray,
    *,
    pixel_size_x_mm: float,
    pixel_size_y_mm: float,
    proximal_cut: float,
) -> HipShaftAxis | None:
    """Ось диафиза по центрам горизонтальных хорд.

    Сэмплируем снизу вверх. Угол — к вертикали скана в мм, не в пикселях:
    шаг Lunar 0.6×1.05, пиксельный atan2 завышает наклон.
    """
    ys = np.flatnonzero(mask.any(axis=1))
    if ys.size == 0:
        return None
    y_top = int(ys[0])
    y_bottom = int(ys[-1])
    span = y_bottom - y_top
    if span < 8:
        return None
    y_limit = y_top + int(round(proximal_cut * span))
    used = y_bottom - y_limit
    step = max(4, used // 12)
    centers: list[Point] = []
    y = y_bottom
    while y >= y_limit:
        x = _largest_run_center_x(mask[y], SHAFT_AXIS_MIN_RUN_PX)
        if x is not None:
            centers.append((x, float(y)))
        y -= step
    if len(centers) < SHAFT_AXIS_MIN_POINTS:
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
    angle = abs(math.degrees(math.atan(slope)))
    x_mean = float(x_mm.mean())
    y_mean = float(y_mm.mean())

    def x_at(y_row: float) -> float:
        x_m = x_mean + slope * (y_row * pixel_size_y_mm - y_mean)
        return x_m / pixel_size_x_mm

    y_dist = float(y_px.max())
    y_prox = float(y_px.min())
    return HipShaftAxis(
        proximal=(x_at(y_prox), y_prox),
        distal=(x_at(y_dist), y_dist),
        angle_deg=angle,
        centers=centers,
    )


@dataclass
class HipRuleInput:
    part_present: dict[str, bool]
    lt_present: bool
    prothesis_area_px: int
    image_size: tuple[int, int]
    anatomy_bbox: BBox | None
    ischium_mask: np.ndarray
    head_mask: np.ndarray
    gt_mask: np.ndarray
    shaft_mask: np.ndarray
    shaft_axis_angle_threshold_deg: float
    shaft_axis_proximal_cut: float
    lt_tip: Point | None
    shaft_medial_a: Point | None
    shaft_medial_b: Point | None
    pixel_size_x_mm: float
    pixel_size_y_mm: float
    roi_margin_x_mm: float
    roi_margin_y_mm: float
    under_rotation_min_mm: float
    prothesis_min_area_px: int


@dataclass
class HipRuleOutput:
    reasons: list[str] = field(default_factory=list)
    closed_violations: list[str] = field(default_factory=list)
    missing_parts: list[str] = field(default_factory=list)
    rotation_state: RotationState = "ok"
    protrusion_mm: float | None = None
    hip_side: HipSide | None = None
    fov_ischium: HipFovRay | None = None
    fov_gt_up: HipFovRay | None = None
    fov_gt: HipFovRay | None = None
    shaft_axis: HipShaftAxis | None = None
    roi_margins_px: tuple[float, float, float, float] | None = None
    roi_ok: bool = True
    prothesis_present: bool = False


def apply_hip_rules(data: HipRuleInput) -> HipRuleOutput:
    reasons: list[str] = []

    missing = [name for name in HIP_POSITIONING_PARTS if not data.part_present.get(name, False)]
    if missing:
        reasons.append(REASON_MISSING_HIP_PARTS)

    protrusion: float | None = None
    if data.lt_tip is not None and data.shaft_medial_a is not None and data.shaft_medial_b is not None:
        protrusion = point_line_distance_mm(
            data.lt_tip,
            data.shaft_medial_a,
            data.shaft_medial_b,
            data.pixel_size_x_mm,
            data.pixel_size_y_mm,
        )

    rotation_state: RotationState = "ok"
    if not data.lt_present:
        rotation_state = "over"
        reasons.append(REASON_OVER_ROTATION)
    elif protrusion is not None and protrusion > data.under_rotation_min_mm:
        rotation_state = "under"
        reasons.append(REASON_UNDER_ROTATION)

    hip_side, fov_ischium, fov_gt_up, fov_gt = measure_fov_rays(
        ischium_mask=data.ischium_mask,
        head_mask=data.head_mask,
        gt_mask=data.gt_mask,
        image_size=data.image_size,
        pixel_size_x_mm=data.pixel_size_x_mm,
        pixel_size_y_mm=data.pixel_size_y_mm,
    )
    if fov_ischium is not None and fov_ischium.distance_mm < data.roi_margin_y_mm:
        reasons.append(REASON_FOV_ISCHIUM)
    if fov_gt_up is not None and fov_gt_up.distance_mm < data.roi_margin_y_mm:
        reasons.append(REASON_FOV_GT_UP)
    if fov_gt is not None and fov_gt.distance_mm < data.roi_margin_x_mm:
        reasons.append(REASON_FOV_GT)
    roi_ok = not (REASON_FOV_ISCHIUM in reasons or REASON_FOV_GT_UP in reasons or REASON_FOV_GT in reasons)

    shaft_axis = shaft_axis_from_mask(
        data.shaft_mask,
        pixel_size_x_mm=data.pixel_size_x_mm,
        pixel_size_y_mm=data.pixel_size_y_mm,
        proximal_cut=data.shaft_axis_proximal_cut,
    )
    if shaft_axis is not None and shaft_axis.angle_deg > data.shaft_axis_angle_threshold_deg:
        reasons.append(REASON_SHAFT_AXIS)

    height, width = data.image_size
    margins: tuple[float, float, float, float] | None = None
    if data.anatomy_bbox is not None:
        x0, y0, x1, y1 = data.anatomy_bbox
        margins = (float(x0), float(y0), float(width - x1), float(height - y1))

    prothesis_present = data.prothesis_area_px >= data.prothesis_min_area_px
    if prothesis_present:
        reasons.append(REASON_PROTHESIS)

    return HipRuleOutput(
        reasons=reasons,
        closed_violations=closed_violations_from_reasons(reasons, "hip"),
        missing_parts=missing,
        rotation_state=rotation_state,
        protrusion_mm=protrusion,
        hip_side=hip_side,
        fov_ischium=fov_ischium,
        fov_gt_up=fov_gt_up,
        fov_gt=fov_gt,
        shaft_axis=shaft_axis,
        roi_margins_px=margins,
        roi_ok=roi_ok,
        prothesis_present=prothesis_present,
    )
