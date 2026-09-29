"""Private-визуализация spine/hip: контуры, точки, ROI."""

from __future__ import annotations

import cv2
import numpy as np

from .labels import HIP_PARTS_LABELS_TO_IDS, REQUIRED_VERTEBRAE
from .results import DXAImageResult, HipFovRay, HipVisualization, SpineVisualization, XlsxClfPrediction

# Как в Label Studio.
VERTEBRA_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "Th12": (28, 26, 228),
    "L1": (184, 126, 55),
    "L2": (74, 175, 77),
    "L3": (163, 78, 152),
    "L4": (0, 127, 255),
    "L5": (40, 86, 166),
}
HIP_PART_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "greater_trochanter": (28, 26, 228),
    "femoral_neck": (184, 126, 55),
    "ischium": (74, 175, 77),
    "lesser_trochanter": (163, 78, 152),
    "femoral_head": (0, 127, 255),
    "femoral_shaft": (40, 86, 166),
}
BONE_COLOR_BGR = (189, 189, 189)
ILIAC_COLOR_BGR = (165, 194, 102)
FOREIGN_COLOR_BGR = (51, 255, 255)
AXIS_COLOR_BGR = (32, 128, 255)
ROI_COLOR_BGR = (0, 220, 255)
FOV_OK_BGR = (74, 175, 77)
FOV_BAD_BGR = (28, 26, 228)
HUD_COLOR_BGR = (240, 240, 240)
KP_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "lt_tip": (28, 26, 228),
    "shaft_medial_a": (184, 126, 55),
    "shaft_medial_b": (74, 175, 77),
}


def _overlay_mask(canvas: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float, thickness: int) -> None:
    if int(mask.sum()) == 0:
        return
    binary = (mask > 0).astype(np.uint8)
    overlay = canvas.copy()
    overlay[binary > 0] = color
    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, dst=canvas)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, color, thickness)


def _xlsx_clf_hud(xlsx_clf: XlsxClfPrediction | None) -> list[str]:
    if xlsx_clf is None:
        return []
    lines = [f"clf_q={xlsx_clf.quality_prob:.2f}"]
    lines.extend(f"clf_{name}={prob:.2f}" for name, prob in xlsx_clf.violation_probs.items())
    return lines


def _fov_color(distance_mm: float, margin_mm: float) -> tuple[int, int, int]:
    return FOV_OK_BGR if distance_mm >= margin_mm else FOV_BAD_BGR


def _draw_fov_ray(
    canvas: np.ndarray,
    ray: HipFovRay,
    margin_mm: float,
) -> None:
    color = _fov_color(ray.distance_mm, margin_mm)
    p0 = (int(round(ray.origin[0])), int(round(ray.origin[1])))
    p1 = (int(round(ray.edge[0])), int(round(ray.edge[1])))
    cv2.line(canvas, p0, p1, color, 2, cv2.LINE_AA)
    cv2.circle(canvas, p0, 4, color, -1, cv2.LINE_AA)
    mx = (p0[0] + p1[0]) // 2
    my = (p0[1] + p1[1]) // 2
    vertical = abs(p1[0] - p0[0]) < abs(p1[1] - p0[1])
    text_xy = (mx + 6, my) if vertical else (mx, max(12, my - 8))
    _put_label(canvas, f"{ray.distance_mm / 10.0:.1f} см", text_xy, color)


def project_point_to_line(
    tip: tuple[float, float],
    shaft_a: tuple[float, float],
    shaft_b: tuple[float, float],
) -> tuple[float, float] | None:
    """Основание перпендикуляра tip → прямая (a, b) в пикселях."""
    ax, ay = shaft_a
    bx, by = shaft_b
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return None
    t = ((tip[0] - ax) * dx + (tip[1] - ay) * dy) / length_sq
    return ax + t * dx, ay + t * dy


def protrusion_geometry(
    shaft_a: tuple[float, float],
    shaft_b: tuple[float, float],
    tip: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None:
    """Отрезок через A и B, продлённый к основанию перпендикуляра с LT. (start, end, foot)."""
    foot = project_point_to_line(tip, shaft_a, shaft_b)
    if foot is None:
        return None
    dx = shaft_b[0] - shaft_a[0]
    dy = shaft_b[1] - shaft_a[1]
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return None

    def param(point: tuple[float, float]) -> float:
        return ((point[0] - shaft_a[0]) * dx + (point[1] - shaft_a[1]) * dy) / length_sq

    ordered = sorted(
        ((param(shaft_a), shaft_a), (param(shaft_b), shaft_b), (param(foot), foot)),
        key=lambda item: item[0],
    )
    return ordered[0][1], ordered[-1][1], foot


def _draw_protrusion(
    canvas: np.ndarray,
    shaft_a: tuple[float, float],
    shaft_b: tuple[float, float],
    tip: tuple[float, float],
    protrusion_mm: float | None,
) -> None:
    geom = protrusion_geometry(shaft_a, shaft_b, tip)
    if geom is None:
        pa = (int(round(shaft_a[0])), int(round(shaft_a[1])))
        pb = (int(round(shaft_b[0])), int(round(shaft_b[1])))
        cv2.line(canvas, pa, pb, AXIS_COLOR_BGR, 1, cv2.LINE_AA)
        return
    start, end, foot = geom
    p0 = (int(round(start[0])), int(round(start[1])))
    p1 = (int(round(end[0])), int(round(end[1])))
    cv2.line(canvas, p0, p1, AXIS_COLOR_BGR, 1, cv2.LINE_AA)
    p_tip = (int(round(tip[0])), int(round(tip[1])))
    p_foot = (int(round(foot[0])), int(round(foot[1])))
    color = KP_COLORS_BGR["lt_tip"]
    cv2.line(canvas, p_tip, p_foot, color, 1, cv2.LINE_AA)
    cv2.circle(canvas, p_tip, 4, color, -1, cv2.LINE_AA)
    cv2.circle(canvas, p_foot, 3, color, -1, cv2.LINE_AA)
    if protrusion_mm is None:
        return
    _put_label(canvas, f"{protrusion_mm:.1f} мм", (p_foot[0] + 6, p_foot[1] - 8), color)


def _put_label(canvas: np.ndarray, text: str, xy: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = xy
    cv2.putText(canvas, text, (x, max(12, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, max(12, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def draw_spine_private_visualization(viz: SpineVisualization) -> np.ndarray:
    """BGR→RGB uint8 overlay."""
    canvas = cv2.cvtColor(viz.image_uint8, cv2.COLOR_GRAY2BGR)
    height, width = canvas.shape[:2]
    radius = max(3, width // 70)

    for name in REQUIRED_VERTEBRAE:
        mask = viz.vertebra_masks.get(name)
        if mask is None:
            continue
        color = VERTEBRA_COLORS_BGR[name]
        _overlay_mask(canvas, mask, color, alpha=0.28, thickness=1)
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            continue
        _put_label(canvas, name, (int(xs.min()), int(ys.min()) - 2), color)

    _overlay_mask(canvas, viz.iliac_mask, ILIAC_COLOR_BGR, alpha=0.25, thickness=1)
    if int(viz.iliac_mask.sum()) > 0:
        ys, xs = np.where(viz.iliac_mask > 0)
        _put_label(canvas, "iliac", (int(xs.min()), int(ys.min()) - 2), ILIAC_COLOR_BGR)

    _overlay_mask(canvas, viz.foreign_mask, FOREIGN_COLOR_BGR, alpha=0.45, thickness=2)
    if int(viz.foreign_mask.sum()) > 0:
        ys, xs = np.where(viz.foreign_mask > 0)
        _put_label(canvas, "foreign", (int(xs.min()), int(ys.min()) - 2), FOREIGN_COLOR_BGR)

    points = [viz.l1_xy, viz.l5_xy]
    if viz.l1_xy is not None and viz.l5_xy is not None:
        p1 = (int(round(viz.l1_xy[0])), int(round(viz.l1_xy[1])))
        p5 = (int(round(viz.l5_xy[0])), int(round(viz.l5_xy[1])))
        cv2.line(canvas, p1, p5, AXIS_COLOR_BGR, 1, cv2.LINE_AA)
    for point, label in zip(points, ("L1", "L5"), strict=True):
        if point is None:
            continue
        xy = (int(round(point[0])), int(round(point[1])))
        cv2.circle(canvas, xy, radius, AXIS_COLOR_BGR, -1, cv2.LINE_AA)
        cv2.circle(canvas, xy, radius, (0, 0, 0), 1, cv2.LINE_AA)
        _put_label(canvas, f"axis_{label}", (xy[0] + radius + 1, xy[1] - 2), AXIS_COLOR_BGR)

    hud = [
        f"quality={'bad' if viz.quality_class == 1 else 'good'}",
        f"angle={viz.axis_angle_deg:.2f} deg" if viz.axis_angle_deg is not None else "angle=n/a",
        f"t12_ratio={viz.t12_height_ratio:.2f}" if viz.t12_height_ratio is not None else "t12_ratio=n/a",
    ]
    hud.extend(_xlsx_clf_hud(viz.xlsx_clf))
    hud.extend(viz.violation_reasons)
    y = 14
    for line in hud:
        cv2.putText(canvas, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, HUD_COLOR_BGR, 1, cv2.LINE_AA)
        y += 12
        if y > height - 4:
            break

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def _draw_hud(canvas: np.ndarray, lines: list[str]) -> None:
    height = canvas.shape[0]
    y = 14
    for line in lines:
        cv2.putText(canvas, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, HUD_COLOR_BGR, 1, cv2.LINE_AA)
        y += 12
        if y > height - 4:
            break


def draw_hip_private_visualization(viz: HipVisualization) -> np.ndarray:
    canvas = cv2.cvtColor(viz.image_uint8, cv2.COLOR_GRAY2BGR)
    _height, width = canvas.shape[:2]
    radius = max(3, width // 70)

    _overlay_mask(canvas, viz.bone_mask, BONE_COLOR_BGR, alpha=0.18, thickness=1)
    for name in HIP_PARTS_LABELS_TO_IDS:
        mask = viz.part_masks.get(name)
        if mask is None:
            continue
        color = HIP_PART_COLORS_BGR[name]
        _overlay_mask(canvas, mask, color, alpha=0.28, thickness=1)
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            continue
        _put_label(canvas, name, (int(xs.min()), int(ys.min()) - 2), color)

    _overlay_mask(canvas, viz.prothesis_mask, FOREIGN_COLOR_BGR, alpha=0.45, thickness=2)
    if int(viz.prothesis_mask.sum()) > 0:
        ys, xs = np.where(viz.prothesis_mask > 0)
        _put_label(canvas, "prothesis", (int(xs.min()), int(ys.min()) - 2), FOREIGN_COLOR_BGR)

    if viz.anatomy_bbox is not None:
        x0, y0, x1, y1 = viz.anatomy_bbox
        cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), ROI_COLOR_BGR, 1, cv2.LINE_AA)
        _put_label(canvas, "roi", (x0, max(12, y0 - 2)), ROI_COLOR_BGR)

    if viz.fov_ischium is not None:
        _draw_fov_ray(canvas, viz.fov_ischium, viz.fov_vertical_margin_mm)
    if viz.fov_gt_up is not None:
        _draw_fov_ray(canvas, viz.fov_gt_up, viz.fov_vertical_margin_mm)
    if viz.fov_gt is not None:
        _draw_fov_ray(canvas, viz.fov_gt, viz.fov_lateral_margin_mm)

    if viz.shaft_axis is not None:
        proximal = (int(round(viz.shaft_axis.proximal[0])), int(round(viz.shaft_axis.proximal[1])))
        distal = (int(round(viz.shaft_axis.distal[0])), int(round(viz.shaft_axis.distal[1])))
        cv2.line(canvas, distal, proximal, AXIS_COLOR_BGR, 2, cv2.LINE_AA)
        for center in viz.shaft_axis.centers:
            xy = (int(round(center[0])), int(round(center[1])))
            cv2.circle(canvas, xy, 2, AXIS_COLOR_BGR, -1, cv2.LINE_AA)
        _put_label(canvas, f"shaft {viz.shaft_axis.angle_deg:.1f} deg", (proximal[0] + 4, proximal[1]), AXIS_COLOR_BGR)

    a, b, tip = viz.shaft_medial_a, viz.shaft_medial_b, viz.lt_tip
    if a is not None and b is not None and tip is not None:
        _draw_protrusion(canvas, a, b, tip, viz.protrusion_mm)
    elif a is not None and b is not None:
        pa = (int(round(a[0])), int(round(a[1])))
        pb = (int(round(b[0])), int(round(b[1])))
        cv2.line(canvas, pa, pb, AXIS_COLOR_BGR, 1, cv2.LINE_AA)
    for point, label in ((tip, "lt_tip"), (a, "shaft_medial_a"), (b, "shaft_medial_b")):
        if point is None:
            continue
        xy = (int(round(point[0])), int(round(point[1])))
        color = KP_COLORS_BGR[label]
        cv2.circle(canvas, xy, radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, xy, radius, (0, 0, 0), 1, cv2.LINE_AA)
        _put_label(canvas, label, (xy[0] + radius + 1, xy[1] - 2), color)

    hud = [
        f"quality={'bad' if viz.quality_class == 1 else 'good'}",
        f"rotation={viz.rotation_state}",
        f"protrusion={viz.protrusion_mm:.1f} mm" if viz.protrusion_mm is not None else "protrusion=n/a",
        f"side={viz.hip_side}" if viz.hip_side is not None else "side=n/a",
        f"shaft={viz.shaft_axis.angle_deg:.1f} deg" if viz.shaft_axis is not None else "shaft=n/a",
    ]
    hud.extend(_xlsx_clf_hud(viz.xlsx_clf))
    hud.extend(viz.violation_reasons)
    _draw_hud(canvas, hud)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def draw_private_visualization(run_result: DXAImageResult) -> np.ndarray:
    match run_result.visualization:
        case SpineVisualization():
            return draw_spine_private_visualization(run_result.visualization)
        case HipVisualization():
            return draw_hip_private_visualization(run_result.visualization)
        case None:
            raise ValueError("Нет visualization в результате")
        case _:
            raise TypeError(f"Неизвестный visualization: {type(run_result.visualization)}")
