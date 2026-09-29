"""Публичная визуализация: снимок с доказательствами вердикта + сайдбар.

Анатомические маски и кейпоинты сюда не выводятся — они только в приватной
визуализации (`visualizations.py`). Здесь остаётся ровно то, что объясняет
вердикт: измерения с подписями и посторонние объекты.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .labels import (
    REGION_VIOLATIONS,
    VIOLATION_AXIS,
    VIOLATION_FOREIGN,
    VIOLATION_LAYOUT,
    VIOLATION_ROI,
    closed_violations_from_reasons,
)
from .results import DXAImageResult, HipFovRay, HipVisualization, SpineVisualization
from .visualizations import (
    AXIS_COLOR_BGR,
    FOREIGN_COLOR_BGR,
    FOV_BAD_BGR,
    FOV_OK_BGR,
    KP_COLORS_BGR,
    ROI_COLOR_BGR,
    protrusion_geometry,
)

ASSETS_DIR = Path(__file__).resolve(strict=True).parent / "assets"
MIN_VIS_HEIGHT = 1200
SERVICE_NAME = "Контроль качества DXA"
SERVICE_VERSION = "0.1"
FOR_DECISION_SUPPORT = "Для поддержки\nпринятия решений"

COLOR_RED = (216, 68, 18)
COLOR_GREEN = (139, 195, 74)
TEXT_WHITE = (255, 255, 255)
TEXT_MUTED = (168, 168, 168)
REFERENCE_GRAY = (210, 210, 210)
DIVIDER_GRAY = (58, 58, 58)
PANEL_BG = (0, 0, 0)

ROTATION_RU: dict[str, str] = {
    "ok": "норма",
    "over": "переротация",
    "under": "недоротация",
}

SIDEBAR_VIOLATION_KEYS: dict[str, str] = {
    VIOLATION_LAYOUT: "Правильная укладка",
    VIOLATION_AXIS: "Смещение оси",
    VIOLATION_FOREIGN: "Посторонние предметы",
    VIOLATION_ROI: "Правильный ROI",
}

# «да» = проблема есть. Остальные строки — вопрос «всё ли правильно».
SIDEBAR_YES_MEANS_BAD: frozenset[str] = frozenset({VIOLATION_AXIS, VIOLATION_FOREIGN})


@dataclass(frozen=True)
class SidebarRow:
    key: str
    value: str
    ball: tuple[int, int, int] | None = None
    value_color: tuple[int, int, int] = TEXT_WHITE


@dataclass(frozen=True)
class VisParams:
    height: int
    width: int
    m: float
    padding: int
    line_step: int
    medium_size: int
    bold_size: int
    font_medium: ImageFont.FreeTypeFont
    font_bold: ImageFont.FreeTypeFont
    font_label: ImageFont.FreeTypeFont


def _bgr_to_rgb(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return color[2], color[1], color[0]


def _load_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def _build_params(width: int, height: int) -> VisParams:
    m = height / 1500.0
    medium_size = max(16, int(22 * m))
    bold_size = max(20, int(30 * m))
    label_size = max(14, int(18 * m))
    return VisParams(
        height=height,
        width=width,
        m=m,
        padding=int(430 * m),
        line_step=medium_size + max(8, int(15 * m)),
        medium_size=medium_size,
        bold_size=bold_size,
        font_medium=_load_font(ASSETS_DIR / "Montserrat-Medium.ttf", medium_size),
        font_bold=_load_font(ASSETS_DIR / "Montserrat-Bold.ttf", bold_size),
        font_label=_load_font(ASSETS_DIR / "Montserrat-Bold.ttf", label_size),
    )


def _scale_hw(height: int, width: int) -> tuple[float, int, int]:
    if height >= MIN_VIS_HEIGHT:
        return 1.0, width, height
    scale = MIN_VIS_HEIGHT / height
    return scale, int(round(width * scale)), int(round(height * scale))


def _resize_gray(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, size_wh, interpolation=cv2.INTER_LINEAR)


def _resize_mask(mask: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), size_wh, interpolation=cv2.INTER_NEAREST)


def _scale_point(point: tuple[float, float] | None, scale: float) -> tuple[float, float] | None:
    if point is None:
        return None
    return point[0] * scale, point[1] * scale


def _scale_fov_ray(ray: HipFovRay | None, scale: float) -> HipFovRay | None:
    if ray is None:
        return None
    origin = _scale_point(ray.origin, scale)
    edge = _scale_point(ray.edge, scale)
    if origin is None or edge is None:
        return None
    return HipFovRay(origin=origin, edge=edge, distance_mm=ray.distance_mm)


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: float,
) -> list[str]:
    """Перенос по словам; слово шире колонки остаётся на своей строке как есть."""
    words = text.split()
    match words:
        case []:
            return [text]
        case [first, *rest]:
            lines: list[str] = []
            current = first
            for word in rest:
                candidate = f"{current} {word}"
                if draw.textlength(candidate, font=font) <= max_width:
                    current = candidate
                    continue
                lines.append(current)
                current = word
            lines.append(current)
            return lines


def _ellipsize(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: float,
) -> str:
    if draw.textlength(text, font=font) <= max_width:
        return text
    cut = text
    while cut and draw.textlength(f"{cut}…", font=font) > max_width:
        cut = cut[:-1]
    return f"{cut}…"


def _blend_mask(rgba: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    if int(mask.sum()) == 0:
        return
    sel = mask > 0
    src = rgba[sel, :3].astype(np.float32)
    tint = np.asarray(color, dtype=np.float32)
    rgba[sel, :3] = np.clip(src * (1.0 - alpha) + tint * alpha, 0, 255).astype(np.uint8)


def _draw_contours(draw: ImageDraw.ImageDraw, mask: np.ndarray, color: tuple[int, int, int], width: int) -> None:
    if int(mask.sum()) == 0:
        return
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        pts = [tuple(int(v) for v in point[0]) for point in contour]
        if len(pts) < 2:
            continue
        draw.line(pts + [pts[0]], fill=color, width=width)


def _mask_label_xy(mask: np.ndarray) -> tuple[int, int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return int(xs.min()), max(12, int(ys.min()) - 2)


def _draw_chip(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    outline: tuple[int, int, int] = TEXT_WHITE,
) -> None:
    x, y = xy
    pad = 6
    box = draw.textbbox((x, y), text, font=font)
    rect = [box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad]
    img_w, img_h = draw.im.size
    shift_x = 0 if rect[0] >= 0 else -rect[0]
    shift_x = shift_x if rect[2] + shift_x <= img_w else img_w - rect[2]
    shift_y = 0 if rect[1] >= 0 else -rect[1]
    shift_y = shift_y if rect[3] + shift_y <= img_h else img_h - rect[3]
    rect = (rect[0] + shift_x, rect[1] + shift_y, rect[2] + shift_x, rect[3] + shift_y)
    draw.rounded_rectangle(rect, radius=6, fill=(0, 0, 0), outline=outline, width=2)
    draw.text((x + shift_x, y + shift_y), text, font=font, fill=TEXT_WHITE)


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    p0: tuple[int, int],
    p1: tuple[int, int],
    color: tuple[int, int, int],
    width: int,
    dash: int,
) -> None:
    length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    if length < 1.0:
        return
    ux = (p1[0] - p0[0]) / length
    uy = (p1[1] - p0[1]) / length
    pos = 0.0
    while pos < length:
        stop = min(pos + dash, length)
        draw.line(
            [
                (int(round(p0[0] + ux * pos)), int(round(p0[1] + uy * pos))),
                (int(round(p0[0] + ux * stop)), int(round(p0[1] + uy * stop))),
            ],
            fill=color,
            width=width,
        )
        pos += 2 * dash


def _draw_angle_arc(
    draw: ImageDraw.ImageDraw,
    vertex: tuple[int, int],
    toward: tuple[int, int],
    radius: int,
    angle_deg: float,
    color: tuple[int, int, int],
    font: ImageFont.FreeTypeFont,
    stroke: int,
    label_side: Literal["tilt", "left"] = "tilt",
) -> None:
    """Сектор между вертикалью вверх и направлением на `toward`, с подписью градусов."""
    a_axis = math.degrees(math.atan2(toward[1] - vertex[1], toward[0] - vertex[0]))
    start, end = sorted((a_axis, -90.0))
    if end - start > 180.0:
        start, end = end, start + 360.0
    box = (vertex[0] - radius, vertex[1] - radius, vertex[0] + radius, vertex[1] + radius)
    draw.arc(box, start, end, fill=color, width=max(2, stroke))
    text = f"{angle_deg:.1f}°"
    text_w = int(draw.textlength(text, font=font))
    match label_side:
        case "left":
            chip_x = vertex[0] - text_w - 12
            chip_y = vertex[1] - radius
        case "tilt":
            gap = radius + max(12, radius // 3)
            chip_x = vertex[0] + gap if a_axis > -90.0 else vertex[0] - gap - text_w
            chip_y = vertex[1] - radius
    _draw_chip(draw, (chip_x, chip_y), text, font, outline=color)


def _draw_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    font: ImageFont.FreeTypeFont,
    color: tuple[int, int, int],
) -> None:
    x, y = xy
    draw.text((x, y), text, font=font, fill=color, stroke_width=2, stroke_fill=(0, 0, 0))


def _gray_to_rgba(image_uint8: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb).convert("RGBA")


def _draw_foreign(
    draw: ImageDraw.ImageDraw,
    mask: np.ndarray,
    label: str,
    font: ImageFont.FreeTypeFont,
    stroke: int,
) -> None:
    if int(mask.sum()) == 0:
        return
    color = _bgr_to_rgb(FOREIGN_COLOR_BGR)
    _draw_contours(draw, mask, color, max(stroke, 3))
    xy = _mask_label_xy(mask)
    if xy is not None:
        _draw_label(draw, label, xy, font, color)


def _draw_fov_ray(
    draw: ImageDraw.ImageDraw,
    ray: HipFovRay,
    margin_mm: float,
    font: ImageFont.FreeTypeFont,
    stroke: int,
) -> None:
    color = _bgr_to_rgb(FOV_OK_BGR if ray.distance_mm >= margin_mm else FOV_BAD_BGR)
    p0 = (int(round(ray.origin[0])), int(round(ray.origin[1])))
    p1 = (int(round(ray.edge[0])), int(round(ray.edge[1])))
    draw.line([p0, p1], fill=color, width=max(2, stroke))
    r = max(3, stroke + 1)
    draw.ellipse((p0[0] - r, p0[1] - r, p0[0] + r, p0[1] + r), fill=color, outline=(0, 0, 0))
    mx = (p0[0] + p1[0]) // 2
    my = (p0[1] + p1[1]) // 2
    vertical = abs(p1[0] - p0[0]) < abs(p1[1] - p0[1])
    chip_xy = (mx + 8, my - 10) if vertical else (mx - 10, max(8, my - 22))
    _draw_chip(draw, chip_xy, f"{ray.distance_mm / 10.0:.1f} см", font, outline=color)


def _draw_sidebar_row(
    draw: ImageDraw.ImageDraw,
    row: SidebarRow,
    font: ImageFont.FreeTypeFont,
    start_xy: tuple[int, int],
    line_end_x: int,
    ball_radius: int,
) -> None:
    x, y = start_xy
    gap = max(8, ball_radius)
    ball_gap = (2 * ball_radius + gap) if row.ball is not None else 0
    value_right = line_end_x - ball_gap
    value_width = draw.textlength(row.value, font=font)
    value_x = value_right - value_width
    key = _ellipsize(draw, row.key, font, max(0.0, value_x - x - gap))
    key_width = draw.textlength(key, font=font)
    pattern = "  ."
    pattern_width = max(draw.textlength(pattern, font=font), 1.0)
    repeats = max(0, int((value_x - gap - (x + key_width)) // pattern_width))
    draw.text((x, y), f"{key}{pattern * repeats}", font=font, fill=TEXT_MUTED)
    draw.text((value_x, y), row.value, font=font, fill=row.value_color)
    if row.ball is None:
        return
    box = draw.textbbox((x, y), "Ag", font=font)
    cy = y + (box[3] - box[1]) / 2
    cx = line_end_x - ball_radius
    draw.ellipse((cx - ball_radius, cy - ball_radius, cx + ball_radius, cy + ball_radius), fill=row.ball)


def _binary_ball(flag: bool) -> tuple[int, int, int]:
    return COLOR_RED if flag else COLOR_GREEN


def _compose_sidebar(image: Image.Image, groups: list[list[SidebarRow]], title: str) -> Image.Image:
    p = _build_params(image.width, image.height)
    canvas = Image.new("RGB", (p.width + p.padding, p.height), PANEL_BG)
    canvas.paste(image.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.line([(p.width, 0), (p.width, p.height)], fill=(36, 36, 36), width=max(1, int(2 * p.m)))

    panel_cx = p.width + p.padding // 2
    draw.multiline_text(
        (panel_cx, p.height // 4),
        FOR_DECISION_SUPPORT,
        font=p.font_bold,
        fill=TEXT_WHITE,
        anchor="ms",
        align="center",
        spacing=max(4, int(9 * p.m)),
    )

    left = p.width + int(20 * p.m)
    right = p.width + p.padding - int(20 * p.m)
    y = int(p.height / 2.5)
    ball_radius = max(5, int(8 * p.m))

    title_step = int(p.bold_size * 1.3)
    for line in _wrap_text(draw, title, p.font_bold, right - left):
        draw.text((left, y), line, font=p.font_bold, fill=TEXT_WHITE)
        y += title_step
    y += int(12 * p.m)

    for index, rows in enumerate(groups):
        if index > 0:
            draw.line([(left, y), (right, y)], fill=DIVIDER_GRAY, width=max(1, int(2 * p.m)))
            y += int(16 * p.m)
        for row in rows:
            _draw_sidebar_row(draw, row, p.font_medium, (left, y), right, ball_radius)
            y += p.line_step

    footer_x = p.width + p.padding - int(10 * p.m)
    footer_y = p.height - int(5 * p.m)
    draw.text((footer_x, footer_y), f"Версия: {SERVICE_VERSION}", font=p.font_medium, fill=TEXT_MUTED, anchor="rd")
    draw.text(
        (footer_x, footer_y - (p.medium_size + max(4, int(5 * p.m)))),
        SERVICE_NAME,
        font=p.font_medium,
        fill=TEXT_MUTED,
        anchor="rd",
    )
    return canvas


def _verdict_row(run_result: DXAImageResult) -> SidebarRow:
    bad = run_result.quality_class == 1
    return SidebarRow(
        key="Заключение",
        value="брак" if bad else "годен",
        ball=_binary_ball(bad),
        value_color=COLOR_RED if bad else COLOR_GREEN,
    )


def _violation_rows(run_result: DXAImageResult) -> list[SidebarRow]:
    if run_result.region_name is None:
        return []
    present = set(closed_violations_from_reasons(run_result.violation_reasons, run_result.region_name))
    rows: list[SidebarRow] = []
    for name in REGION_VIOLATIONS[run_result.region_name]:
        hit = name in present
        yes = hit if name in SIDEBAR_YES_MEANS_BAD else not hit
        rows.append(
            SidebarRow(
                key=SIDEBAR_VIOLATION_KEYS[name],
                value="да" if yes else "нет",
                ball=_binary_ball(hit),
            )
        )
    return rows


def _draw_spine_overlay(viz: SpineVisualization) -> Image.Image:
    scale, new_w, new_h = _scale_hw(*viz.image_uint8.shape[:2])
    size = (new_w, new_h)
    canvas = _gray_to_rgba(_resize_gray(viz.image_uint8, size))
    arr = np.array(canvas)
    stroke = max(2, int(round(2 * new_h / 1500)))

    foreign = _resize_mask(viz.foreign_mask, size)
    _blend_mask(arr, foreign, _bgr_to_rgb(FOREIGN_COLOR_BGR), 0.35)
    canvas = Image.fromarray(arr, "RGBA").convert("RGB")
    draw = ImageDraw.Draw(canvas)
    p = _build_params(new_w, new_h)

    _draw_foreign(draw, foreign, "артефакт", p.font_label, stroke)

    l1 = _scale_point(viz.l1_xy, scale)
    l5 = _scale_point(viz.l5_xy, scale)
    if l1 is not None and l5 is not None:
        axis = _bgr_to_rgb(AXIS_COLOR_BGR)
        dot = max(3, new_w // 110)
        p1 = (int(round(l1[0])), int(round(l1[1])))
        p5 = (int(round(l5[0])), int(round(l5[1])))
        draw.line([p1, p5], fill=axis, width=max(2, stroke))
        for xy in (p1, p5):
            draw.ellipse((xy[0] - dot, xy[1] - dot, xy[0] + dot, xy[1] + dot), fill=axis, outline=(0, 0, 0))
        low, high = (p1, p5) if p1[1] > p5[1] else (p5, p1)
        length = math.hypot(high[0] - low[0], high[1] - low[1])
        _draw_dashed_line(
            draw,
            low,
            (low[0], int(round(low[1] - length))),
            REFERENCE_GRAY,
            max(1, stroke - 1),
            max(6, int(12 * p.m)),
        )
        if viz.axis_angle_deg is not None:
            _draw_angle_arc(
                draw,
                low,
                high,
                max(18, int(26 * p.m)),
                viz.axis_angle_deg,
                axis,
                p.font_label,
                stroke,
                label_side="left",
            )
    return canvas


def _draw_protrusion(
    draw: ImageDraw.ImageDraw,
    viz: HipVisualization,
    scale: float,
    font: ImageFont.FreeTypeFont,
    stroke: int,
) -> None:
    a = _scale_point(viz.shaft_medial_a, scale)
    b = _scale_point(viz.shaft_medial_b, scale)
    tip = _scale_point(viz.lt_tip, scale)
    if a is None or b is None:
        return
    axis = _bgr_to_rgb(AXIS_COLOR_BGR)
    geom = protrusion_geometry(a, b, tip) if tip is not None else None
    if geom is None or tip is None:
        pa = (int(round(a[0])), int(round(a[1])))
        pb = (int(round(b[0])), int(round(b[1])))
        draw.line([pa, pb], fill=axis, width=max(2, stroke))
        return
    start, end, foot = geom
    draw.line(
        [(int(round(start[0])), int(round(start[1]))), (int(round(end[0])), int(round(end[1])))],
        fill=axis,
        width=max(2, stroke),
    )
    p_tip = (int(round(tip[0])), int(round(tip[1])))
    p_foot = (int(round(foot[0])), int(round(foot[1])))
    perp = _bgr_to_rgb(KP_COLORS_BGR["lt_tip"])
    draw.line([p_tip, p_foot], fill=perp, width=max(2, stroke))
    r = max(4, stroke + 2)
    draw.ellipse((p_tip[0] - r, p_tip[1] - r, p_tip[0] + r, p_tip[1] + r), fill=perp, outline=(0, 0, 0))
    r_foot = max(3, stroke)
    draw.ellipse(
        (p_foot[0] - r_foot, p_foot[1] - r_foot, p_foot[0] + r_foot, p_foot[1] + r_foot),
        fill=perp,
        outline=(0, 0, 0),
    )
    if viz.protrusion_mm is None:
        return
    text = f"{viz.protrusion_mm:.1f} мм"
    text_box = draw.textbbox((0, 0), text, font=font)
    text_h = text_box[3] - text_box[1]
    _draw_chip(draw, (p_foot[0] + 8, p_foot[1] - text_h - 14), text, font, outline=perp)


def _draw_shaft_axis(
    draw: ImageDraw.ImageDraw,
    viz: HipVisualization,
    scale: float,
    font: ImageFont.FreeTypeFont,
    stroke: int,
    m: float,
) -> None:
    axis_geom = viz.shaft_axis
    if axis_geom is None:
        return
    proximal = _scale_point(axis_geom.proximal, scale)
    distal = _scale_point(axis_geom.distal, scale)
    if proximal is None or distal is None:
        return
    color = _bgr_to_rgb(AXIS_COLOR_BGR)
    high = (int(round(proximal[0])), int(round(proximal[1])))
    low = (int(round(distal[0])), int(round(distal[1])))
    draw.line([low, high], fill=color, width=max(2, stroke))
    length = math.hypot(high[0] - low[0], high[1] - low[1])
    _draw_dashed_line(
        draw,
        low,
        (low[0], int(round(low[1] - length))),
        REFERENCE_GRAY,
        max(1, stroke - 1),
        max(6, int(12 * m)),
    )
    _draw_angle_arc(
        draw,
        low,
        high,
        max(18, int(26 * m)),
        axis_geom.angle_deg,
        color,
        font,
        stroke,
        label_side="left",
    )


def _draw_hip_overlay(viz: HipVisualization) -> Image.Image:
    scale, new_w, new_h = _scale_hw(*viz.image_uint8.shape[:2])
    size = (new_w, new_h)
    canvas = _gray_to_rgba(_resize_gray(viz.image_uint8, size))
    arr = np.array(canvas)
    stroke = max(2, int(round(2 * new_h / 1500)))

    prothesis = _resize_mask(viz.prothesis_mask, size)
    _blend_mask(arr, prothesis, _bgr_to_rgb(FOREIGN_COLOR_BGR), 0.35)
    canvas = Image.fromarray(arr, "RGBA").convert("RGB")
    draw = ImageDraw.Draw(canvas)
    p = _build_params(new_w, new_h)

    _draw_foreign(draw, prothesis, "протез", p.font_label, stroke)

    if viz.anatomy_bbox is not None:
        x0, y0, x1, y1 = viz.anatomy_bbox
        box = (int(round(x0 * scale)), int(round(y0 * scale)), int(round(x1 * scale)) - 1, int(round(y1 * scale)) - 1)
        draw.rectangle(box, outline=_bgr_to_rgb(ROI_COLOR_BGR), width=max(1, stroke - 1))

    for ray, margin_mm in (
        (_scale_fov_ray(viz.fov_ischium, scale), viz.fov_vertical_margin_mm),
        (_scale_fov_ray(viz.fov_gt_up, scale), viz.fov_vertical_margin_mm),
        (_scale_fov_ray(viz.fov_gt, scale), viz.fov_lateral_margin_mm),
    ):
        if ray is None:
            continue
        _draw_fov_ray(draw, ray, margin_mm, p.font_label, stroke)

    _draw_protrusion(draw, viz, scale, p.font_label, stroke)
    _draw_shaft_axis(draw, viz, scale, p.font_label, stroke, p.m)
    return canvas


def _spine_rows(run_result: DXAImageResult, viz: SpineVisualization) -> list[list[SidebarRow]]:
    return [
        [_verdict_row(run_result)],
        _violation_rows(run_result),
        [
            SidebarRow(
                "Угол L1–L5",
                f"{viz.axis_angle_deg:.1f}°" if viz.axis_angle_deg is not None else "н/д",
                None
                if viz.axis_angle_deg is None
                else _binary_ball(viz.axis_angle_deg > viz.axis_angle_threshold_deg),
            ),
            SidebarRow(
                "Видимость Th12",
                f"{viz.t12_height_ratio * 100:.0f}%" if viz.t12_height_ratio is not None else "н/д",
                None
                if viz.t12_height_ratio is None
                else _binary_ball(viz.t12_height_ratio < viz.t12_height_ratio_threshold),
            ),
        ],
    ]


def _fov_worst(viz: HipVisualization) -> tuple[float, bool] | None:
    """Луч с наименьшим запасом относительно своего порога (3 см по вертикали, 2 см латерально)."""
    pairs = [
        (ray.distance_mm, margin_mm)
        for ray, margin_mm in (
            (viz.fov_ischium, viz.fov_vertical_margin_mm),
            (viz.fov_gt_up, viz.fov_vertical_margin_mm),
            (viz.fov_gt, viz.fov_lateral_margin_mm),
        )
        if ray is not None
    ]
    if not pairs:
        return None
    distance_mm, margin_mm = min(pairs, key=lambda item: item[0] / item[1])
    return distance_mm, distance_mm < margin_mm


def _hip_rows(run_result: DXAImageResult, viz: HipVisualization) -> list[list[SidebarRow]]:
    worst = _fov_worst(viz)
    return [
        [_verdict_row(run_result)],
        _violation_rows(run_result),
        [
            SidebarRow(
                "Ось бедра",
                f"{viz.shaft_axis.angle_deg:.1f}°" if viz.shaft_axis is not None else "н/д",
                None
                if viz.shaft_axis is None
                else _binary_ball(viz.shaft_axis.angle_deg > viz.shaft_axis_threshold_deg),
            ),
            SidebarRow("Ротация", ROTATION_RU[viz.rotation_state], _binary_ball(viz.rotation_state != "ok")),
            SidebarRow("Выступ МВ", f"{viz.protrusion_mm:.1f} мм" if viz.protrusion_mm is not None else "н/д"),
            SidebarRow(
                "Запас поля",
                "н/д" if worst is None else f"{worst[0] / 10.0:.1f} см",
                None if worst is None else _binary_ball(worst[1]),
            ),
        ],
    ]


def draw_visualization(run_result: DXAImageResult) -> np.ndarray:
    match run_result.visualization:
        case SpineVisualization() as viz:
            overlay = _draw_spine_overlay(viz)
            canvas = _compose_sidebar(overlay, _spine_rows(run_result, viz), run_result.anatomical_region)
        case HipVisualization() as viz:
            overlay = _draw_hip_overlay(viz)
            canvas = _compose_sidebar(overlay, _hip_rows(run_result, viz), run_result.anatomical_region)
        case None:
            raise ValueError("Нет visualization в результате")
        case _:
            raise TypeError(f"Неизвестный visualization: {type(run_result.visualization)}")
    return np.asarray(canvas)
