"""Нормализация кадра, bbox по маске, фильтр компонент связности, кропы ТБС."""

from __future__ import annotations

import cv2
import numpy as np

BONE_CROP_PAD = 0.2
LT_BONE_TOP_CUT = 0.35


def keep_largest_components(mask: np.ndarray, n_components: int = 1) -> np.ndarray:
    """Оставляет n самых больших 8-связных компонент. Пустая маска без изменений."""
    binary = (mask > 0).astype(np.uint8)
    if n_components < 1 or int(binary.sum()) == 0:
        return binary
    n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        return binary
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep_ids = 1 + np.argsort(areas)[::-1][:n_components]
    return np.isin(labels, keep_ids).astype(np.uint8)


def minmax_uint8(image: np.ndarray) -> np.ndarray:
    """Как KeypointsToRegressionDataset: min_max по кропу, потом uint8."""
    image = image.astype(np.float32)
    lo = float(image.min())
    hi = float(image.max())
    if hi - lo < 1e-6:
        return np.zeros_like(image, dtype=np.uint8)
    return np.clip((image - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def to_uint8_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = image[..., 0]
    match image.dtype:
        case np.uint8:
            return image
        case _:
            image = image.astype(np.float32)
            if float(image.max()) <= 1.0:
                image = image * 255.0
            return np.clip(image, 0, 255).astype(np.uint8)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(min_x, min_y, max_x+1, max_y+1) либо None, если маска пустая."""
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def bbox_height(bbox: tuple[int, int, int, int]) -> int:
    _x0, y0, _x1, y1 = bbox
    return y1 - y0


def union_bboxes(bboxes: list[tuple[int, int, int, int] | None]) -> tuple[int, int, int, int] | None:
    valid = [box for box in bboxes if box is not None]
    if not valid:
        return None
    return (
        min(box[0] for box in valid),
        min(box[1] for box in valid),
        max(box[2] for box in valid),
        max(box[3] for box in valid),
    )


def padded_crop_box(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    pad: float,
    min_size: int = 8,
) -> tuple[int, int, int, int] | None:
    x0, y0, x1, y1 = bbox
    box_w = max(x1 - x0, 1)
    box_h = max(y1 - y0, 1)
    x0 = max(0, int(x0 - pad * box_w))
    y0 = max(0, int(y0 - pad * box_h))
    x1 = min(image_width, int(np.ceil(x1 + pad * box_w)))
    y1 = min(image_height, int(np.ceil(y1 + pad * box_h)))
    if x1 - x0 < min_size or y1 - y0 < min_size:
        return None
    return x0, y0, x1, y1


def bbox_from_rel_points(
    points: list[tuple[float, float]],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """Bbox в пикселях из относительных точек 0–1, формат как у mask_bbox."""
    if not points:
        return None
    xs = [float(x) * width for x, _y in points]
    ys = [float(y) * height for _x, y in points]
    x0 = int(np.floor(min(xs)))
    y0 = int(np.floor(min(ys)))
    x1 = int(np.ceil(max(xs)))
    y1 = int(np.ceil(max(ys)))
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    return x0, y0, x1, y1


def lt_crop_box(
    bone_bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    pad: float = BONE_CROP_PAD,
    top_cut: float = LT_BONE_TOP_CUT,
    min_size: int = 8,
) -> tuple[int, int, int, int] | None:
    """Дистальные ~65% bone-кропа: отрезаем проксимальные top_cut (голова/шейка)."""
    cropped = padded_crop_box(bone_bbox, image_width, image_height, pad, min_size=min_size)
    if cropped is None:
        return None
    x0, y0, x1, y1 = cropped
    y0_lt = y0 + int((y1 - y0) * top_cut)
    if y1 - y0_lt < min_size:
        return cropped
    return x0, y0_lt, x1, y1


def paste_crop_mask(
    full_hw: tuple[int, int],
    crop_mask: np.ndarray,
    crop_box: tuple[int, int, int, int],
) -> np.ndarray:
    full = np.zeros(full_hw, dtype=np.uint8)
    x0, y0, x1, y1 = crop_box
    region = full[y0:y1, x0:x1]
    if region.size == 0 or crop_mask.size == 0:
        return full
    if crop_mask.shape[:2] != region.shape[:2]:
        crop_mask = cv2.resize(
            crop_mask.astype(np.uint8),
            (region.shape[1], region.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    full[y0:y1, x0:x1] = crop_mask.astype(np.uint8)
    return full
