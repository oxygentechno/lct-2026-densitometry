"""Обратный LongestMaxSize + PadIfNeeded(position=center) в координаты исходного кадра."""

from __future__ import annotations

import numpy as np
import torch


def convert_resized_and_padded_coordinates_to_original(
    x: int | float,
    y: int | float,
    original_width: int,
    original_height: int,
    resized_max_size: int,
) -> tuple[float, float]:
    ratio = float(resized_max_size) / max(original_width, original_height)
    new_height, new_width = round(original_height * ratio), round(original_width * ratio)
    top = (resized_max_size - new_height) // 2
    left = (resized_max_size - new_width) // 2
    return (x - left) / ratio, (y - top) / ratio


def crop_padding_in_prediction_mask(
    mask: np.ndarray | torch.Tensor,
    resized_max_size: int | tuple[int, int],
    input_width: int,
    input_height: int,
) -> np.ndarray | torch.Tensor:
    """Срезает паддинг с маски. Только binary и multiclass, без канала класса."""
    match resized_max_size:
        case int() as side:
            ratio = float(side) / max(input_width, input_height)
            new_height, new_width = round(input_height * ratio), round(input_width * ratio)
            delta_w = (side - new_width) // 2
            delta_h = (side - new_height) // 2
        case (resized_h, resized_w):
            ratio = float(max(resized_max_size)) / max(input_width, input_height)
            new_height, new_width = round(input_height * ratio), round(input_width * ratio)
            delta_w = (resized_w - new_width) // 2
            delta_h = (resized_h - new_height) // 2
        case _:
            raise ValueError(f"resized_max_size: {type(resized_max_size)}")

    if delta_h == 0 and delta_w == 0:
        return mask
    if delta_h == 0:
        return mask[:, delta_w:-delta_w]
    if delta_w == 0:
        return mask[delta_h:-delta_h, :]
    return mask[delta_h:-delta_h, delta_w:-delta_w]
