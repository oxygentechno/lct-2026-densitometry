"""Heatmap keypoints: DeepLabV3Plus B3, 256. 1 точка (axis / lt_tip / shaft)."""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from albumentations.pytorch.transforms import ToTensorV2

from ..image_utils import minmax_uint8
from ..utils.augmentations import ToRGB
from ..utils.keypoints import extract_coordinates_softargmax
from ..utils.postprocessing import convert_resized_and_padded_coordinates_to_original
from ..weights import load_lightning_weights


class AxisKeypointContext:
    def __init__(self, weights: Path, device: str, resized_max_size: int = 256, n_points: int = 1):
        self.device = device
        self.resized_max_size = resized_max_size
        self.n_points = n_points
        self.model = smp.DeepLabV3Plus(
            encoder_name="efficientnet-b3",
            encoder_weights=None,
            in_channels=3,
            classes=n_points,
        )
        load_lightning_weights(self.model, weights, device)
        self.transform = A.Compose(
            [
                A.LongestMaxSize(resized_max_size),
                ToRGB(),
                A.PadIfNeeded(
                    resized_max_size,
                    resized_max_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    position="center",
                ),
                A.ToFloat(),
                ToTensorV2(),
            ]
        )

    def predict_points(self, crop_uint8: np.ndarray) -> list[tuple[float, float]]:
        """Точки в координатах кропа. pred = [x, y, x, y, ...]."""
        crop_uint8 = minmax_uint8(crop_uint8)
        height, width = crop_uint8.shape[:2]
        batch = self.transform(image=crop_uint8)["image"].unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(batch)
            if logits.shape[-2] != self.resized_max_size or logits.shape[-1] != self.resized_max_size:
                logits = F.interpolate(
                    logits,
                    size=(self.resized_max_size, self.resized_max_size),
                    mode="bilinear",
                    align_corners=False,
                )
            pred = extract_coordinates_softargmax(logits).flatten(1, 2)[0]
        points: list[tuple[float, float]] = []
        for idx in range(self.n_points):
            x, y = float(pred[2 * idx].item()), float(pred[2 * idx + 1].item())
            x, y = convert_resized_and_padded_coordinates_to_original(x, y, width, height, self.resized_max_size)
            points.append((x, y))
        return points

    def predict_xy(self, crop_uint8: np.ndarray) -> tuple[float, float]:
        return self.predict_points(crop_uint8)[0]
