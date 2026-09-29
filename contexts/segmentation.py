"""Сегментация Unet++ B3: multiclass (позвонки | proximal ТБС) / binary."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch.transforms import ToTensorV2

from ..image_utils import keep_largest_components
from ..labels import VERTEBRA_LABELS_TO_IDS
from ..utils.augmentations import AddInversionAndEqualizedByHistChannels
from ..utils.postprocessing import crop_padding_in_prediction_mask
from ..weights import load_lightning_weights

ApplyOnOutput = Literal["softmax", "sigmoid"]


class SemanticSegmentationContext:
    def __init__(
        self,
        weights: Path,
        device: str,
        classes: int,
        apply_on_output: ApplyOnOutput,
        labels_to_ids: dict[str, int] | None = None,
        resized_max_size: int = 384,
        threshold: float = 0.5,
        keep_n_components: int | None = 1,
    ):
        self.device = device
        self.resized_max_size = resized_max_size
        self.apply_on_output = apply_on_output
        self.threshold = threshold
        self.keep_n_components = keep_n_components
        self.labels_to_ids = labels_to_ids if labels_to_ids is not None else VERTEBRA_LABELS_TO_IDS
        self.model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b3",
            encoder_weights=None,
            in_channels=3,
            classes=classes,
        )
        load_lightning_weights(self.model, weights, device)
        self.transform = A.Compose(
            [
                A.ToFloat(),
                A.LongestMaxSize(resized_max_size),
                AddInversionAndEqualizedByHistChannels(),
                A.PadIfNeeded(
                    resized_max_size,
                    resized_max_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    position="center",
                ),
                ToTensorV2(),
            ]
        )

    def _forward(self, image_uint8: np.ndarray) -> np.ndarray:
        height, width = image_uint8.shape[:2]
        batch = self.transform(image=image_uint8)["image"].unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(batch)
            match self.apply_on_output:
                case "softmax":
                    pred = logits.softmax(1).argmax(1)[0].detach().cpu().numpy()
                case "sigmoid":
                    pred = (logits.sigmoid()[0, 0] > self.threshold).to(torch.uint8).detach().cpu().numpy()
        pred = crop_padding_in_prediction_mask(pred, self.resized_max_size, width, height)
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if pred.size == 0:
            return np.zeros((height, width), dtype=np.uint8)
        return cv2.resize(pred.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)

    def _filter_components(self, mask: np.ndarray) -> np.ndarray:
        match self.keep_n_components:
            case None:
                return (mask > 0).astype(np.uint8)
            case n:
                return keep_largest_components(mask, n)

    def predict_multiclass(self, image_uint8: np.ndarray) -> dict[str, np.ndarray]:
        class_map = self._forward(image_uint8)
        return {
            name: self._filter_components((class_map == idx).astype(np.uint8))
            for name, idx in self.labels_to_ids.items()
        }

    def predict_binary(self, image_uint8: np.ndarray) -> np.ndarray:
        return self._filter_components(self._forward(image_uint8))
