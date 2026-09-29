"""EfficientNet-b3: spine vs hip. Val-ауги как в region_classification/1_base."""

from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2

from ..config import RegionClassificationConfig
from ..image_utils import to_uint8_gray
from ..labels import REGION_LABELS_TO_IDS
from ..results import RegionName
from ..utils.augmentations import AddInversionAndEqualizedByHistChannels
from ..utils.efficientnet import EfficientNet
from ..weights import load_lightning_weights, resolve_checkpoint, resolve_device

RegionNameAndScore = tuple[RegionName, float]


class RegionClassificationContext:
    def __init__(self, config: RegionClassificationConfig):
        self.config = config
        self.device = resolve_device(config.device)
        self.ids_to_labels = {idx: name for name, idx in REGION_LABELS_TO_IDS.items()}
        self._model: EfficientNet | None = None
        self.transform = A.Compose(
            [
                A.ToFloat(),
                A.LongestMaxSize(config.resized_max_size),
                AddInversionAndEqualizedByHistChannels(),
                A.PadIfNeeded(config.resized_max_size, config.resized_max_size, border_mode=cv2.BORDER_CONSTANT),
                ToTensorV2(),
            ]
        )

    def _ensure_model(self) -> EfficientNet:
        if self._model is None:
            model = EfficientNet(
                version="efficientnet-b3",
                pretrained=False,
                num_classes=len(REGION_LABELS_TO_IDS),
                in_channels=3,
            )
            load_lightning_weights(
                model,
                resolve_checkpoint(self.config.weights, "region"),
                self.device,
            )
            self._model = model
        return self._model

    def predict(self, image: np.ndarray) -> RegionNameAndScore:
        model = self._ensure_model()
        image_uint8 = to_uint8_gray(image)
        batch = self.transform(image=image_uint8)["image"].unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = model(batch)
            probs = logits.softmax(dim=1)[0]
            idx = int(probs.argmax().item())
        region = self.ids_to_labels[idx]
        match region:
            case "spine" | "hip":
                return region, float(probs[idx].item())
            case _:
                raise ValueError(f"Неизвестная зона: {region}")
