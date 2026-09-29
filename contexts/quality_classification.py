"""xlsx quality clf: EfficientNet + quality/violations. В пайплайне пока только прогон, в правила не идёт."""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch import Tensor, nn

from ..image_utils import to_uint8_gray
from ..results import XlsxClfPrediction
from ..utils.augmentations import AddInversionAndEqualizedByHistChannels
from ..utils.efficientnet import EfficientNet
from ..weights import load_lightning_weights, resolve_checkpoint, resolve_device


class RegionQualityEfficientNet(nn.Module):
    def __init__(self, version: str, n_violations: int, pretrained: bool = False, in_channels: int = 3):
        super().__init__()
        self.backbone = EfficientNet(
            version=version,
            pretrained=pretrained,
            num_classes=1,
            in_channels=in_channels,
        )
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()
        self.quality_head = nn.Linear(in_features, 1)
        self.violation_head = nn.Linear(in_features, n_violations)

    def extract_pooled(self, x: Tensor) -> Tensor:
        features = self.backbone.extract_features(x)
        features = self.backbone._avg_pooling(features)
        features = features.flatten(start_dim=1)
        return self.backbone._dropout(features)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        features = self.extract_pooled(x)
        return {
            "quality": self.quality_head(features).squeeze(-1),
            "violations": self.violation_head(features),
        }


class QualityClassificationContext:
    def __init__(
        self,
        weights: Path | None,
        device: str,
        backbone: str,
        violation_names: tuple[str, ...],
        resized_max_size: int,
        experiment_key: str,
    ):
        self.weights = weights
        self.experiment_key = experiment_key
        self.device = resolve_device(device)
        self.backbone = backbone
        self.violation_names = violation_names
        self.resized_max_size = resized_max_size
        self._model: RegionQualityEfficientNet | None = None
        self.transform = A.Compose(
            [
                A.ToFloat(),
                A.LongestMaxSize(resized_max_size),
                AddInversionAndEqualizedByHistChannels(),
                A.PadIfNeeded(resized_max_size, resized_max_size, border_mode=cv2.BORDER_CONSTANT),
                ToTensorV2(),
            ]
        )

    def _ensure_model(self) -> RegionQualityEfficientNet:
        if self._model is None:
            model = RegionQualityEfficientNet(
                version=self.backbone,
                n_violations=len(self.violation_names),
                pretrained=False,
                in_channels=3,
            )
            load_lightning_weights(
                model,
                resolve_checkpoint(self.weights, self.experiment_key),
                self.device,
            )
            self._model = model
        return self._model

    def predict(self, image: np.ndarray) -> XlsxClfPrediction:
        model = self._ensure_model()
        image_uint8 = to_uint8_gray(image)
        batch = self.transform(image=image_uint8)["image"].unsqueeze(0).to(self.device)
        with torch.inference_mode():
            outputs = model(batch)
            quality_prob = float(outputs["quality"].sigmoid().reshape(-1)[0].item())
            violation_probs = outputs["violations"].sigmoid().reshape(-1)
        return XlsxClfPrediction(
            quality_prob=quality_prob,
            violation_probs={
                name: float(violation_probs[idx].item()) for idx, name in enumerate(self.violation_names)
            },
        )
