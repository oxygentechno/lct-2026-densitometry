"""EfficientNet с тем же конструктором, что на обучении. Веса грузятся снаружи."""

from __future__ import annotations

import torch
from efficientnet_pytorch import VALID_MODELS
from efficientnet_pytorch import EfficientNet as EfficientNetOriginal
from efficientnet_pytorch.utils import get_model_params, load_pretrained_weights


class EfficientNet(EfficientNetOriginal):
    def __init__(
        self,
        version: str,
        pretrained: bool = False,
        num_classes: int | None = None,
        in_channels: int = 3,
    ):
        if version not in VALID_MODELS:
            raise ValueError(f"version должен быть одним из {VALID_MODELS}, получено {version}")
        override_params: dict[str, int] = {}
        if num_classes is not None:
            override_params["num_classes"] = int(num_classes)
        blocks_args, global_params = get_model_params(version, override_params)
        super().__init__(blocks_args, global_params)
        if pretrained:
            load_pretrained_weights(model=self, model_name=version, load_fc=(num_classes == 1000))
        self._change_in_channels(int(in_channels))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return super().forward(inputs)
