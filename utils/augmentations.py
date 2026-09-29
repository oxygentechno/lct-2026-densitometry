"""Ауги инференса: те же каналы, что на обучении."""

from __future__ import annotations

import warnings

import albumentations as A
import cv2
import numpy as np
from albumentations import MAX_VALUES_BY_DTYPE, ImageOnlyTransform, is_grayscale_image
from skimage import exposure


class AddInversionAndEqualizedByHistChannels(ImageOnlyTransform):
    """Grayscale → 3 канала: исходный, инверсия, equalize_hist."""

    def __init__(
        self,
        always_apply: bool = True,
        p: float = 1.0,
        apply_only_to_one_channel_of_rgb_image: bool = False,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.apply_only_to_first_channel_of_rgb_image = apply_only_to_one_channel_of_rgb_image

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        if not is_grayscale_image(img) and self.apply_only_to_first_channel_of_rgb_image is False:
            raise ValueError(
                "Аугментация ждёт grayscale, либо RGB с apply_only_to_one_channel_of_rgb_image=True"
            )
        if len(img.shape) == 3 and img.shape[-1] == 1:
            img = img[:, :, 0]
        elif len(img.shape) == 3 and self.apply_only_to_first_channel_of_rgb_image is True:
            img = img[:, :, 0]

        old_dtype = img.dtype
        coef = MAX_VALUES_BY_DTYPE[img.dtype]
        img = img.astype(np.float32) * (1 / coef)
        img = np.stack([img, 1 - img, exposure.equalize_hist(img)], axis=-1)
        return (img * coef).astype(old_dtype)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("apply_only_to_first_channel_of_rgb_image",)


class ToRGB(A.ImageOnlyTransform):
    """Grayscale → RGB. Для heatmap-кейпоинтов, где энкодер на 3 каналах без hist-eq."""

    def __init__(self, always_apply: bool = True, p: float = 1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        if A.is_rgb_image(img):
            warnings.warn("Кадр уже RGB.", stacklevel=1)
            return img
        if not A.is_grayscale_image(img):
            raise TypeError("ToRGB ждёт 2D или 3D с последней осью 1.")
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ()
