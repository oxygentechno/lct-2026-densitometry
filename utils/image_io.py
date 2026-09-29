"""Чтение кадра в [0, 1], 2D. min_max по всему динамическому диапазону."""

from __future__ import annotations

import cv2
import numpy as np
from pydicom import dcmread

_RASTER_EXTENSIONS = ("png", "jpg", "jpeg", "bmp")
_DICOM_EXTENSIONS = ("dcm", "dicom")


def _minmax(image: np.ndarray) -> np.ndarray:
    min_ = np.min(image)
    max_ = np.max(image)
    image = (image - min_) / (max_ - min_)
    image[image > 1.0] = 1.0
    image[image < 0.0] = 0.0
    return image


def _read_dicom(path: str) -> np.ndarray:
    dataset = dcmread(path)
    image = dataset.pixel_array.astype(np.float64)
    if "RescaleSlope" in dataset and "RescaleIntercept" in dataset:
        image = image * dataset.RescaleSlope + dataset.RescaleIntercept

    photometric = dataset["PhotometricInterpretation"].value
    match photometric:
        case "MONOCHROME1":
            image = -image
        case "MONOCHROME2":
            pass
        case "RGB" if image.shape[0] == 3:
            image = np.mean(image, axis=0)
        case "RGB" if image.shape[-1] == 3:
            image = np.mean(image, axis=-1)
        case _:
            raise ValueError(f"Неизвестный PhotometricInterpretation: {photometric}")

    if image.ndim > 2:
        image = image[0]
    return image


def read_and_normalize_image(path: str) -> np.ndarray:
    """png/jpg/bmp или dicom → float [0, 1], без цветового канала."""
    extension = path.split(".")[-1]
    match extension:
        case ext if ext in _RASTER_EXTENSIONS:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Не удалось прочитать {path}")
        case ext if ext in _DICOM_EXTENSIONS:
            image = _read_dicom(path)
        case _:
            raise ValueError(f"Неизвестное расширение: {extension}, путь: {path}")
    return _minmax(image)
