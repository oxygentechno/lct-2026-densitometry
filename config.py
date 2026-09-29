"""Конфиг DXA quality pipeline."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class RegionClassificationConfig(BaseModel, extra="forbid"):
    device: str = "cuda"
    weights: Path | None = Path("weights/region_classification_epoch=2-loss=0.0057-f1=1.000.ckpt")
    resized_max_size: int = 384


class SpineSubPipelineConfig(BaseModel, extra="forbid"):
    device: str = "cuda"
    vertebra_weights: Path | None = Path(
        "weights/vertebra_semantic_segmentation_epoch=126-loss=0.2264-IoU=0.656.ckpt"
    )
    iliac_weights: Path | None = Path("weights/iliac_semantic_segmentation_epoch=64-loss=0.0812-IoU=0.838.ckpt")
    foreign_weights: Path | None = Path(
        "weights/foreign_semantic_segmentation_epoch=77-loss=0.2516-IoU=0.578.ckpt"
    )
    axis_weights: Path | None = Path(
        "weights/vertebra_axis_keypoint_detection_epoch=20-loss=2.2565-MSE=14.984.ckpt"
    )
    mask_threshold: float = 0.5
    vertebra_min_area_px: int = 10
    iliac_min_area_px: int = 1
    foreign_min_area_px: int = 1
    t12_height_ratio_threshold: float = 0.5
    axis_angle_threshold_deg: float = 5.0
    # Смещение оси по центрам позвонков от середины кадра. В тексте порога нет.
    center_offset_threshold_mm: float = 15.0
    pixel_size_x_mm: float = 0.6
    pixel_size_y_mm: float = 1.05
    crop_pad: float = 0.2
    seg_resized_max_size: int = 384
    axis_resized_max_size: int = 256
    xlsx_clf_weights: Path | None = Path(
        "weights/spine_quality_classification_epoch=4-loss=2.4814-f1=0.621.ckpt"
    )
    xlsx_clf_backbone: str = "efficientnet-b3"
    xlsx_clf_resized_max_size: int = 256


class HipSubPipelineConfig(BaseModel, extra="forbid"):
    device: str = "cuda"
    ischium_weights: Path | None = Path(
        "weights/hip_ischium_semantic_segmentation_epoch=88-loss=0.1456-IoU=0.864.ckpt"
    )
    proximal_weights: Path | None = Path(
        "weights/hip_proximal_semantic_segmentation_epoch=109-loss=0.2092-IoU=0.820.ckpt"
    )
    gt_weights: Path | None = Path(
        "weights/hip_greater_trochanter_semantic_segmentation_epoch=111-loss=0.1615-IoU=0.868.ckpt"
    )
    lt_weights: Path | None = Path(
        "weights/hip_lesser_trochanter_semantic_segmentation_epoch=47-loss=0.4140-IoU=0.485.ckpt"
    )
    bone_weights: Path | None = Path("weights/hip_bone_semantic_segmentation_epoch=89-loss=0.0837-IoU=0.931.ckpt")
    prothesis_weights: Path | None = Path("weights/hip_prothesis_semantic_segmentation_last.ckpt")
    lt_tip_weights: Path | None = Path(
        "weights/hip_lt_tip_keypoint_detection_epoch=43-loss=3.9429-MSE=34.096.ckpt"
    )
    shaft_medial_a_weights: Path | None = Path(
        "weights/hip_shaft_medial_a_keypoint_detection_epoch=81-loss=4.9360-MSE=43.714.ckpt"
    )
    shaft_medial_b_weights: Path | None = Path(
        "weights/hip_shaft_medial_b_keypoint_detection_epoch=46-loss=4.5210-MSE=43.847.ckpt"
    )
    mask_threshold: float = 0.5
    part_min_area_px: int = 10
    bone_min_area_px: int = 10
    lt_min_area_px: int = 5
    prothesis_min_area_px: int = 1
    crop_pad: float = 0.2
    lt_crop_top_cut: float = 0.35
    seg_resized_max_size: int = 384
    rotation_resized_max_size: int = 256
    pixel_size_x_mm: float = 0.6
    pixel_size_y_mm: float = 1.05
    # ТЗ 2.3: 3 см сверху и снизу от области интереса, 2 см от латерального края.
    roi_margin_x_mm: float = 20.0
    roi_margin_y_mm: float = 30.0
    # Порог недоротации по протрузии LT; переротация = нет маски lesser_trochanter.
    under_rotation_min_mm: float = 10.0
    # Угол оси диафиза к вертикали скана. В методичке числа нет, тот же допуск, что у оси L1–L5.
    shaft_axis_angle_threshold_deg: float = 5.0
    # Верхняя доля маски femoral_shaft: метафиз расширяется и уводит среднюю линию.
    shaft_axis_proximal_cut: float = 0.3
    xlsx_clf_weights: Path | None = Path(
        "weights/hip_quality_classification_epoch=19-loss=1.1833-f1=0.800.ckpt"
    )
    xlsx_clf_backbone: str = "efficientnet-b0"
    xlsx_clf_resized_max_size: int = 384


class DXAPipelineConfig(BaseModel, extra="forbid"):
    region: RegionClassificationConfig = RegionClassificationConfig()
    spine: SpineSubPipelineConfig = SpineSubPipelineConfig()
    hip: HipSubPipelineConfig = HipSubPipelineConfig()
