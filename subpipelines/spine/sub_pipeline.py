"""Spine quality: 4 модели всегда, все правила, все найденные fail."""

from __future__ import annotations

import numpy as np

from ...config import SpineSubPipelineConfig
from ...contexts.axis_keypoint import AxisKeypointContext
from ...contexts.quality_classification import QualityClassificationContext
from ...contexts.segmentation import SemanticSegmentationContext
from ...image_utils import mask_bbox, padded_crop_box, to_uint8_gray
from ...labels import AXIS_VERTEBRAE, REQUIRED_VERTEBRAE, SPINE_XLSX_VIOLATIONS, format_violation_type
from ...results import SpineSubPipelineResult, SpineVisualization
from ...weights import resolve_checkpoint, resolve_device
from .calculations import SpineRuleInput, apply_spine_rules


class SpineSubPipeline:
    def __init__(self, config: SpineSubPipelineConfig):
        self.config = config
        self.device = resolve_device(config.device)
        self._vertebra: SemanticSegmentationContext | None = None
        self._iliac: SemanticSegmentationContext | None = None
        self._foreign: SemanticSegmentationContext | None = None
        self._axis: AxisKeypointContext | None = None
        self.xlsx_clf = QualityClassificationContext(
            weights=config.xlsx_clf_weights,
            device=self.device,
            backbone=config.xlsx_clf_backbone,
            violation_names=SPINE_XLSX_VIOLATIONS,
            resized_max_size=config.xlsx_clf_resized_max_size,
            experiment_key="spine_xlsx_clf",
        )

    def _ensure_models(self) -> tuple[
        SemanticSegmentationContext,
        SemanticSegmentationContext,
        SemanticSegmentationContext,
        AxisKeypointContext,
    ]:
        if self._vertebra is None:
            self._vertebra = SemanticSegmentationContext(
                weights=resolve_checkpoint(self.config.vertebra_weights, "vertebra"),
                device=self.device,
                classes=len(REQUIRED_VERTEBRAE) + 1,
                apply_on_output="softmax",
                resized_max_size=self.config.seg_resized_max_size,
            )
        if self._iliac is None:
            self._iliac = SemanticSegmentationContext(
                weights=resolve_checkpoint(self.config.iliac_weights, "iliac"),
                device=self.device,
                classes=1,
                apply_on_output="sigmoid",
                resized_max_size=self.config.seg_resized_max_size,
                threshold=self.config.mask_threshold,
                keep_n_components=2,
            )
        if self._foreign is None:
            self._foreign = SemanticSegmentationContext(
                weights=resolve_checkpoint(self.config.foreign_weights, "foreign"),
                device=self.device,
                classes=1,
                apply_on_output="sigmoid",
                resized_max_size=self.config.seg_resized_max_size,
                threshold=self.config.mask_threshold,
                keep_n_components=None,
            )
        if self._axis is None:
            self._axis = AxisKeypointContext(
                weights=resolve_checkpoint(self.config.axis_weights, "axis"),
                device=self.device,
                resized_max_size=self.config.axis_resized_max_size,
            )
        return self._vertebra, self._iliac, self._foreign, self._axis

    def _axis_points(
        self,
        image_uint8: np.ndarray,
        vertebra_bboxes: dict[str, tuple[int, int, int, int] | None],
        axis_ctx: AxisKeypointContext,
    ) -> dict[str, tuple[float, float] | None]:
        height, width = image_uint8.shape[:2]
        points: dict[str, tuple[float, float] | None] = {name: None for name in AXIS_VERTEBRAE}
        for name in AXIS_VERTEBRAE:
            bbox = vertebra_bboxes.get(name)
            if bbox is None:
                continue
            crop_box = padded_crop_box(bbox, width, height, self.config.crop_pad)
            if crop_box is None:
                continue
            x0, y0, x1, y1 = crop_box
            crop = image_uint8[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            x_crop, y_crop = axis_ctx.predict_xy(crop)
            points[name] = (x_crop + x0, y_crop + y0)
        return points

    def run(self, image: np.ndarray) -> SpineSubPipelineResult:
        vertebra_ctx, iliac_ctx, foreign_ctx, axis_ctx = self._ensure_models()
        image_uint8 = to_uint8_gray(image)

        vertebra_masks = vertebra_ctx.predict_multiclass(image_uint8)
        iliac_mask = iliac_ctx.predict_binary(image_uint8)
        foreign_mask = foreign_ctx.predict_binary(image_uint8)

        vertebra_bboxes: dict[str, tuple[int, int, int, int] | None] = {}
        for name in REQUIRED_VERTEBRAE:
            mask = vertebra_masks[name]
            if int(mask.sum()) < self.config.vertebra_min_area_px:
                vertebra_bboxes[name] = None
                continue
            vertebra_bboxes[name] = mask_bbox(mask)

        axis_points = self._axis_points(image_uint8, vertebra_bboxes, axis_ctx)
        rules = apply_spine_rules(
            SpineRuleInput(
                vertebra_bboxes=vertebra_bboxes,
                iliac_area_px=int(iliac_mask.sum()),
                foreign_area_px=int(foreign_mask.sum()),
                vertebra_masks=vertebra_masks,
                image_width=image_uint8.shape[1],
                l1_xy=axis_points["L1"],
                l5_xy=axis_points["L5"],
                axis_angle_threshold_deg=self.config.axis_angle_threshold_deg,
                t12_height_ratio_threshold=self.config.t12_height_ratio_threshold,
                center_offset_threshold_mm=self.config.center_offset_threshold_mm,
                pixel_size_x_mm=self.config.pixel_size_x_mm,
                pixel_size_y_mm=self.config.pixel_size_y_mm,
                iliac_min_area_px=self.config.iliac_min_area_px,
                foreign_min_area_px=self.config.foreign_min_area_px,
            )
        )

        quality_class = 1 if rules.reasons else 0
        xlsx_clf = self.xlsx_clf.predict(image_uint8)
        visualization = SpineVisualization(
            image_uint8=image_uint8,
            vertebra_masks=vertebra_masks,
            iliac_mask=iliac_mask,
            foreign_mask=foreign_mask,
            l1_xy=axis_points["L1"],
            l5_xy=axis_points["L5"],
            axis_angle_deg=rules.axis_angle_deg,
            axis_angle_threshold_deg=self.config.axis_angle_threshold_deg,
            t12_height_ratio=rules.t12_height_ratio,
            t12_height_ratio_threshold=self.config.t12_height_ratio_threshold,
            column_axis=rules.column_axis,
            column_center_threshold_mm=self.config.center_offset_threshold_mm,
            quality_class=quality_class,
            violation_reasons=rules.reasons,
            xlsx_clf=xlsx_clf,
        )
        return SpineSubPipelineResult(
            quality_class=quality_class,
            violation_type=format_violation_type(rules.closed_violations),
            quality_prob=xlsx_clf.quality_prob,
            violation_reasons=rules.reasons,
            missing_vertebrae=rules.missing_vertebrae,
            iliac_present=rules.iliac_present,
            axis_angle_deg=rules.axis_angle_deg,
            t12_height_ratio=rules.t12_height_ratio,
            column_offset_mm=rules.column_axis.offset_mm if rules.column_axis is not None else None,
            foreign_area_px=rules.foreign_area_px,
            l1_xy=axis_points["L1"],
            l5_xy=axis_points["L5"],
            xlsx_clf=xlsx_clf,
            visualization=visualization,
        )
