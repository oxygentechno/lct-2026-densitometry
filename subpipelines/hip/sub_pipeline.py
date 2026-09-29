"""Hip/TBS quality: bone → 4 части + prothesis + rotation; все правила, все fail."""

from __future__ import annotations

import numpy as np

from ...config import HipSubPipelineConfig
from ...contexts.axis_keypoint import AxisKeypointContext
from ...contexts.quality_classification import QualityClassificationContext
from ...contexts.segmentation import SemanticSegmentationContext
from ...image_utils import (
    lt_crop_box,
    mask_bbox,
    padded_crop_box,
    paste_crop_mask,
    to_uint8_gray,
    union_bboxes,
)
from ...labels import (
    HIP_GT_LABELS_TO_IDS,
    HIP_ISCHIUM_LABELS_TO_IDS,
    HIP_LT_LABELS_TO_IDS,
    HIP_PARTS_LABELS_TO_IDS,
    HIP_PROXIMAL_LABELS_TO_IDS,
    HIP_ROI_PARTS,
    HIP_XLSX_VIOLATIONS,
    ROTATION_POINT_ORDER,
    format_violation_type,
)
from ...results import HipSubPipelineResult, HipVisualization
from ...weights import resolve_checkpoint, resolve_device
from .calculations import HipRuleInput, apply_hip_rules, infer_hip_side, snap_lt_tip_to_edge, snap_point_to_bone_edge


class HipSubPipeline:
    def __init__(self, config: HipSubPipelineConfig):
        self.config = config
        self.device = resolve_device(config.device)
        self._ischium: SemanticSegmentationContext | None = None
        self._proximal: SemanticSegmentationContext | None = None
        self._gt: SemanticSegmentationContext | None = None
        self._lt: SemanticSegmentationContext | None = None
        self._bone: SemanticSegmentationContext | None = None
        self._prothesis: SemanticSegmentationContext | None = None
        self._lt_tip: AxisKeypointContext | None = None
        self._shaft_a: AxisKeypointContext | None = None
        self._shaft_b: AxisKeypointContext | None = None
        self.xlsx_clf = QualityClassificationContext(
            weights=config.xlsx_clf_weights,
            device=self.device,
            backbone=config.xlsx_clf_backbone,
            violation_names=HIP_XLSX_VIOLATIONS,
            resized_max_size=config.xlsx_clf_resized_max_size,
            experiment_key="hip_xlsx_clf",
        )

    def _ensure_models(self) -> None:
        if self._ischium is None:
            self._ischium = SemanticSegmentationContext(
                weights=resolve_checkpoint(self.config.ischium_weights, "hip_ischium"),
                device=self.device,
                classes=1,
                apply_on_output="sigmoid",
                labels_to_ids=HIP_ISCHIUM_LABELS_TO_IDS,
                resized_max_size=self.config.seg_resized_max_size,
                threshold=self.config.mask_threshold,
            )
        if self._proximal is None:
            self._proximal = SemanticSegmentationContext(
                weights=resolve_checkpoint(self.config.proximal_weights, "hip_proximal"),
                device=self.device,
                classes=len(HIP_PROXIMAL_LABELS_TO_IDS) + 1,
                apply_on_output="softmax",
                labels_to_ids=HIP_PROXIMAL_LABELS_TO_IDS,
                resized_max_size=self.config.seg_resized_max_size,
                threshold=self.config.mask_threshold,
            )
        if self._gt is None:
            self._gt = SemanticSegmentationContext(
                weights=resolve_checkpoint(self.config.gt_weights, "hip_gt"),
                device=self.device,
                classes=1,
                apply_on_output="sigmoid",
                labels_to_ids=HIP_GT_LABELS_TO_IDS,
                resized_max_size=self.config.seg_resized_max_size,
                threshold=self.config.mask_threshold,
            )
        if self._lt is None:
            self._lt = SemanticSegmentationContext(
                weights=resolve_checkpoint(self.config.lt_weights, "hip_lt"),
                device=self.device,
                classes=1,
                apply_on_output="sigmoid",
                labels_to_ids=HIP_LT_LABELS_TO_IDS,
                resized_max_size=self.config.seg_resized_max_size,
                threshold=self.config.mask_threshold,
            )
        if self._bone is None:
            self._bone = SemanticSegmentationContext(
                weights=resolve_checkpoint(self.config.bone_weights, "hip_bone"),
                device=self.device,
                classes=1,
                apply_on_output="sigmoid",
                resized_max_size=self.config.seg_resized_max_size,
                threshold=self.config.mask_threshold,
            )
        if self._prothesis is None:
            self._prothesis = SemanticSegmentationContext(
                weights=resolve_checkpoint(self.config.prothesis_weights, "hip_prothesis"),
                device=self.device,
                classes=1,
                apply_on_output="sigmoid",
                resized_max_size=self.config.seg_resized_max_size,
                threshold=self.config.mask_threshold,
            )
        if self._lt_tip is None:
            self._lt_tip = AxisKeypointContext(
                weights=resolve_checkpoint(self.config.lt_tip_weights, "hip_lt_tip"),
                device=self.device,
                resized_max_size=self.config.rotation_resized_max_size,
                n_points=1,
            )
        if self._shaft_a is None:
            self._shaft_a = AxisKeypointContext(
                weights=resolve_checkpoint(self.config.shaft_medial_a_weights, "hip_shaft_a"),
                device=self.device,
                resized_max_size=self.config.rotation_resized_max_size,
                n_points=1,
            )
        if self._shaft_b is None:
            self._shaft_b = AxisKeypointContext(
                weights=resolve_checkpoint(self.config.shaft_medial_b_weights, "hip_shaft_b"),
                device=self.device,
                resized_max_size=self.config.rotation_resized_max_size,
                n_points=1,
            )

    def _empty_part_masks(self, height: int, width: int) -> dict[str, np.ndarray]:
        return {name: np.zeros((height, width), dtype=np.uint8) for name in HIP_PARTS_LABELS_TO_IDS}

    def _predict_parts(
        self,
        image_uint8: np.ndarray,
        bone_bbox: tuple[int, int, int, int] | None,
    ) -> dict[str, np.ndarray]:
        height, width = image_uint8.shape[:2]
        assert self._ischium is not None
        assert self._proximal is not None
        assert self._gt is not None
        assert self._lt is not None

        part_masks = self._empty_part_masks(height, width)
        part_masks["ischium"] = self._ischium.predict_binary(image_uint8)

        if bone_bbox is None:
            return part_masks
        bone_box = padded_crop_box(bone_bbox, width, height, self.config.crop_pad)
        if bone_box is None:
            return part_masks
        x0, y0, x1, y1 = bone_box
        bone_crop = image_uint8[y0:y1, x0:x1]
        if bone_crop.size == 0:
            return part_masks

        proximal = self._proximal.predict_multiclass(bone_crop)
        for name, mask in proximal.items():
            part_masks[name] = paste_crop_mask((height, width), mask, bone_box)
        part_masks["greater_trochanter"] = paste_crop_mask(
            (height, width),
            self._gt.predict_binary(bone_crop),
            bone_box,
        )

        lt_box = lt_crop_box(
            bone_bbox,
            width,
            height,
            pad=self.config.crop_pad,
            top_cut=self.config.lt_crop_top_cut,
        )
        if lt_box is None:
            return part_masks
        lx0, ly0, lx1, ly1 = lt_box
        lt_crop = image_uint8[ly0:ly1, lx0:lx1]
        if lt_crop.size == 0:
            return part_masks
        part_masks["lesser_trochanter"] = paste_crop_mask(
            (height, width),
            self._lt.predict_binary(lt_crop),
            lt_box,
        )
        return part_masks

    def _rotation_points(
        self,
        image_uint8: np.ndarray,
        bone_bbox: tuple[int, int, int, int] | None,
    ) -> dict[str, tuple[float, float] | None]:
        height, width = image_uint8.shape[:2]
        crop_box = padded_crop_box(bone_bbox, width, height, self.config.crop_pad) if bone_bbox is not None else None
        if crop_box is None:
            crop_box = (0, 0, width, height)
        x0, y0, x1, y1 = crop_box
        crop = image_uint8[y0:y1, x0:x1]
        empty: dict[str, tuple[float, float] | None] = {name: None for name in ROTATION_POINT_ORDER}
        if crop.size == 0:
            return empty
        assert self._lt_tip is not None
        assert self._shaft_a is not None
        assert self._shaft_b is not None
        points: dict[str, tuple[float, float] | None] = {}
        for name, ctx in (
            ("lt_tip", self._lt_tip),
            ("shaft_medial_a", self._shaft_a),
            ("shaft_medial_b", self._shaft_b),
        ):
            x_crop, y_crop = ctx.predict_xy(crop)
            points[name] = (x_crop + x0, y_crop + y0)
        return points

    def run(self, image: np.ndarray) -> HipSubPipelineResult:
        self._ensure_models()
        assert self._bone is not None
        assert self._prothesis is not None
        assert self._lt_tip is not None
        assert self._shaft_a is not None
        assert self._shaft_b is not None

        image_uint8 = to_uint8_gray(image)
        height, width = image_uint8.shape[:2]

        bone_mask = self._bone.predict_binary(image_uint8)
        prothesis_mask = self._prothesis.predict_binary(image_uint8)
        bone_bbox = mask_bbox(bone_mask) if int(bone_mask.sum()) >= self.config.bone_min_area_px else None
        part_masks = self._predict_parts(image_uint8, bone_bbox)

        part_present = {
            name: int(part_masks[name].sum()) >= self.config.part_min_area_px for name in HIP_PARTS_LABELS_TO_IDS
        }
        lt_present = int(part_masks["lesser_trochanter"].sum()) >= self.config.lt_min_area_px
        hip_side = infer_hip_side(
            part_masks["greater_trochanter"],
            part_masks["femoral_head"],
            part_masks["ischium"],
            width,
        )

        roi_boxes = [mask_bbox(part_masks[name]) if part_present[name] else None for name in HIP_ROI_PARTS]
        anatomy_bbox = union_bboxes([*roi_boxes, bone_bbox])

        rotation_points = self._rotation_points(image_uint8, bone_bbox)
        rotation_points["lt_tip"] = snap_lt_tip_to_edge(
            rotation_points["lt_tip"],
            part_masks["lesser_trochanter"],
            bone_mask,
            hip_side,
        )
        rotation_points["shaft_medial_a"] = snap_point_to_bone_edge(rotation_points["shaft_medial_a"], bone_mask)
        rotation_points["shaft_medial_b"] = snap_point_to_bone_edge(rotation_points["shaft_medial_b"], bone_mask)
        rules = apply_hip_rules(
            HipRuleInput(
                part_present=part_present,
                lt_present=lt_present,
                prothesis_area_px=int(prothesis_mask.sum()),
                image_size=(height, width),
                anatomy_bbox=anatomy_bbox,
                ischium_mask=part_masks["ischium"],
                head_mask=part_masks["femoral_head"],
                gt_mask=part_masks["greater_trochanter"],
                shaft_mask=part_masks["femoral_shaft"],
                shaft_axis_angle_threshold_deg=self.config.shaft_axis_angle_threshold_deg,
                shaft_axis_proximal_cut=self.config.shaft_axis_proximal_cut,
                lt_tip=rotation_points["lt_tip"],
                shaft_medial_a=rotation_points["shaft_medial_a"],
                shaft_medial_b=rotation_points["shaft_medial_b"],
                pixel_size_x_mm=self.config.pixel_size_x_mm,
                pixel_size_y_mm=self.config.pixel_size_y_mm,
                roi_margin_x_mm=self.config.roi_margin_x_mm,
                roi_margin_y_mm=self.config.roi_margin_y_mm,
                under_rotation_min_mm=self.config.under_rotation_min_mm,
                prothesis_min_area_px=self.config.prothesis_min_area_px,
            )
        )

        quality_class = 1 if rules.reasons else 0
        xlsx_clf = self.xlsx_clf.predict(image_uint8)
        visualization = HipVisualization(
            image_uint8=image_uint8,
            part_masks=part_masks,
            bone_mask=bone_mask,
            prothesis_mask=prothesis_mask,
            lt_tip=rotation_points["lt_tip"],
            shaft_medial_a=rotation_points["shaft_medial_a"],
            shaft_medial_b=rotation_points["shaft_medial_b"],
            anatomy_bbox=anatomy_bbox,
            protrusion_mm=rules.protrusion_mm,
            rotation_state=rules.rotation_state,
            hip_side=rules.hip_side,
            fov_ischium=rules.fov_ischium,
            fov_gt_up=rules.fov_gt_up,
            fov_gt=rules.fov_gt,
            shaft_axis=rules.shaft_axis,
            shaft_axis_threshold_deg=self.config.shaft_axis_angle_threshold_deg,
            fov_vertical_margin_mm=self.config.roi_margin_y_mm,
            fov_lateral_margin_mm=self.config.roi_margin_x_mm,
            quality_class=quality_class,
            violation_reasons=rules.reasons,
            xlsx_clf=xlsx_clf,
        )
        return HipSubPipelineResult(
            quality_class=quality_class,
            violation_type=format_violation_type(rules.closed_violations),
            quality_prob=xlsx_clf.quality_prob,
            violation_reasons=rules.reasons,
            missing_parts=rules.missing_parts,
            rotation_state=rules.rotation_state,
            protrusion_mm=rules.protrusion_mm,
            hip_side=rules.hip_side,
            fov_ischium_mm=rules.fov_ischium.distance_mm if rules.fov_ischium is not None else None,
            fov_gt_up_mm=rules.fov_gt_up.distance_mm if rules.fov_gt_up is not None else None,
            fov_gt_mm=rules.fov_gt.distance_mm if rules.fov_gt is not None else None,
            shaft_axis_angle_deg=rules.shaft_axis.angle_deg if rules.shaft_axis is not None else None,
            roi_ok=rules.roi_ok,
            roi_margins_px=rules.roi_margins_px,
            prothesis_present=rules.prothesis_present,
            lt_tip=rotation_points["lt_tip"],
            shaft_medial_a=rotation_points["shaft_medial_a"],
            shaft_medial_b=rotation_points["shaft_medial_b"],
            xlsx_clf=xlsx_clf,
            visualization=visualization,
        )
