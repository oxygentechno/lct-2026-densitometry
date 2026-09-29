"""DXA quality: region clf → spine | hip subpipeline."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

from .config import DXAPipelineConfig
from .contexts.region_classification import RegionClassificationContext
from .labels import REGION_TO_ANATOMICAL_REGION
from .public_visualization import draw_visualization as draw_public_viz
from .results import DXAImageResult, DXATzResult, TZ_COLUMNS
from .subpipelines.hip.sub_pipeline import HipSubPipeline
from .subpipelines.spine.sub_pipeline import SpineSubPipeline
from .utils.image_io import read_and_normalize_image
from .visualizations import draw_private_visualization as draw_private_viz


def _dicom_ids(path: Path) -> tuple[str, str]:
    """StudyInstanceUID и SOPInstanceUID. Битый заголовок не роняет пакет."""
    try:
        dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return "", ""
    study_uid = getattr(dataset, "StudyInstanceUID", "") or ""
    image_uid = getattr(dataset, "SOPInstanceUID", "") or ""
    return str(study_uid), str(image_uid)


class DXAPipeline:
    def __init__(self, config: DXAPipelineConfig | None = None):
        self.config = config or DXAPipelineConfig()
        self.region_clf = RegionClassificationContext(self.config.region)
        self.spine = SpineSubPipeline(self.config.spine)
        self.hip = HipSubPipeline(self.config.hip)

    def _run_image(self, image: np.ndarray) -> DXAImageResult:
        region, region_confidence = self.region_clf.predict(image)
        match region:
            case "spine":
                spine = self.spine.run(image)
                return DXAImageResult(
                    anatomical_region=REGION_TO_ANATOMICAL_REGION["spine"],
                    quality_class=spine.quality_class,
                    violation_type=spine.violation_type,
                    quality_prob=spine.quality_prob,
                    region_name="spine",
                    region_confidence=region_confidence,
                    violation_reasons=spine.violation_reasons,
                    xlsx_clf=spine.xlsx_clf,
                    visualization=spine.visualization,
                )
            case "hip":
                hip = self.hip.run(image)
                return DXAImageResult(
                    anatomical_region=REGION_TO_ANATOMICAL_REGION["hip"],
                    quality_class=hip.quality_class,
                    violation_type=hip.violation_type,
                    quality_prob=hip.quality_prob,
                    region_name="hip",
                    region_confidence=region_confidence,
                    violation_reasons=hip.violation_reasons,
                    xlsx_clf=hip.xlsx_clf,
                    visualization=hip.visualization,
                )
            case _:
                raise ValueError(f"Неизвестная зона: {region}")

    def run(self, image: np.ndarray) -> DXAImageResult:
        started = time.perf_counter()
        result = self._run_image(image)
        return result.model_copy(update={"time_of_processing": round(time.perf_counter() - started, 3)})

    def create_result(self, run_result: DXAImageResult) -> dict[str, str | int | float]:
        """Строка итоговой таблицы по ТЗ 2.5. Поля берутся из результата кадра."""
        return DXATzResult(
            path_to_study=run_result.path_to_study,
            study_uid=run_result.study_uid,
            image_uid=run_result.image_uid,
            anatomical_region=run_result.anatomical_region,
            quality_class=run_result.quality_class,
            violation_type=run_result.violation_type,
            processing_status=run_result.processing_status,
            time_of_processing=run_result.time_of_processing,
            quality_prob=run_result.quality_prob,
        ).to_row()

    def run_from_path(self, dicom_path: str | Path) -> DXAImageResult:
        started = time.perf_counter()
        path = Path(dicom_path)
        study_uid, image_uid = _dicom_ids(path)
        image = read_and_normalize_image(str(path))
        result = self._run_image(image)
        return result.model_copy(
            update={
                "path_to_study": str(path),
                "study_uid": study_uid,
                "image_uid": image_uid,
                "time_of_processing": round(time.perf_counter() - started, 3),
            }
        )

    def run_batch(self, paths: list[str]) -> tuple[pd.DataFrame, list[DXAImageResult]]:
        """Один путь — одна строка. Упавший кадр остаётся в таблице со статусом Failure."""
        results: list[DXAImageResult] = []
        for path in paths:
            started = time.perf_counter()
            try:
                results.append(self.run_from_path(path))
            except Exception:
                study_uid, image_uid = _dicom_ids(Path(path))
                results.append(
                    DXAImageResult(
                        path_to_study=path,
                        study_uid=study_uid,
                        image_uid=image_uid,
                        anatomical_region="",
                        quality_class=0,
                        violation_type="",
                        quality_prob=0.0,
                        region_name=None,
                        region_confidence=0.0,
                        processing_status="Failure",
                        time_of_processing=round(time.perf_counter() - started, 3),
                    )
                )
        frame = pd.DataFrame([self.create_result(item) for item in results], columns=list(TZ_COLUMNS))
        return frame, results

    def draw_private_visualization(self, run_result: DXAImageResult) -> np.ndarray:
        return draw_private_viz(run_result)

    def draw_visualization(self, run_result: DXAImageResult) -> np.ndarray:
        return draw_public_viz(run_result)
