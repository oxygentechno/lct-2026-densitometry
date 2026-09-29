"""Загрузка Lightning-чекпоинтов: только ключи `model.*`."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

PIPELINE_DIR = Path(__file__).resolve(strict=True).parent
REPO_ROOT = PIPELINE_DIR.parents[1]

DEFAULT_CHECKPOINT_DIRS: dict[str, Path] = {
    "region": REPO_ROOT / "experiments/dxa/region_classification/1_base/checkpoints",
    "vertebra": REPO_ROOT / "experiments/dxa/vertebra_semantic_segmentation/1_base/checkpoints",
    "iliac": REPO_ROOT / "experiments/dxa/iliac_semantic_segmentation/1_base/checkpoints",
    "foreign": REPO_ROOT / "experiments/dxa/foreign_semantic_segmentation/1_base/checkpoints",
    "axis": REPO_ROOT / "experiments/dxa/vertebra_axis_keypoint_detection/1_base/checkpoints",
    "hip_ischium": REPO_ROOT / "experiments/dxa/hip_ischium_semantic_segmentation/1_base/checkpoints",
    "hip_proximal": REPO_ROOT / "experiments/dxa/hip_proximal_semantic_segmentation/1_base/checkpoints",
    "hip_gt": REPO_ROOT / "experiments/dxa/hip_greater_trochanter_semantic_segmentation/1_base/checkpoints",
    "hip_lt": REPO_ROOT / "experiments/dxa/hip_lesser_trochanter_semantic_segmentation/1_base/checkpoints",
    "hip_bone": REPO_ROOT / "experiments/dxa/hip_bone_semantic_segmentation/1_base/checkpoints",
    "hip_prothesis": REPO_ROOT / "experiments/dxa/hip_prothesis_semantic_segmentation/1_base/checkpoints",
    "hip_lt_tip": REPO_ROOT / "experiments/dxa/hip_lt_tip_keypoint_detection/1_base/checkpoints",
    "hip_shaft_a": REPO_ROOT / "experiments/dxa/hip_shaft_medial_a_keypoint_detection/1_base/checkpoints",
    "hip_shaft_b": REPO_ROOT / "experiments/dxa/hip_shaft_medial_b_keypoint_detection/1_base/checkpoints",
    "spine_xlsx_clf": REPO_ROOT / "experiments/dxa/spine_quality_classification/1_optuna/checkpoints",
    "hip_xlsx_clf": REPO_ROOT / "experiments/dxa/hip_quality_classification/1_optuna/checkpoints",
}


def resolve_device(requested: str) -> str:
    match requested.startswith("cuda") and not torch.cuda.is_available():
        case True:
            return "cpu"
        case False:
            return requested


def _localize_path(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        return PIPELINE_DIR / path
    return path


def _latest_checkpoint(ckpt_dir: Path) -> Path | None:
    if not ckpt_dir.exists():
        return None
    lasts = sorted(ckpt_dir.rglob("last.ckpt"), key=lambda item: item.stat().st_mtime, reverse=True)
    if lasts:
        return lasts[0]
    ckpts = sorted(ckpt_dir.rglob("*.ckpt"), key=lambda item: item.stat().st_mtime, reverse=True)
    return ckpts[0] if ckpts else None


def resolve_checkpoint(explicit: Path | None, experiment_key: str) -> Path:
    if explicit is not None:
        path = _localize_path(Path(explicit))
        if path.exists():
            return path
        fallback = _latest_checkpoint(DEFAULT_CHECKPOINT_DIRS[experiment_key])
        if fallback is not None:
            return fallback
        raise FileNotFoundError(f"Нет чекпоинта: {explicit}")

    ckpt_dir = DEFAULT_CHECKPOINT_DIRS[experiment_key]
    found = _latest_checkpoint(ckpt_dir)
    if found is not None:
        return found
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Нет папки чекпоинтов {ckpt_dir}. Передай явный путь к *.ckpt в конфиге пайплайна.")
    raise FileNotFoundError(f"В {ckpt_dir} нет *.ckpt")


def load_lightning_weights(model: nn.Module, checkpoint_path: Path, device: str) -> nn.Module:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    match payload:
        case dict() if "state_dict" in payload:
            state = {
                key.removeprefix("model."): value
                for key, value in payload["state_dict"].items()
                if key.startswith("model.")
            }
        case dict():
            state = {
                key.removeprefix("model."): value
                for key, value in payload.items()
                if not key.startswith("criterion.")
            }
        case _:
            raise ValueError(f"Непонятный чекпоинт: {checkpoint_path}")
    if not state:
        raise ValueError(f"В {checkpoint_path} нет весов model.* — это не Lightning-чекпоинт модели")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
