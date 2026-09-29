from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from .config import DXAPipelineConfig
    from .pipeline import DXAPipeline
    from .results import DXAImageResult

__all__ = ["DXAPipeline", "DXAPipelineConfig", "draw_visualization", "run_batch"]


def run_batch(
    paths: list[str],
    config: DXAPipelineConfig | None = None,
) -> tuple[pd.DataFrame, list[DXAImageResult]]:
    from .pipeline import DXAPipeline

    return DXAPipeline(config).run_batch(paths)


def draw_visualization(run_result: DXAImageResult) -> np.ndarray:
    from .public_visualization import draw_visualization as draw

    return draw(run_result)


def __getattr__(name: str):
    match name:
        case "DXAPipeline":
            from .pipeline import DXAPipeline as pipeline_cls

            return pipeline_cls
        case "DXAPipelineConfig":
            from .config import DXAPipelineConfig as config_cls

            return config_cls
        case _:
            raise AttributeError(name)