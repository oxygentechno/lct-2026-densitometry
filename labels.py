"""Закрытый словарь ТЗ + детальные причины spine/hip-правил."""

from __future__ import annotations

from typing import Literal

RotationState = Literal["ok", "over", "under"]
HipSide = Literal["left", "right"]
RegionName = Literal["spine", "hip"]

VERTEBRA_LABELS_TO_IDS: dict[str, int] = {
    "Th12": 1,
    "L1": 2,
    "L2": 3,
    "L3": 4,
    "L4": 5,
    "L5": 6,
}

REQUIRED_VERTEBRAE: tuple[str, ...] = ("Th12", "L1", "L2", "L3", "L4", "L5")
L14_VERTEBRAE: tuple[str, ...] = ("L1", "L2", "L3", "L4")
AXIS_VERTEBRAE: tuple[str, ...] = ("L1", "L5")

REGION_LABELS_TO_IDS: dict[str, int] = {
    "spine": 0,
    "hip": 1,
}

HIP_PARTS_LABELS_TO_IDS: dict[str, int] = {
    "greater_trochanter": 1,
    "femoral_neck": 2,
    "femoral_head": 3,
    "ischium": 4,
    "femoral_shaft": 5,
    "lesser_trochanter": 6,
}

HIP_ISCHIUM_LABELS_TO_IDS: dict[str, int] = {"ischium": 1}
HIP_PROXIMAL_LABELS_TO_IDS: dict[str, int] = {
    "femoral_neck": 1,
    "femoral_head": 2,
    "femoral_shaft": 3,
}
HIP_GT_LABELS_TO_IDS: dict[str, int] = {"greater_trochanter": 1}
HIP_LT_LABELS_TO_IDS: dict[str, int] = {"lesser_trochanter": 1}

HIP_POSITIONING_PARTS: tuple[str, ...] = ("greater_trochanter", "femoral_neck", "ischium")
HIP_ROI_PARTS: tuple[str, ...] = (
    "greater_trochanter",
    "femoral_neck",
    "femoral_head",
    "ischium",
    "femoral_shaft",
)
ROTATION_POINT_ORDER: tuple[str, ...] = ("lt_tip", "shaft_medial_a", "shaft_medial_b")

SPINE_XLSX_VIOLATIONS: tuple[str, ...] = ("spine_layout", "spine_axis", "spine_artifacts")
HIP_XLSX_VIOLATIONS: tuple[str, ...] = ("hip_posrot", "hip_roi")

# Закрытый словарь violation_type из разъяснений организатора (вопрос 6).
VIOLATION_LAYOUT = "Некорректная укладка"
VIOLATION_AXIS = "Не выравнена ось позвоночника"
VIOLATION_FOREIGN = "Присутствуют посторонние предметы"
VIOLATION_ROI = "Некорректная область интереса"

# У каждой зоны свой допустимый список; значения вне него в отчёт не попадают.
REGION_VIOLATIONS: dict[RegionName, tuple[str, ...]] = {
    "spine": (VIOLATION_LAYOUT, VIOLATION_AXIS, VIOLATION_FOREIGN),
    "hip": (VIOLATION_LAYOUT, VIOLATION_ROI),
}

# anatomical_region из разъяснений организатора (вопрос 15). Сторона не указывается.
REGION_TO_ANATOMICAL_REGION: dict[RegionName, str] = {
    "spine": "Поясничный отдел позвоночника",
    "hip": "Проксимальный отдел бедра",
}

VIOLATION_SEPARATOR = "; "

REASON_ILIAC = "iliac не виден"
REASON_T12_HEIGHT = "Th12 виден менее чем наполовину"
REASON_SPINE_CENTER = "Позвоночник не по центру кадра"
REASON_MISSING_VERTEBRAE = "Не все позвонки Th12–L5"
REASON_AXIS = VIOLATION_AXIS
REASON_FOREIGN = VIOLATION_FOREIGN

REASON_MISSING_HIP_PARTS = "Нет GT/шейки/седалищной"
REASON_OVER_ROTATION = "Переротация: малый вертел не виден"
REASON_UNDER_ROTATION = "Недоротация: малый вертел слишком выступает"
REASON_SHAFT_AXIS = "Ось бедра не вдоль линии сканирования"
REASON_FOV_ISCHIUM = "Менее 3 см вниз от седалищной"
REASON_FOV_GT_UP = "Менее 3 см вверх от большого вертела"
REASON_FOV_GT = "Менее 2 см латерально от большого вертела"
REASON_PROTHESIS = "Протез/металл в зоне измерения"

REASON_TO_CLOSED: dict[str, str] = {
    REASON_ILIAC: VIOLATION_LAYOUT,
    REASON_T12_HEIGHT: VIOLATION_LAYOUT,
    REASON_SPINE_CENTER: VIOLATION_LAYOUT,
    REASON_MISSING_VERTEBRAE: VIOLATION_LAYOUT,
    REASON_AXIS: VIOLATION_AXIS,
    REASON_FOREIGN: VIOLATION_FOREIGN,
    # ТЗ 2.3: позиционирование и ротация — это укладка.
    REASON_MISSING_HIP_PARTS: VIOLATION_LAYOUT,
    REASON_OVER_ROTATION: VIOLATION_LAYOUT,
    REASON_UNDER_ROTATION: VIOLATION_LAYOUT,
    REASON_SHAFT_AXIS: VIOLATION_LAYOUT,
    # ТЗ 2.3: запас поля 3/3/2 см — это критерий корректности области интереса.
    REASON_FOV_ISCHIUM: VIOLATION_ROI,
    REASON_FOV_GT_UP: VIOLATION_ROI,
    REASON_FOV_GT: VIOLATION_ROI,
    # Для бедра «посторонних предметов» в словаре нет: протез делает ROI непригодной.
    REASON_PROTHESIS: VIOLATION_ROI,
}

CLOSED_VIOLATION_ORDER: tuple[str, ...] = (
    VIOLATION_LAYOUT,
    VIOLATION_AXIS,
    VIOLATION_FOREIGN,
    VIOLATION_ROI,
)


def closed_violations_from_reasons(reasons: list[str], region: RegionName | None = None) -> list[str]:
    """Причины → значения violation_type, отфильтрованные словарём зоны."""
    allowed = CLOSED_VIOLATION_ORDER if region is None else REGION_VIOLATIONS[region]
    closed: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        item = REASON_TO_CLOSED[reason]
        if item in seen or item not in allowed:
            continue
        seen.add(item)
        closed.append(item)
    closed.sort(key=lambda name: CLOSED_VIOLATION_ORDER.index(name))
    return closed


def format_violation_type(closed_violations: list[str]) -> str:
    """Пустая строка при отсутствии нарушений, иначе перечисление через «; »."""
    return VIOLATION_SEPARATOR.join(closed_violations)
