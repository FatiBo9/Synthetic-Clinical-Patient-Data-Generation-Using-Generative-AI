from __future__ import annotations
from typing import Any, Mapping
from ..core.class_assigner import assign_class
from ..core.patient_schema import ClassLabel

def check_class_coherence(
    p: Mapping[str, Any],
    expected_class: ClassLabel,
) -> bool:
    #retourne True si `assign_class(p)` = `expected_class`.
    return assign_class(p) == expected_class
