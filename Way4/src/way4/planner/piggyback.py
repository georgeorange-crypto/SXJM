"""Joint time objective for dedicated and piggyback REFINE variants."""

from dataclasses import dataclass
from typing import Iterable

WAIT_FOR_BACKBONE = "WAIT_FOR_BACKBONE"


@dataclass(frozen=True)
class RefineVariantEvaluation:
    candidate: object
    delta_route_s: float
    measure_time_s: float
    expected_remaining_s: float

    @property
    def total_expected_s(self) -> float:
        return self.delta_route_s + self.measure_time_s + self.expected_remaining_s


def evaluate_refine_variant(candidate, *, delta_route_s: float,
                            measure_time_s: float,
                            expected_remaining_s: float) -> RefineVariantEvaluation:
    """Evaluate J(q)=ΔT_route+T_measure+E[T_remaining]."""
    values = (delta_route_s, measure_time_s, expected_remaining_s)
    if any(float(value) < 0.0 for value in values):
        raise ValueError("REFINE time components must be non-negative")
    return RefineVariantEvaluation(candidate, *(float(value) for value in values))


def choose_refine_variant(evaluations: Iterable[RefineVariantEvaluation]):
    """Choose the minimum joint expected completion-time variant."""
    options = tuple(evaluations)
    if not options:
        raise ValueError("at least one REFINE variant is required")
    return min(options, key=lambda item: item.total_expected_s)


def mark_wait_for_backbone(candidate, *, backbone_index: int,
                           detour_s: float, max_detour_s: float):
    """Annotate a candidate as waiting for a near future backbone visit."""
    if detour_s < 0 or max_detour_s < 0:
        raise ValueError("detours must be non-negative")
    if detour_s <= max_detour_s:
        candidate.meta.update({"task_status": WAIT_FOR_BACKBONE,
                               "backbone_index": int(backbone_index),
                               "backbone_route_delta": float(detour_s)})
    return candidate
