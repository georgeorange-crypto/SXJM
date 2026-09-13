from way4.core import MacroActionType, MacroCandidate
from way4.planner import WAIT_FOR_BACKBONE, mark_wait_for_backbone


def test_near_backbone_refine_can_wait_instead_of_returning():
    candidate = MacroCandidate(MacroActionType.REFINE, (1., 2.), scan_channels=(1,))
    mark_wait_for_backbone(candidate, backbone_index=7, detour_s=3.0, max_detour_s=5.0)
    assert candidate.meta["task_status"] == WAIT_FOR_BACKBONE
    assert candidate.meta["backbone_index"] == 7


def test_far_backbone_does_not_get_wait_status():
    candidate = MacroCandidate(MacroActionType.REFINE, (1., 2.), scan_channels=(1,))
    mark_wait_for_backbone(candidate, backbone_index=7, detour_s=6.0, max_detour_s=5.0)
    assert "task_status" not in candidate.meta
