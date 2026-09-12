from way4.belief import ChannelStatus
from way4.core import MacroActionType, get_allowed_actions


def test_unified_action_mask_lifecycle():
    assert get_allowed_actions(ChannelStatus.CLEARED) == frozenset()
    assert MacroActionType.CLEAR in get_allowed_actions(ChannelStatus.LOCALIZED)
    assert MacroActionType.REFINE in get_allowed_actions(ChannelStatus.DETECTED)
    assert MacroActionType.EXPLORE in get_allowed_actions(ChannelStatus.UNKNOWN)
