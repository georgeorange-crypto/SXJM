"""
pytest 配置与共享夹具。把项目根加入 sys.path，使 `from src...` 可用。
"""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.domain.types import ProblemConstants, RobotConstants  # noqa: E402


@pytest.fixture
def problem() -> ProblemConstants:
    return ProblemConstants()


@pytest.fixture
def robot() -> RobotConstants:
    return RobotConstants()
