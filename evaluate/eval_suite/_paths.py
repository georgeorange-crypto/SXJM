"""
让 evaluate 包能 import Way3 的 jammerhunt（同一仓库根下的兄弟目录）。

evaluate/ 与 Way3/ 都在仓库根 D:\\George\\SX 下。jammerhunt 是纯标准库包，
本评价体系复用它的 environment（真值来源）、agent.Hunter（OURS 策略）、geometry。
import 本模块即把 Way3 目录挂到 sys.path。
"""

from __future__ import annotations

import os
import sys


def repo_root() -> str:
    # .../evaluate/eval_suite/_paths.py → 上三级 = 仓库根
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ensure_jammerhunt_on_path() -> str:
    """把 Way3 目录加入 sys.path，返回该目录。"""
    way3 = os.path.join(repo_root(), "Way3")
    if way3 not in sys.path:
        sys.path.insert(0, way3)
    return way3


ensure_jammerhunt_on_path()
