"""让 tests 直接 import eval_suite（evaluate/ 加入 sys.path）。"""

import os
import sys

_EVAL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EVAL_ROOT not in sys.path:
    sys.path.insert(0, _EVAL_ROOT)
