"""让 `import jammerhunt` 在任意工作目录下都可用：把 Way3 根（tests/ 的上级）加入 sys.path。

这样 `pytest`（无论从哪运行）与 `python tests/test_xxx.py` 都能导入本包。仅标准库。
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
