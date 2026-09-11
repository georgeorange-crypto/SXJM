"""Make ``import way4`` and ``import sxjm_core`` work without an editable
install (src-layout, §13). ``sxjm_core`` is the shared geometry core living at
the SXJM repo root, a sibling of Way4/."""

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
_SRC = _HERE.parent.parent / "src"          # Way4/src  (the way4 package)
_REPO_ROOT = _HERE.parent.parent.parent     # SXJM/      (sxjm_core lives here)
for _p in (_SRC, _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
