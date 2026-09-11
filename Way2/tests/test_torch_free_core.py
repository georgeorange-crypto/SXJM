"""The mathematical core must run without torch being imported.

Architecture principle: env + geometry + candidates + heuristic + evaluation and
the default :class:`~radio_rl.pipeline.Pipeline` never import torch; the tensor
stack (features/models/training) is pulled in lazily only on the learnable path
(``needs_features``). We check this in a *fresh subprocess* so the verdict does
not depend on whether the surrounding test session already imported torch.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_default_pipeline_never_imports_torch():
    script = textwrap.dedent(
        """
        import sys
        from radio_rl.core.config import compose
        from radio_rl.pipeline import Pipeline

        cfg = compose()                      # default: heuristic greedy_math agent
        pipe = Pipeline(cfg)
        pipe.run_episode(seed=0, max_steps=500)

        leaked = sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))
        assert not leaked, "core imported torch: " + repr(leaked)
        print("TORCH_FREE_OK")
        """
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "TORCH_FREE_OK" in proc.stdout
