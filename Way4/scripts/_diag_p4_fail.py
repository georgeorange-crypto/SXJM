"""Diagnose the P4 Way4 regression: is a hidden directional source wrongly
certified ABSENT via the unsound omni arbitrary-disc-cover (禁止5)?"""
from __future__ import annotations
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[1] / "src", _HERE.parents[2], _HERE.parents[2] / "Way3"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from jammerhunt import environment as env
from way4.executor import Way3EngineAdapter
from way4.pipeline import Way4Pipeline
from way4.belief import ChannelStatus

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 2000

case = env.generate_case(seed=SEED, problem=4, field_kind="smooth")
engine = env.Engine(case); engine.enter()
pipe = Way4Pipeline(Way3EngineAdapter(engine), n_channels=20, problem=4, max_steps=4000)
res = pipe.run()

# ground truth: channel -> (kind, cleared)
truth = {j.channel: (j.kind, j.cleared, j.direction_deg if j.kind == "dir" else None)
         for j in case.jammers}
print(f"seed={SEED}  cleared={case.cleared_count}/{case.total}  "
      f"exited={res.exited} error={res.error} steps={res.steps}")
print(f"  dir sources: {[c for c,(k,_,_) in truth.items() if k=='dir']}  "
      f"omni: {[c for c,(k,_,_) in truth.items() if k=='omni']}")
print("  --- uncleared / mis-certified channels ---")
for c in range(1, 21):
    b = pipe.belief[c]
    src = pipe.certificate.certificate_source(c)
    if c in truth:
        kind, cleared, dirdeg = truth[c]
        if not cleared:
            print(f"  ch{c:2d} TRUE {kind} dir={dirdeg} -> belief={b.status.value} "
                  f"cert_source={src}  <== UNCLEARED SOURCE STILL PRESENT")
    else:
        if b.status == ChannelStatus.ABSENT_CERTIFIED and src is not None:
            pass  # correctly absent (no source)
