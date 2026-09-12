"""Trace WHY a detected directional source never clears (no_progress_stall).

Instruments one channel across a Way4 P4 episode: every NBV viewpoint chosen, each
measurement's result, and the MEC radius after — to confirm the hypothesis that the
omni MinimaxNBV sends the robot to out-of-arc viewpoints that return NO_SIGNAL, so
the MEC never shrinks below the clear threshold (18 m) and the channel livelocks.
"""
from __future__ import annotations
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[1] / "src", _HERE.parents[2], _HERE.parents[2] / "Way3"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from jammerhunt import environment as env
from way4.executor import Way3EngineAdapter
from way4.executor.way3_engine import Way3EngineAdapter as _Adapter
from way4.pipeline import Way4Pipeline
from way4.sensing.nbv import MinimaxNBV

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
WATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 11

case = env.generate_case(seed=SEED, problem=4, field_kind="smooth")
engine = env.Engine(case); engine.enter()

# ground truth for the watched channel
gt = None
for j in case.jammers:
    if j.channel == WATCH:
        gt = j
print(f"seed={SEED} watch=ch{WATCH}")
if gt is not None:
    print("  ground truth:", {k: getattr(gt, k) for k in vars(gt)})

log = []

_orig_measure = _Adapter.measure
def _measure(self, x, y, channel):
    obs, vts = _orig_measure(self, x, y, channel)
    if channel == WATCH:
        kind = obs.kind.name if obs is not None else "DEADLINE"
        log.append(("MEASURE", round(x, 1), round(y, 1), kind))
    return obs, vts
_Adapter.measure = _measure

_orig_choose = MinimaxNBV.choose
def _choose(self, channel, robot_pos):
    res = _orig_choose(self, channel, robot_pos)
    if getattr(channel, "channel", None) == WATCH and res is not None:
        log.append(("NBV", round(res.point[0], 1), round(res.point[1], 1),
                    f"u={res.worst_case_diameter:.1f} cur={res.current_diameter:.1f} "
                    f"improved={res.improved} mec_r={channel.mec_radius:.1f}"))
    return res
MinimaxNBV.choose = _choose

pipe = Way4Pipeline(Way3EngineAdapter(engine), n_channels=20, problem=4, max_steps=4000)
res = pipe.run()

print(f"  result: cleared={case.cleared_count}/{case.total} exited={res.exited} "
      f"error={res.error} steps={res.steps}")
b = pipe.belief[WATCH]
print(f"  ch{WATCH} final: status={b.status.value} mec_r={b.mec_radius:.1f} "
      f"n_bearings={len(b.bearings)} n_neg_discs={len(b.negative_discs)} "
      f"cleared_truth={gt.cleared if gt else '?'}")
print(f"  --- ch{WATCH} event trace ({len(log)} events) ---")
for e in log:
    print("   ", e)
