"""throwaway: trace a regression seed step-by-step to find the stall signature."""
import sys
from pathlib import Path
_H = Path(__file__).resolve()
for _p in (_H.parents[1] / "src", _H.parents[2], _H.parents[2] / "Way3"):
    sys.path.insert(0, str(_p))

from jammerhunt import environment as env
from way4.belief import ChannelStatus
from way4.executor import Way3EngineAdapter
from way4.pipeline import Way4Pipeline

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1017
MAXSTEP = int(sys.argv[2]) if len(sys.argv) > 2 else 80

case = env.generate_case(seed=SEED, problem=3, field_kind="smooth")
present = {j.channel for j in case.jammers}
print(f"seed={SEED} total={case.total} present_channels={sorted(present)}")
engine = env.Engine(case); engine.enter()
pipe = Way4Pipeline(Way3EngineAdapter(engine), n_channels=20, max_steps=MAXSTEP)

last_key = None
stall = 0
for step in range(MAXSTEP):
    pipe.certificate.apply_certifications(pipe.belief, force=True)
    if pipe.belief.all_resolved():
        print(f"[{step}] ALL RESOLVED"); break
    cands = pipe.generator.generate(pipe.belief, pipe.certificate, pipe.state,
                                    scan_mode=pipe._scan_mode())
    if not cands:
        print(f"[{step}] NO CANDIDATES"); break
    if not pipe._exit_allowed():
        cands = [c for c in cands if c.action_type.value != "EXIT"]
    plan = pipe.planner.plan(pipe.belief, pipe.certificate, pipe.state, cands)
    macro = plan.best
    key = pipe._progress_key()
    ev = next((e for e in plan.evaluations if e.candidate is macro), None)
    tag = ""
    if macro.clear_channel:
        tag = f"clr{macro.clear_channel}"
    elif "refine_channel" in macro.meta:
        tag = f"ref{macro.meta['refine_channel']}(mec={pipe.belief[macro.meta['refine_channel']].mec_radius:.0f})"
    qtxt = f"Q={ev.q_value:.1f} C={ev.immediate_cost:.1f} J={ev.future_cost:.1f}" if ev else ""
    print(f"[{step:3d}] {macro.action_type.value:8s} {tag:14s} "
          f"tgt=({macro.target[0]:6.0f},{macro.target[1]:6.0f}) "
          f"nch={len(macro.scan_channels)} cov={macro.certificate_gain:.4f} {qtxt} "
          f"| key={key}")
    res = pipe.executor.execute(macro, pipe.state, float("inf"))
    pipe.state = res.state
    if key == last_key:
        stall += 1
    else:
        stall = 0
        last_key = key
    if stall >= 6:
        print(f"    >>> STALLED (key frozen {stall} steps)")
        break

print("\n== channel belief ==")
for c in range(1, 21):
    b = pipe.belief[c]
    if b.status == ChannelStatus.UNKNOWN and c not in present:
        continue  # skip boring unknown-absent
    isp = "PRES" if c in present else "abs "
    absc = pipe.certificate.is_absent_certified(c, force=True)
    print(f"ch{c:2d} [{isp}] status={b.status.value:16s} mec_r={b.mec_radius:8.1f} "
          f"clearable={b.is_clearable!s:5s} nbear={len(b.bearings)} "
          f"ndisc={len(b.negative_discs)} abs_cert={absc!s:5s}")

# where are the present sources vs robot?
print(f"\nrobot at ({pipe.state.x:.0f},{pipe.state.y:.0f}) ch={pipe.state.channel}")
print("present source truth:")
for j in case.jammers:
    print(f"  ch{j.channel:2d} at ({j.x:7.0f},{j.y:7.0f}) R_eff={j.r_eff:.0f} kind={j.kind}")
