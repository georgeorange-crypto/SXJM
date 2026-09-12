"""throwaway: dump the stuck state on seed 1003 (channels + candidate Q-values)."""
import sys
from pathlib import Path
_H = Path(__file__).resolve()
for _p in (_H.parents[1] / "src", _H.parents[2], _H.parents[2] / "Way3"):
    sys.path.insert(0, str(_p))

from jammerhunt import environment as env
from way4.executor import Way3EngineAdapter
from way4.pipeline import Way4Pipeline

case = env.generate_case(seed=1003, problem=3, field_kind="smooth")
present = {j.channel for j in case.jammers}
engine = env.Engine(case); engine.enter()
pipe = Way4Pipeline(Way3EngineAdapter(engine), n_channels=20, max_steps=30)

# run until stuck (step 25ish)
last_tgt = None; repeats = 0
for step in range(30):
    pipe.certificate.apply_certifications(pipe.belief, force=True)
    if pipe.belief.all_resolved():
        print("resolved before stuck"); break
    cands = pipe.generator.generate(pipe.belief, pipe.certificate, pipe.state, scan_mode=pipe._scan_mode())
    res_plan = pipe.planner.plan(pipe.belief, pipe.certificate, pipe.state, cands)
    macro = res_plan.best
    tgt = (round(macro.target[0]), round(macro.target[1]))
    if tgt == last_tgt:
        repeats += 1
    else:
        repeats = 0
    last_tgt = tgt
    if repeats >= 2:
        print(f"STUCK at step {step}, target {tgt}, robot at ({pipe.state.x:.0f},{pipe.state.y:.0f}) ch={pipe.state.channel}\n")
        break
    res = pipe.executor.execute(macro, pipe.state, float("inf"))
    pipe.state = res.state

print("== channel belief ==")
for c in range(1, 21):
    b = pipe.belief[c]
    isp = "PRES" if c in present else "abs "
    absc = pipe.certificate.is_absent_certified(c, force=True)
    print(f"ch{c:2d} [{isp}] status={b.status.value:16s} mec_r={b.mec_radius:8.1f} "
          f"clearable={b.is_clearable!s:5s} nbear={len(b.bearings)} abs_cert={absc!s:5s}")

print("\n== candidates (sorted by Q) ==")
cands = pipe.generator.generate(pipe.belief, pipe.certificate, pipe.state, scan_mode=pipe._scan_mode())
res_plan = pipe.planner.plan(pipe.belief, pipe.certificate, pipe.state, cands)
for ev in sorted(res_plan.evaluations, key=lambda e: e.q_value):
    a = ev.candidate
    tag = ""
    if a.clear_channel: tag = f"clr_ch{a.clear_channel}"
    elif "refine_channel" in a.meta: tag = f"ref_ch{a.meta['refine_channel']}"
    print(f"  {a.action_type.value:9s} {tag:10s} tgt=({a.target[0]:7.0f},{a.target[1]:7.0f}) "
          f"nch={len(a.scan_channels):2d} covgain={a.certificate_gain:7.4f} "
          f"C={ev.immediate_cost:8.1f} J={ev.future_cost:9.1f} Q={ev.q_value:9.1f}")
print(f"\nscan_mode={pipe._scan_mode().value}  unknown={pipe.belief.unknown_channels()}")
