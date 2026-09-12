"""throwaway: dump candidate set + Q values when ch1,ch2 are DETECTED."""
import sys
sys.path.insert(0, r"D:\George\SX\Way4\src")
sys.path.insert(0, r"D:\George\SX")

from way4.executor import Way3EngineAdapter, load_way3_environment
from way4.pipeline import Way4Pipeline
from way4.belief import ChannelStatus

mod = load_way3_environment()
Jammer, Case = mod.Jammer, mod.Case
case = Case([Jammer(1, 600.0, 0.0, 1400.0, "omni"),
            Jammer(2, -600.0, 300.0, 1400.0, "omni")], field=mod.ConstantField(1.0))
eng = mod.Engine(case); eng.enter()
pipe = Way4Pipeline(Way3EngineAdapter(eng), n_channels=20, max_steps=40)

# run a few steps to get ch1,ch2 detected
for step in range(5):
    pipe.certificate.apply_certifications(pipe.belief, force=True)
    cands = pipe.generator.generate(pipe.belief, pipe.certificate, pipe.state, scan_mode=pipe._scan_mode())
    macro = pipe.planner.plan(pipe.belief, pipe.certificate, pipe.state, cands).best
    res = pipe.executor.execute(macro, pipe.state, float("inf"))
    pipe.state = res.state

# now dump status of ch1,ch2
for c in (1, 2):
    b = pipe.belief[c]
    print(f"ch{c}: status={b.status.value} mec_r={b.mec_radius:.1f} diam={b.diameter:.1f} "
          f"clearable={b.is_clearable} nbearings={len(b.bearings)} F_c_verts={len(b.F_c) if b.F_c else 0}")

print("\n-- candidates at this state --")
cands = pipe.generator.generate(pipe.belief, pipe.certificate, pipe.state, scan_mode=pipe._scan_mode())
res_plan = pipe.planner.plan(pipe.belief, pipe.certificate, pipe.state, cands)
for ev in sorted(res_plan.evaluations, key=lambda e: e.q_value):
    a = ev.candidate
    tag = ""
    if a.clear_channel: tag = f"clear_ch{a.clear_channel}"
    elif "refine_channel" in a.meta: tag = f"refine_ch{a.meta['refine_channel']}"
    print(f"  {a.action_type.value:9s} {tag:12s} tgt=({a.target[0]:7.1f},{a.target[1]:7.1f}) "
          f"nchan={len(a.scan_channels)} C={ev.immediate_cost:8.1f} J={ev.future_cost:10.1f} Q={ev.q_value:10.1f}")
print(f"\nBEST: {res_plan.best.action_type.value} "
      f"{'refine_ch'+str(res_plan.best.meta.get('refine_channel')) if 'refine_channel' in res_plan.best.meta else ''}")
