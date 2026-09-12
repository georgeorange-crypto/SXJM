"""throwaway diagnostic: time each pipeline tick on the 2-source case."""
import sys, time
sys.path.insert(0, r"D:\George\SX\Way4\src")
sys.path.insert(0, r"D:\George\SX")

from way4.executor import Way3EngineAdapter, load_way3_environment
from way4.pipeline import Way4Pipeline
from way4.core import RobotState
from way4.channels import SchedulerMode

mod = load_way3_environment()
Jammer, Case = mod.Jammer, mod.Case
case = Case([Jammer(1, 600.0, 0.0, 1400.0, "omni"),
            Jammer(2, -600.0, 300.0, 1400.0, "omni")], field=mod.ConstantField(1.0))
eng = mod.Engine(case); eng.enter()

pipe = Way4Pipeline(Way3EngineAdapter(eng), n_channels=20, max_steps=40)

# instrument by manually stepping
t0 = time.perf_counter()
for step in range(40):
    tc0 = time.perf_counter()
    pipe.certificate.apply_certifications(pipe.belief, force=True)
    tc1 = time.perf_counter()
    if pipe.belief.all_resolved():
        print(f"step {step}: ALL RESOLVED"); break
    tg0 = time.perf_counter()
    cands = pipe.generator.generate(pipe.belief, pipe.certificate, pipe.state, scan_mode=pipe._scan_mode())
    tg1 = time.perf_counter()
    if not cands:
        print(f"step {step}: NO CANDIDATES"); break
    tp0 = time.perf_counter()
    res_plan = pipe.planner.plan(pipe.belief, pipe.certificate, pipe.state, cands)
    tp1 = time.perf_counter()
    macro = res_plan.best
    te0 = time.perf_counter()
    res = pipe.executor.execute(macro, pipe.state, float("inf"))
    te1 = time.perf_counter()
    pipe.state = res.state
    unknown = len(pipe.belief.unknown_channels())
    resolved = sum(1 for c in range(1,21) if pipe.belief[c].is_resolved)
    print(f"step {step}: {macro.action_type.value:9s} tgt=({macro.target[0]:.0f},{macro.target[1]:.0f}) "
          f"nchan={len(macro.scan_channels)} | cert={tc1-tc0:.3f} gen={tg1-tg0:.3f} plan={tp1-tp0:.3f} exec={te1-te0:.3f} "
          f"| unknown={unknown} resolved={resolved} present={pipe.certificate.present_count()} "
          f"vt={pipe.state.virtual_time_s:.0f} fin={res.finished}")
    if res.finished:
        print(f"step {step}: ENV FINISHED"); break
print(f"total {time.perf_counter()-t0:.2f}s")
