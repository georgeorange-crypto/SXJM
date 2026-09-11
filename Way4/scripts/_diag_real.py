"""throwaway: time one Way4 episode on a real 10-16 source case, per-step."""
import sys, time
from pathlib import Path
_H = Path(__file__).resolve()
for _p in (_H.parents[1] / "src", _H.parents[2], _H.parents[2] / "Way3"):
    sys.path.insert(0, str(_p))

from jammerhunt import environment as env
from way4.executor import Way3EngineAdapter
from way4.pipeline import Way4Pipeline

case = env.generate_case(seed=1003, problem=3, field_kind="smooth")
print(f"case: total={case.total} channels={sorted(j.channel for j in case.jammers)}")
engine = env.Engine(case); engine.enter()
pipe = Way4Pipeline(Way3EngineAdapter(engine), n_channels=20, max_steps=120)

t0 = time.perf_counter()
for step in range(120):
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
    print(f"step {step:3d}: {macro.action_type.value:9s} tgt=({macro.target[0]:6.0f},{macro.target[1]:6.0f}) "
          f"nchan={len(macro.scan_channels):2d} ncand={len(cands):3d} | cert={tc1-tc0:.3f} gen={tg1-tg0:.3f} "
          f"plan={tp1-tp0:.3f} exec={te1-te0:.3f} | unk={unknown} res={resolved} "
          f"pres={pipe.certificate.present_count()} vt={pipe.state.virtual_time_s:.0f} fin={res.finished}",
          flush=True)
    if res.finished:
        print(f"step {step}: ENV FINISHED reason={engine.finish_reason}"); break
print(f"total {time.perf_counter()-t0:.2f}s  cleared={case.cleared_count}/{case.total}")
