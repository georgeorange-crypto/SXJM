"""throwaway: per-seed wall time for both stacks, seeds 1000-1004."""
import sys, time
from pathlib import Path
_H = Path(__file__).resolve()
for _p in (_H.parents[1] / "src", _H.parents[2], _H.parents[2] / "Way3"):
    sys.path.insert(0, str(_p))

from jammerhunt import environment as env
from jammerhunt.agent import Hunter
from jammerhunt.interface import LocalWorld
from way4.executor import Way3EngineAdapter
from way4.pipeline import Way4Pipeline

for seed in range(1000, 1005):
    case3 = env.generate_case(seed=seed, problem=3, field_kind="smooth")
    e3 = env.Engine(case3)
    t0 = time.perf_counter()
    try:
        Hunter(problem=3).run(LocalWorld(e3)); err3 = None
    except Exception as e:
        err3 = f"{type(e).__name__}: {e}"
    w3 = time.perf_counter() - t0
    print(f"seed={seed} tot={case3.total} WAY3: {w3:6.2f}s clr={case3.cleared_count}/{case3.total} "
          f"vt={e3.virtual_time_s:.0f} reason={e3.finish_reason} err={err3}", flush=True)

    case4 = env.generate_case(seed=seed, problem=3, field_kind="smooth")
    e4 = env.Engine(case4); e4.enter()
    pipe = Way4Pipeline(Way3EngineAdapter(e4), n_channels=20, max_steps=3000)
    t0 = time.perf_counter()
    r = pipe.run()
    w4 = time.perf_counter() - t0
    print(f"seed={seed} tot={case4.total} WAY4: {w4:6.2f}s clr={case4.cleared_count}/{case4.total} "
          f"vt={r.virtual_time_s:.0f} steps={r.steps} err={r.error}", flush=True)
