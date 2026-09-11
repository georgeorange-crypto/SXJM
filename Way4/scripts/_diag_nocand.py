"""throwaway: diagnose the no_candidate termination. Drives the REAL pipeline loop
and, at the abort point, dumps per-channel certificate state for every UNKNOWN
un-certified channel (heuristic ratio / holes / visited backbone anchors / hard
verdict). Delete before commit."""
import sys
from pathlib import Path
_H = Path(__file__).resolve()
for _p in (_H.parents[1] / "src", _H.parents[2], _H.parents[2] / "Way3"):
    sys.path.insert(0, str(_p))

from jammerhunt import environment as env
from way4.belief import ChannelStatus
from way4.executor import Way3EngineAdapter
from way4.pipeline import Way4Pipeline

seeds = [int(s) for s in sys.argv[1:]] or [1013]

for SEED in seeds:
    case = env.generate_case(seed=SEED, problem=3, field_kind="smooth")
    present = {j.channel for j in case.jammers}
    engine = env.Engine(case); engine.enter()
    pipe = Way4Pipeline(Way3EngineAdapter(engine), n_channels=20, max_steps=3000)

    outcome = "hit max_steps"
    step = 0
    while step < pipe.max_steps:
        pipe.certificate.apply_certifications(pipe.belief, force=True)
        if pipe.belief.all_resolved():
            outcome = f"ALL RESOLVED cleanly @step {step}"; break
        macro = pipe._choose_macro()
        if macro is None:
            outcome = f">>> NO_CANDIDATE @step {step} <<<"; break
        res = pipe.executor.execute(macro, pipe.state, float("inf"))
        pipe.state = res.state
        step += 1
        if res.finished or res.stopped_early:
            outcome = f"env finished/stopped @step {step}"; break

    cm = pipe.certificate
    n_anchors = len(cm.anchors)
    print(f"\n=== seed={SEED} :: {outcome} ===")
    print(f"present={sorted(present)} robot=({pipe.state.x:.0f},{pipe.state.y:.0f}) "
          f"n_anchors={n_anchors} present_count={cm.present_count()} cleared={cm.cleared_count()}")
    for c in range(1, 21):
        b = pipe.belief[c]
        if b.status != ChannelStatus.UNKNOWN:
            continue
        cert = cm.certs[c]
        holes = len(cm.remaining_holes(c))
        absf = cm.is_absent_certified(c, force=True)
        # backbone: how many anchors within reach tol have we actually scanned NO_SIGNAL?
        print(f"  ch{c:2d} [{'PRES' if c in present else 'abs '}] "
              f"ratio={cert.heuristic_coverage_ratio:.4f} holes={holes:4d} "
              f"nscan={len(cert.negative_scan_points):3d} "
              f"anchors={len(cert.visited_anchor_idx)}/{n_anchors} "
              f"hard={cert.hard_complete!s:5s} abs_cert(force)={absf!s:5s}")
