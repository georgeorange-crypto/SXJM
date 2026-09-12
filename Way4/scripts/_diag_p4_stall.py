"""Trace WHERE seed 2000 stalls on P4 after the soundness fix: what candidates
are generated in the final stalled ticks, and what the certificate state is."""
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
pipe = Way4Pipeline(Way3EngineAdapter(engine), n_channels=20, problem=4, max_steps=6000)
res = pipe.run()

truth = {j.channel: (j.kind, j.cleared) for j in case.jammers}
print(f"seed={SEED} cleared={case.cleared_count}/{case.total} error={res.error} steps={res.steps}")
print(f"  n anchors (backbone)={len(pipe.certificate.anchors)}")
print("  --- per-source-channel final state ---")
for c in sorted(truth):
    kind, cleared = truth[c]
    b = pipe.belief[c]
    cert = pipe.certificate.certs[c]
    print(f"  ch{c:2d} {kind:3s} truth_cleared={cleared}  belief={b.status.value:16s} "
          f"present={cert.present} anchors_visited={len(cert.visited_anchor_idx)}/{len(pipe.certificate.anchors)} "
          f"cov_ratio={cert.heuristic_coverage_ratio:.3f}")
# what does the generator propose from the final state?
from way4.channels import SchedulerMode
cands = pipe.generator.generate(pipe.belief, pipe.certificate, pipe.state, scan_mode=SchedulerMode.VERIFICATION)
print(f"  --- generator proposes {len(cands)} candidates from stalled state ---")
for m in cands[:12]:
    print(f"    {m.action_type.value:8s} tgt=({m.target[0]:.0f},{m.target[1]:.0f}) "
          f"scan={m.scan_channels} cert_gain={m.certificate_gain:.3f} t={m.expected_time:.1f} meta={m.meta}")
