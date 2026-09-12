"""Acceptance gates for algorithm ablation rows."""
from __future__ import annotations
from typing import Iterable

def algorithm_gate(baseline: Iterable[dict], variant: Iterable[dict]) -> dict:
    """Compare same-seed rows without inventing missing outcomes.

    A variant can only be promoted when full-clear does not regress.  Runtime
    speed/distance are reported separately because a short-horizon gain with
    worse resolution is not an acceptance pass.
    """
    b = {int(r["seed"]): r for r in baseline}
    v = {int(r["seed"]): r for r in variant}
    seeds = sorted(set(b) & set(v))
    regressions = [s for s in seeds if bool(b[s].get("success_ground_truth", b[s].get("success")))
                   and not bool(v[s].get("success_ground_truth", v[s].get("success")))]
    bt = [float(b[s]["virtual_time_s"]) for s in seeds]
    vt = [float(v[s]["virtual_time_s"]) for s in seeds]
    bd = [float(b[s].get("distance_m", b[s].get("move_distance_m", 0.0))) for s in seeds]
    vd = [float(v[s].get("distance_m", v[s].get("move_distance_m", 0.0))) for s in seeds]
    br = [float(b[s].get("resolved", 0.0)) for s in seeds]
    vr = [float(v[s].get("resolved", 0.0)) for s in seeds]
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    return {
        "n_paired": len(seeds),
        "full_clear_regressions": regressions,
        "full_clear_nonregression": not regressions,
        "mean_time_baseline_s": mean(bt), "mean_time_variant_s": mean(vt),
        "mean_time_delta_s": None if not seeds else mean(vt) - mean(bt),
        "mean_distance_baseline_m": mean(bd), "mean_distance_variant_m": mean(vd),
        "mean_distance_delta_m": None if not seeds else mean(vd) - mean(bd),
        "mean_resolved_baseline": mean(br), "mean_resolved_variant": mean(vr),
        "mean_resolved_delta": None if not seeds else mean(vr) - mean(br),
        "resolved_nonregression": (not seeds or mean(vr) >= mean(br) - 1e-12),
        "promote_default": bool(seeds) and not regressions and mean(vr) >= mean(br) - 1e-12,
    }
