"""Decision-trace attribution reports for algorithm ablations."""
from __future__ import annotations
from collections import defaultdict
from typing import Iterable

def summarize_decision_trace(trace: Iterable[dict]) -> dict:
    """Aggregate which ranking signals changed decisions and their outcomes.

    This is diagnostic only. Missing/unfinished realized outcomes are excluded
    from outcome means rather than replaced with synthetic values.
    """
    rows = list(trace)
    summary = {
        "n_decisions": len({r.get("decision_id", r.get("step")) for r in rows}),
        "n_candidates": len(rows),
        "selected": sum(bool(r.get("selected")) for r in rows),
        "rl_changed_actions": sum(bool(r.get("rl_changed_action")) for r in rows if r.get("selected")),
        "soft_information_selected": sum(
            bool(r.get("selected")) and float(r.get("information_gain_est", 0.0)) > 0.0 for r in rows
        ),
        "bundle_selected": sum(
            bool(r.get("selected")) and r.get("candidate_type") == "STOP" for r in rows
        ),
        "by_candidate_type": {},
    }
    groups = defaultdict(list)
    for row in rows:
        if row.get("selected"):
            groups[str(row.get("candidate_type", "UNKNOWN"))].append(row)
    for kind, selected in sorted(groups.items()):
        realized = [r["realized"] for r in selected if isinstance(r.get("realized"), dict)]
        entry = {"selected": len(selected), "realized": len(realized)}
        if realized:
            entry["mean_time_s"] = sum(float(x.get("time_until_next_replan", 0.0)) for x in realized) / len(realized)
            entry["mean_distance_m"] = sum(float(x.get("distance", 0.0)) for x in realized) / len(realized)
            reductions = [float(x["belief_area_reduction"]) for x in realized
                          if x.get("belief_area_reduction") is not None]
            if reductions:
                entry["mean_belief_reduction"] = sum(reductions) / len(reductions)
            entry["mean_certificate_gain"] = sum(float(x.get("certificate_gain", 0.0)) for x in realized) / len(realized)
        summary["by_candidate_type"][kind] = entry
    return summary
