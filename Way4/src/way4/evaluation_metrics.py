"""Frozen paired evaluation metrics."""
import statistics
from .planner.lower_bounds import certified_gap

def episode_row(result, *, n_sources=None, clear_route_reference=None,
                full_clear=None, lower_bound_s=None):
    """Canonical benchmark row shared by trace and paired evaluators.

    ``clear_route_reference`` is optional because it is an external baseline;
    absent references remain ``None`` rather than being fabricated.
    """
    n = int(n_sources if n_sources is not None else getattr(result, "n_channels", 0))
    total = float(getattr(result, "virtual_time_s", 0.0))
    length = float(getattr(result, "total_distance_m", getattr(result, "move_distance_m", 0.0)))
    ref = None if clear_route_reference is None else float(clear_route_reference)
    lb = None if lower_bound_s is None else max(0.0, float(lower_bound_s))
    row = {
        "full_clear": bool(getattr(result, "full_clear", False) if full_clear is None else full_clear),
        "T_total_s": total, "T_per_source_s": total / n if n else None,
        "T_lower_bound_s": lb,
        "rho_T_over_LB": (total / lb) if lb and lb > 0.0 else None,
        "certified_gap": (certified_gap(total, lb) if lb is not None else None),
        "L_total_m": length, "L_over_L_ref": length / ref if ref and ref > 0 else None,
        "N_measure": int(getattr(result, "n_measure", 0)),
        "N_empty_scan": int(getattr(result, "n_empty_scan", 0)),
        "N_longjump": int(getattr(result, "n_longjump", 0)),
        "N_crossing": int(getattr(result, "n_crossing", 0)),
        "T_move_s": float(getattr(result, "time_move_s", 0.0)),
        "T_measure_s": float(getattr(result, "time_measure_s", 0.0)),
        "T_switch_s": float(getattr(result, "time_switch_s", 0.0)),
        "T_clear_s": float(getattr(result, "time_clear_s", 0.0)),
        "T_other_s": float(getattr(result, "time_other_s", 0.0)),
        "T_accounting_error_s": float(getattr(result, "time_accounting_error_s", 0.0)),
        "L_equivalent_m": float(getattr(
            result, "equivalent_route_cost_m",
            length + 30.0 * int(getattr(result, "n_measure", 0)),
        )),
        "planner_version": getattr(result, "planner_version", "unknown"),
        "metric_schema_version": getattr(result, "metric_schema_version", "way4-metrics-v1"),
        "L_clear_m": float(getattr(result, "clear_distance_m", 0.0)),
        "longest_waiting_sources": list(getattr(result, "longest_waiting_sources", [])),
        "efficiency_metrics": dict(getattr(result, "efficiency_metrics", {})),
        "no_progress_time_s": float(getattr(result, "no_progress_time_s", 0.0)),
        "backtrack_m": float(getattr(result, "backtrack_m", 0.0)),
        "repeated_edge_m": float(getattr(result, "repeated_edge_m", 0.0)),
        "unnecessary_return_m": float(getattr(result, "unnecessary_return_m", 0.0)),
    }
    return row
def summarize(rows):
    times=[float(r['time_s']) for r in rows if r.get('full_clear')]
    return {'n':len(rows), 'full_clear':sum(bool(r.get('full_clear')) for r in rows),
            'full_clear_rate':sum(bool(r.get('full_clear')) for r in rows)/len(rows) if rows else 0.,
            'mean':statistics.mean(times) if times else None,
            'median':statistics.median(times) if times else None,
            'p90':_pct(times,.90), 'p95':_pct(times,.95),
            'max':max(times) if times else None}
def _pct(xs, p):
    if not xs:return None
    ys=sorted(xs); return ys[min(len(ys)-1, int((len(ys)-1)*p))]
def paired(base, other):
    b={r['seed']:r for r in base}; o={r['seed']:r for r in other}
    pairs=[(b[s],o[s]) for s in sorted(set(b)&set(o))]
    deltas=[x['time_s']-y['time_s'] for x,y in pairs]
    return {'n':len(pairs),'win_rate':sum(d>0 for d in deltas)/len(deltas) if deltas else 0.,
            'regressions_over_100s':sum(d < -100 for d in deltas),
            'regressions_over_300s':sum(d < -300 for d in deltas),
            'worst_regression':min(deltas) if deltas else None}
