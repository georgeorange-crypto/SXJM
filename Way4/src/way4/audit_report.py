"""Machine-readable hard-trace route, measurement and time audit (AG)."""


def build_hard_trace_audit(*, route_metrics, primitive_trace=(), total_time_s=0.0,
                           time_buckets=None, route_regret_waste_s=0.0):
    """Assemble stable AG01--AG03 fields from authoritative trace inputs."""
    trace = list(primitive_trace)
    measures = [x for x in trace if str(x.get("action", "")).lower() == "measure"]
    useful = sum(bool(x.get("useful", x.get("progress", False))) for x in measures)
    audit = {
        "route": {"total_move_m": float(getattr(route_metrics, "total_distance", 0.0)),
                  "new_route_m": float(getattr(route_metrics, "total_distance", 0.0)
                                         - getattr(route_metrics, "repeated_edge_m", 0.0)),
                  "backtrack_m": float(getattr(route_metrics, "backtrack_m", 0.0)),
                  "repeated_edge_m": float(getattr(route_metrics, "repeated_edge_m", 0.0)),
                  "certificate_only_m": float(getattr(route_metrics, "certificate_only_travel", 0.0)),
                  "unnecessary_return_m": float(getattr(route_metrics, "unnecessary_return_m", 0.0)),
                  "avoidable_crossing_m": float(getattr(route_metrics, "avoidable_crossing_m", 0.0))},
        "measurement": {"total_measurements": len(measures),
                        "useful_measurements": int(useful),
                        "no_progress_measurements": len(measures) - int(useful),
                        "repeat_measurements": sum(bool(x.get("repeat", False)) for x in measures),
                        "certificate_measurements": sum(bool(x.get("certificate", False)) for x in measures),
                        "localization_measurements": sum(bool(x.get("localization", False)) for x in measures)},
        "time": {"total_time_s": float(total_time_s),
                 "route_regret_waste_s": float(route_regret_waste_s)},
    }
    for key, value in (time_buckets or {}).items():
        audit["time"][str(key)] = float(value)
    return audit
