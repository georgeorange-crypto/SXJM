"""Independent validation gate; missing safety evidence never counts as zero."""
from math import isfinite
from statistics import mean, median


def validation_gate(rows, expected_seeds):
    expected = list(expected_seeds)
    seeds = [r.get('seed') for r in rows]
    reasons = []
    if not expected or len(set(expected)) != len(expected):
        reasons.append('invalid_validation_split')
    if len(seeds) != len(expected) or set(seeds) != set(expected):
        reasons.append('incomplete_or_duplicate_validation')
    for row in rows:
        if row.get('full_clear') is not True:
            reasons.append('full_clear_failed')
        for key in ('illegal_clear', 'safety_violation'):
            if type(row.get(key)) is not int or row[key] != 0:
                reasons.append(key + '_failed_or_unmeasured')
        if row.get('error') or row.get('greedy') is not True:
            reasons.append('invalid_evaluation')
        value = row.get('time_s')
        if not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
            reasons.append('invalid_time')
    result = {'passed': not reasons, 'reasons': sorted(set(reasons))}
    if not reasons:
        times = sorted(r['time_s'] for r in rows)
        result.update(mean_time_s=mean(times), median_time_s=median(times),
                      p90_time_s=times[int((len(times)-1)*.9)], max_time_s=max(times))
    return result
