"""Frozen paired evaluation metrics."""
import statistics
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
