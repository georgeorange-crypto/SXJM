"""Run the P4 deterministic gate on the bundled simulator.

This is the mandatory pre-training gate: it records ground-truth full-clear,
time and failure reasons on a frozen seed list. It never tunes or writes a model.
"""
import argparse, json, sys, statistics
from pathlib import Path
HERE = Path(__file__).resolve(); ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE.parents[1] / 'src')]
from offline_sim.case import generate_case
from offline_sim.engine import Engine
from way4.executor import Way3EngineAdapter
from way4.pipeline import Way4Pipeline
from way4.planner.lower_bounds import mission_lower_bound
from way4.evaluation_protocol import DEFAULT_SPLITS

def run(seed, max_steps, batch_stop=True, hard=False, field_kind='smooth'):
    case = generate_case(seed=seed, problem=4, field_kind=field_kind, mode='formal', hard=hard)
    eng = Engine(case); eng.enter()
    result = Way4Pipeline(Way3EngineAdapter(eng), n_channels=20, problem=4,
                          max_steps=max_steps, planner_mode='route_math',
                          batch_stop=batch_stop).run()
    # Post-hoc truth-backed lower bound for reporting only.  It is not exposed
    # to the planner and uses max(travel alternatives) to avoid double counting.
    source_points = [(float(j.x), float(j.y)) for j in case.jammers]
    lower_bound_s = mission_lower_bound(
        start=(0.0, 0.0), coverage_points=source_points,
        source_points=source_points, clear_centers=source_points,
        speed=5.0, n_measure_min=case.total, n_sources=case.total,
        n_switch_min=0)
    row = {'seed': seed, 'source_count': case.total,
            'full_clear': case.cleared_count == case.total,
            'truth_cleared': case.cleared_count, 'truth_total': case.total,
            'time_s': result.virtual_time_s, 'error': result.error,
            'steps': result.steps,
            'move_distance_m': result.move_distance_m,
            'n_measure': result.n_measure, 'n_switch': result.n_switch,
            'n_clear': result.n_clear, 'n_clear_hit': result.n_clear_hit,
            'time_move_s': result.time_move_s,
            'time_measure_s': result.time_measure_s,
            'time_switch_s': result.time_switch_s,
            'time_clear_s': result.time_clear_s,
            'route_n_stops': result.route_n_stops,
            'revisit_distance': result.revisit_distance,
            'certificate_only_travel': result.certificate_only_travel,
            'seconds_per_target': (result.virtual_time_s / case.total
                                   if case.total else None),
            'T_lower_bound_s': lower_bound_s,
            'rho_T_over_LB': (result.virtual_time_s / lower_bound_s
                              if lower_bound_s > 0 else None),
            'certified_gap': ((result.virtual_time_s - lower_bound_s) / lower_bound_s
                              if lower_bound_s > 0 else None)}
    return row

def main():
    p=argparse.ArgumentParser(); p.add_argument('--seeds', default='2000-2009')
    p.add_argument('--split', choices=('train', 'validation', 'test', 'stress', 'adversarial'),
                   help='use the frozen evaluation split instead of --seeds')
    p.add_argument('--max-steps', type=int, default=3000); p.add_argument('--out')
    p.add_argument('--no-batch-stop', action='store_true',
                   help='disable safe skipping of already-resolved channel scans')
    p.add_argument('--hard', action='store_true',
                   help='最难 P4 基准：全部干扰源为定向源')
    p.add_argument('--target-s-per-target', type=float, default=400.0,
                   help='community performance gate; reported for every run')
    p.add_argument('--resume', action='store_true',
                   help='resume rows already present in --out (same seed list only)')
    p.add_argument('--field-kind', choices=('smooth', 'iid', 'biased', 'adversarial', 'piecewise'),
                   default=None, help='error-field family; adversarial split defaults to adversarial')
    p.add_argument('--require-target', action='store_true',
                   help='return failure unless every full-clear row meets the target')
    a=p.parse_args()
    if a.split:
        seeds=list(getattr(DEFAULT_SPLITS, a.split))
    else:
        lo,hi=(map(int,a.seeds.split('-')) if '-' in a.seeds else (None,None))
        seeds=list(range(lo,hi+1)) if lo is not None else [int(x) for x in a.seeds.split(',')]
    field_kind = a.field_kind or ('adversarial' if a.split == 'adversarial' else 'smooth')
    rows=[]
    if a.resume and a.out and Path(a.out).is_file():
        old=json.loads(Path(a.out).read_text(encoding='utf-8'))
        if (old.get('seeds') == seeds and old.get('hard') == a.hard
                and old.get('field_kind') == field_kind):
            rows=list(old.get('rows', []))
    done={int(r['seed']) for r in rows}
    for seed in seeds:
        if seed in done:
            continue
        row=run(seed,a.max_steps, batch_stop=not a.no_batch_stop, hard=a.hard,
                field_kind=field_kind)
        rows.append(row)
        # Write a truthful partial artifact after every completed episode so a
        # long adversarial sweep is observable and resumable.
        if a.out:
            partial={'problem':4,'hard':a.hard,'split':a.split,'field_kind':field_kind,'seeds':seeds,
                     'batch_stop':not a.no_batch_stop,'rows':rows,
                     'completed_seeds':[int(r['seed']) for r in rows],
                     'status':'partial'}
            Path(a.out).parent.mkdir(parents=True,exist_ok=True)
            Path(a.out).write_text(json.dumps(partial,indent=2),encoding='utf-8')
    full_clear_rate = sum(r['full_clear'] for r in rows)/len(rows)
    target_rows = [r for r in rows if r['full_clear'] and
                   r['seconds_per_target'] is not None]
    target_pass = bool(target_rows) and all(
        r['seconds_per_target'] <= a.target_s_per_target for r in target_rows)
    success_times = sorted(float(r['time_s']) for r in rows if r['full_clear']
                           and r.get('time_s') is not None)
    def percentile(p):
        if not success_times:
            return None
        idx = p * (len(success_times) - 1)
        lo, hi = int(idx), int(idx) + (1 if idx % 1 else 0)
        return (success_times[lo] if lo == hi else
                success_times[lo] + (success_times[hi] - success_times[lo]) * (idx - lo))
    payload={'problem':4,'hard':a.hard,'split':a.split,'field_kind':field_kind,'seeds':seeds,'batch_stop':not a.no_batch_stop,'rows':rows,
             'completed_seeds':[int(r['seed']) for r in rows], 'status':'complete',
             'full_clear_rate':full_clear_rate,
             'full_clear_time_s': {
                 'n': len(success_times),
                 'mean': statistics.fmean(success_times) if success_times else None,
                 'median': statistics.median(success_times) if success_times else None,
                 'p90': percentile(0.90), 'p95': percentile(0.95),
                 'max': max(success_times) if success_times else None,
             },
             'target_s_per_target':a.target_s_per_target,
             'target_pass_full_clear_rows':target_pass,
             'target_required':bool(a.require_target)}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if a.out: Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(payload,indent=2),encoding='utf-8')
    ok = payload['full_clear_rate'] == 1.0
    if a.require_target:
        ok = ok and target_pass
    return 0 if ok else 2
if __name__=='__main__': raise SystemExit(main())
