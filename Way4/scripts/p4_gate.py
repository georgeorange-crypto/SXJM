"""Run the P4 deterministic gate on the bundled simulator.

This is the mandatory pre-training gate: it records ground-truth full-clear,
time and failure reasons on a frozen seed list. It never tunes or writes a model.
"""
import argparse, json, sys
from pathlib import Path
HERE = Path(__file__).resolve(); ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE.parents[1] / 'src')]
from offline_sim.case import generate_case
from offline_sim.engine import Engine
from way4.executor import Way3EngineAdapter
from way4.pipeline import Way4Pipeline

def run(seed, max_steps):
    case = generate_case(seed=seed, problem=4, field_kind='smooth', mode='formal')
    eng = Engine(case); eng.enter()
    result = Way4Pipeline(Way3EngineAdapter(eng), n_channels=20, problem=4,
                          max_steps=max_steps, planner_mode='route_math').run()
    return {'seed': seed, 'full_clear': case.cleared_count == case.total,
            'truth_cleared': case.cleared_count, 'truth_total': case.total,
            'time_s': result.virtual_time_s, 'error': result.error,
            'steps': result.steps}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--seeds', default='2000-2009')
    p.add_argument('--max-steps', type=int, default=3000); p.add_argument('--out')
    a=p.parse_args(); lo,hi=(map(int,a.seeds.split('-')) if '-' in a.seeds else (None,None))
    seeds=list(range(lo,hi+1)) if lo is not None else [int(x) for x in a.seeds.split(',')]
    rows=[run(s,a.max_steps) for s in seeds]
    payload={'problem':4,'seeds':seeds,'rows':rows,'full_clear_rate':sum(r['full_clear'] for r in rows)/len(rows)}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if a.out: Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(payload,indent=2),encoding='utf-8')
    return 0 if payload['full_clear_rate']==1.0 else 2
if __name__=='__main__': raise SystemExit(main())
