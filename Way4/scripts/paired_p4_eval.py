"""Same-seed paired Math vs Candidate-PPO evaluation."""
import argparse,json,sys
from pathlib import Path
HERE=Path(__file__).resolve(); ROOT=HERE.parents[2]; sys.path[:0]=[str(ROOT),str(HERE.parents[1]/'src')]
from offline_sim.case import generate_case
from offline_sim.engine import Engine
from way4.executor import Way3EngineAdapter
from way4.pipeline import Way4Pipeline
from way4.planner import RecedingHorizonPlanner
from way4.rl.final_planner import load_candidate_policy,CandidatePPOPlanner
from way4.evaluation_metrics import paired, episode_row
def run(seed, mode, policy=None, max_steps=300):
 c=generate_case(seed=seed,problem=4,field_kind='smooth',mode='formal'); e=Engine(c); e.enter()
 p=CandidatePPOPlanner(RecedingHorizonPlanner(),policy) if mode=='ppo' else RecedingHorizonPlanner()
 r=Way4Pipeline(Way3EngineAdapter(e),n_channels=20,problem=4,max_steps=max_steps,planner=p).run()
 row = episode_row(r, n_sources=c.total, full_clear=(c.cleared_count == c.total))
 # Keep the historical short names for downstream artifacts, while making the
 # versioned accounting schema authoritative for new runs.
 row.update({'seed': seed, 'time_s': r.virtual_time_s,
             'move_m': r.move_distance_m, 'measure': r.n_measure,
             'switch': r.n_switch, 'clear': r.n_clear, 'error': r.error,
             'source_count': c.total})
 return row
def main():
 a=argparse.ArgumentParser();a.add_argument('--checkpoint',required=True);a.add_argument('--seeds',default='2050,2051');a.add_argument('--out',required=True);a.add_argument('--max-steps',type=int,default=300);x=a.parse_args()
 seeds=[int(s) for s in x.seeds.split(',')]
 try:
  policy,_=load_candidate_policy(x.checkpoint)
 except Exception as exc:
  # Persist startup failures too: an incompatible checkpoint must be auditable.
  out=Path(x.out); out.parent.mkdir(parents=True,exist_ok=True)
  payload={'checkpoint':x.checkpoint,'seeds':seeds,'math':[],'ppo':[],
           'paired':[],'complete':False,'startup_error':f'{type(exc).__name__}: {exc}'}
  tmp=out.with_suffix(out.suffix+'.tmp'); tmp.write_text(json.dumps(payload,indent=2),encoding='utf-8'); tmp.replace(out)
  raise
 out=Path(x.out); out.parent.mkdir(parents=True,exist_ok=True)
 math=[]; ppo=[]
 if out.exists():
  try:
   old=json.loads(out.read_text(encoding='utf-8'))
   if old.get('seeds') == seeds:
    math=list(old.get('math', [])); ppo=list(old.get('ppo', []))
  except (OSError, ValueError):
   pass
 def flush(complete=False):
  payload={'math':math,'ppo':ppo,
           'paired':paired(math,ppo) if math and ppo else None,
           'complete':bool(complete),
           'seeds':seeds}
  tmp=out.with_suffix(out.suffix+'.tmp')
  tmp.write_text(json.dumps(payload,indent=2),encoding='utf-8')
  tmp.replace(out)
 for seed in seeds:
  if not any(r.get('seed') == seed for r in math): math.append(run(seed,'math',max_steps=x.max_steps)); flush()
 for seed in seeds:
  if not any(r.get('seed') == seed for r in ppo): ppo.append(run(seed,'ppo',policy,x.max_steps)); flush()
 flush(True)
 print(json.dumps(json.loads(out.read_text(encoding='utf-8')),indent=2))
if __name__=='__main__':main()
