"""Independent P4 Candidate-PPO checkpoint validation."""
import argparse, json, sys
from pathlib import Path
HERE=Path(__file__).resolve(); ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(HERE.parents[1]/'src')]
from offline_sim.case import generate_case
from offline_sim.engine import Engine
from way4.executor import Way3EngineAdapter
from way4.pipeline import Way4Pipeline
from way4.planner import RecedingHorizonPlanner
from way4.rl.final_planner import load_candidate_policy, CandidatePPOPlanner

def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--seeds',default='2040,2041'); p.add_argument('--max-steps',type=int,default=3000); p.add_argument('--out'); a=p.parse_args()
 seeds=[int(x) for x in a.seeds.split(',')]; policy,_=load_candidate_policy(a.checkpoint)
 rows=[]
 for seed in seeds:
  case=generate_case(seed=seed,problem=4,field_kind='smooth',mode='formal'); e=Engine(case); e.enter()
  planner=CandidatePPOPlanner(RecedingHorizonPlanner(),policy)
  r=Way4Pipeline(Way3EngineAdapter(e),n_channels=20,problem=4,max_steps=a.max_steps,planner=planner).run()
  rows.append({'seed':seed,'full_clear':case.cleared_count==case.total,'cleared':case.cleared_count,'total':case.total,'time_s':r.virtual_time_s,'policy_decisions':len(planner.decisions),'error':r.error})
 payload={'checkpoint':a.checkpoint,'problem':4,'rows':rows,'full_clear_rate':sum(x['full_clear'] for x in rows)/len(rows)}
 print(json.dumps(payload,ensure_ascii=False,indent=2))
 if a.out: Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(payload,indent=2),encoding='utf-8')
 return 0 if payload['full_clear_rate']==1 else 2
if __name__=='__main__': raise SystemExit(main())
