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
from way4.evaluation_metrics import paired
def run(seed, mode, policy=None):
 c=generate_case(seed=seed,problem=4,field_kind='smooth',mode='formal'); e=Engine(c); e.enter()
 p=CandidatePPOPlanner(RecedingHorizonPlanner(),policy) if mode=='ppo' else RecedingHorizonPlanner()
 r=Way4Pipeline(Way3EngineAdapter(e),n_channels=20,problem=4,max_steps=3000,planner=p).run()
 return {'seed':seed,'full_clear':c.cleared_count==c.total,'time_s':r.virtual_time_s,'move_m':r.move_distance_m,'measure':r.n_measure,'switch':r.n_switch,'clear':r.n_clear,'error':r.error}
def main():
 a=argparse.ArgumentParser();a.add_argument('--checkpoint',required=True);a.add_argument('--seeds',default='2050,2051');a.add_argument('--out',required=True);x=a.parse_args()
 seeds=[int(s) for s in x.seeds.split(',')]; policy,_=load_candidate_policy(x.checkpoint)
 math=[run(s,'math') for s in seeds]; ppo=[run(s,'ppo',policy) for s in seeds]
 payload={'math':math,'ppo':ppo,'paired':paired(math,ppo)};Path(x.out).parent.mkdir(parents=True,exist_ok=True);Path(x.out).write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
