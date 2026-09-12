"""P4 Candidate-PPO training entry point (offline simulator)."""
import argparse, sys
from pathlib import Path
HERE=Path(__file__).resolve(); ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(HERE.parents[1]/'src')]
from offline_sim.case import generate_case
from offline_sim.engine import Engine
from way4.executor import Way3EngineAdapter
from way4.pipeline import Way4Pipeline
from way4.rl.candidate_ppo import CandidateActorCritic, CandidatePPOTrainer, PPOConfig
from way4.rl.candidate_rollout import CandidateTransition, finish_episode, collate
from way4.rl.final_planner import CandidatePPOPlanner, TorchCandidatePolicy, legacy_evaluation_features

def episode(model, seed, max_steps):
    case=generate_case(seed=seed, problem=4, field_kind='smooth', mode='formal'); eng=Engine(case); eng.enter()
    policy=TorchCandidatePolicy(model, legacy_evaluation_features, temperature=1.0, stochastic=True)
    planner=CandidatePPOPlanner(__import__('way4.planner',fromlist=['RecedingHorizonPlanner']).RecedingHorizonPlanner(), policy)
    result=Way4Pipeline(Way3EngineAdapter(eng),n_channels=20,problem=4,max_steps=max_steps,planner=planner).run()
    reward= -result.virtual_time_s/1000. if case.cleared_count==case.total else -100.-(case.total-case.cleared_count)
    ts=[CandidateTransition(r['observation'],r['action'],r['log_prob'],r['value'],0.) for r in policy.records]
    if ts: ts[-1].reward=reward; finish_episode(ts)
    if planner.last_error: print('policy fallback:', planner.last_error, flush=True)
    return ts, result, case

def main():
    p=argparse.ArgumentParser(); p.add_argument('--seeds',default='2000,2001'); p.add_argument('--validation-seeds',default='2040,2041'); p.add_argument('--epochs',type=int,default=2); p.add_argument('--out',default='SXJM/Way4/runs/candidate_ppo_p4.pt'); a=p.parse_args()
    seeds=[int(x) for x in a.seeds.split(',')]; model=CandidateActorCritic(210); trainer=CandidatePPOTrainer(model,PPOConfig(epochs=a.epochs))
    val_seeds={int(x) for x in a.validation_seeds.split(',')}
    if set(seeds) & val_seeds: raise ValueError('training and validation seeds overlap')
    all_t=[]; rows=[]
    for s in seeds:
        ts,r,c=episode(model,s,3000); all_t.extend(ts); rows.append((s,c.cleared_count,c.total,r.virtual_time_s)); print(rows[-1],flush=True)
    if all_t:
        obs,act,old,ret,adv,mask=collate(all_t); trainer.update(obs,act,old,ret,adv,mask)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    import torch; torch.save({'state_dict':model.state_dict(),'input_dim':210,'problem':4},a.out)
    print({'checkpoint':a.out,'episodes':len(rows),'transitions':len(all_t),'full_clear':sum(c==t for _,c,t,_ in rows)})
if __name__=='__main__': main()
