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
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--seeds',default='2040,2041'); p.add_argument('--problem',type=int,default=4); p.add_argument('--max-steps',type=int,default=3000); p.add_argument('--out'); p.add_argument('--compact',action='store_true'); a=p.parse_args()
 seeds=[int(x) for x in a.seeds.split(',')]; policy,_=load_candidate_policy(a.checkpoint)
 rows=[]
 def persist():
  if a.out:
   Path(a.out).parent.mkdir(parents=True,exist_ok=True)
   Path(a.out).write_text(json.dumps({'checkpoint':a.checkpoint,'problem':a.problem,'planner_mode':'final_ppo','rows':rows,'full_clear_rate':sum(x['full_clear'] for x in rows)/len(rows)}, indent=2), encoding='utf-8')
 for seed in seeds:
  case=generate_case(seed=seed,problem=a.problem,field_kind='smooth',mode='formal'); e=Engine(case); e.enter()
  planner=CandidatePPOPlanner(RecedingHorizonPlanner(),policy)
  pipe=Way4Pipeline(Way3EngineAdapter(e),n_channels=20,problem=a.problem,max_steps=a.max_steps,planner=planner,
                    planner_mode='final_ppo', adaptive_scan=True, batch_stop=True,
                    opportunistic_clear=True, routing_strategy='tspn', nbv_objective='minimax',
                    coverage_strategy='active_fallback', enable_no_signal=True,
                    enable_cardinality=True)
  r=pipe.run()
  row={'seed':seed,'full_clear':case.cleared_count==case.total,'cleared':case.cleared_count,'total':case.total,'time_s':r.virtual_time_s,'avg_time_per_source_s':r.virtual_time_s/case.total if case.total else None,'steps':r.steps,'move_m':r.move_distance_m,'measure':r.n_measure,'switch':r.n_switch,'clear':r.n_clear,'clear_hit':r.n_clear_hit,'time_move_s':r.time_move_s,'time_measure_s':r.time_measure_s,'time_switch_s':r.time_switch_s,'time_clear_s':r.time_clear_s,'services_per_stop':r.services_per_stop,'revisit_distance':r.revisit_distance,'shared_stop_ratio':r.shared_stop_ratio,'total_distance_m':r.total_distance_m,'clear_distance_m':r.clear_distance_m,'longjump':r.n_longjump,'crossing':r.n_crossing,'source_diagnostics':r.source_diagnostics,'policy_decisions':len(planner.decisions),'ppo_fallback':bool(planner.watchdog.fallback),'ppo_fallback_reason':planner.fallback_reason,'error':r.error}
  if not a.compact:
   row.update({'jammers': case.reveal()['jammers'], 'events': [e.__dict__ | {'event_type': e.event_type.value} for e in pipe.events],
               'field': case.field.config(), 'transitions': [t.to_record() for t in pipe.transitions],
               'macro_trace': pipe.macro_trace, 'primitive_trace': pipe.primitive_trace,
               'ppo_records': getattr(policy, 'records', [])})
  rows.append(row)
  persist()
 payload={'checkpoint':a.checkpoint,'problem':a.problem,'configuration':{'planner_mode':'final_ppo','adaptive_scan':True,'batch_stop':True,'opportunistic_clear':True,'routing_strategy':'tspn','nbv_objective':'minimax','coverage_strategy':'active_fallback','enable_no_signal':True,'enable_cardinality':True},'rows':rows,'full_clear_rate':sum(x['full_clear'] for x in rows)/len(rows)}
 print(json.dumps(payload,ensure_ascii=False,indent=2))
 return 0 if payload['full_clear_rate']==1 else 2
if __name__=='__main__': raise SystemExit(main())
