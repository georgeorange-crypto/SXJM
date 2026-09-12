"""Emit the frozen Way4 ablation matrix as machine-readable JSON."""
import argparse,json
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
 rows=[{'name':'math','planner_mode':'route_math','features':'all'},
 {'name':'candidate_ppo','planner_mode':'final_ppo','features':'all'},
 {'name':'no_route_features','planner_mode':'final_ppo','features':'no_route'},
 {'name':'no_multi_service','planner_mode':'final_ppo','features':'no_multi_service'},
 {'name':'no_transformer','planner_mode':'final_ppo','features':'mlp'},
 {'name':'no_math_features','planner_mode':'final_ppo','features':'no_math'},
 {'name':'reinforce','planner_mode':'legacy','features':'residual_reinforce'},
 {'name':'small','planner_mode':'final_ppo','hidden':64,'layers':1},
 {'name':'medium','planner_mode':'final_ppo','hidden':128,'layers':2},
 {'name':'large','planner_mode':'final_ppo','hidden':256,'layers':3}]
 json.dump({'problem':4,'same_seeds_required':True,'rows':rows},open(a.out,'w'),indent=2);print(a.out)
if __name__=='__main__':main()
