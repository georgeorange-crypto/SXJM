"""Candidate-PPO model scaling benchmark (forward cost only)."""
import argparse,time,sys
from pathlib import Path
HERE=Path(__file__).resolve(); ROOT=HERE.parents[2]; sys.path[:0]=[str(HERE.parents[1]/'src'),str(ROOT)]
from way4.rl.candidate_ppo import CandidateActorCritic
def main():
 p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=10); a=p.parse_args()
 import torch
 dev='cuda' if torch.cuda.is_available() else 'cpu'; rows=[]
 for name,h,l in [('small',64,1),('medium',128,2),('large',256,3)]:
  m=CandidateActorCritic(210,hidden=h,layers=l).to(dev); x=torch.zeros(1,32,210,device=dev)
  for _ in range(2): m(x)
  if dev=='cuda': torch.cuda.synchronize()
  t=time.perf_counter()
  for _ in range(a.steps): m(x)
  if dev=='cuda': torch.cuda.synchronize()
  rows.append({'name':name,'hidden':h,'layers':l,'parameters':sum(p.numel() for p in m.parameters()),'ms_per_batch':(time.perf_counter()-t)*1000/a.steps,'device':dev})
 print(rows)
if __name__=='__main__': main()
