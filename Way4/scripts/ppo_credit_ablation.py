"""Reproducible F01/F02 credit-horizon contract check (no environment run)."""
import argparse, json
from pathlib import Path
from way4.rl.candidate_ppo import compute_gae


def run():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='SXJM/Way4/results/ppo_credit_ablation.json')
    a = p.parse_args()
    rewards = [-.1] * 6 + [-100.0]
    values = [0.0] * len(rewards)
    rows = []
    for gamma in (1.0, .995, .99):
        for lam in (.95, .97, .99):
            adv, ret = compute_gae(rewards, values,
                type('Config', (), {'gamma': gamma, 'gae_lambda': lam})())
            rows.append({'gamma': gamma, 'gae_lambda': lam,
                         'first_advantage': adv[0], 'first_return': ret[0],
                         'terminal_advantage': adv[-1]})
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'rewards': rewards, 'rows': rows}, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(out), 'rows': len(rows)}))


if __name__ == '__main__':
    run()
