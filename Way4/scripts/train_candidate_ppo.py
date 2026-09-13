"""P4 Candidate-PPO training entry point (offline simulator)."""
import argparse, sys, json, copy, time, random, math
from pathlib import Path
HERE=Path(__file__).resolve(); ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(HERE.parents[1]/'src')]
from offline_sim.case import generate_case
from offline_sim.engine import Engine
from way4.executor import Way3EngineAdapter
from way4.pipeline import Way4Pipeline
from way4.rl.candidate_ppo import CandidateActorCritic, CandidatePPOTrainer, PPOConfig, MODEL_SCHEMA, entropy_coefficient
from way4.rl.candidate_rollout import timed_policy_transitions, collate
from way4.rl.final_planner import CandidatePPOPlanner, TorchCandidatePolicy, legacy_evaluation_features
from way4.rl.checkpoint_gate import validation_gate
from way4.rl.hard_case_mining import choose_seed
from way4.planner.lower_bounds import mission_lower_bound


def _case_difficulty_tags(case):
    """Derive reproducible hard-case tags from the generated case geometry."""
    jammers = list(getattr(case, 'jammers', ()))
    tags = set()
    if any(getattr(j, 'kind', '') == 'dir' for j in jammers):
        tags.add('edge_directional')
    if any(math.hypot(float(j.x), float(j.y)) >= 1700.0 for j in jammers):
        tags.add('boundary')
    for i, left in enumerate(jammers):
        for right in jammers[i + 1:]:
            if math.hypot(float(left.x) - float(right.x), float(left.y) - float(right.y)) <= 100.0:
                tags.add('far_pair')
                break
        if 'far_pair' in tags:
            break
    if len(jammers) >= 15:
        tags.add('high_measurement')
    return sorted(tags)


def save_candidate_checkpoint(model, out_path, *, rows, transitions, updates=0,
                              history=None, best_score=None, complete=False,
                              safety_qualified=False, selected_update=None,
                              validation_seeds=(), test_seeds=()):
    """Atomically persist a current-schema checkpoint, including partial progress."""
    import torch
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'schema': MODEL_SCHEMA, 'state_dict': model.state_dict(),
        'input_dim': 210, 'problem': 4,
        'architecture': 'candidate_gnn_transformer',
        'complete': bool(complete),
        'seeds': [r['seed'] if isinstance(r, dict) else r[0] for r in rows],
        'validation_seeds': list(validation_seeds), 'test_seeds': list(test_seeds),
        'episodes': len(rows), 'transitions': int(transitions),
        'updates': int(updates), 'history': [] if history is None else history,
        'best_score': best_score,
        'safety_qualified': bool(safety_qualified),
        'selected_update': selected_update,
    }
    tmp = out.with_suffix(out.suffix + '.tmp')
    torch.save(payload, tmp)
    tmp.replace(out)


def save_training_state(model, optimizer, out_path, *, rows=(), transitions=0, updates=0):
    """Persist resumable model/optimizer and RNG state without partial writes."""
    import random
    import torch
    import numpy as np
    payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
               'rows': list(rows), 'transitions': int(transitions), 'updates': int(updates),
               'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
               'torch_rng': torch.get_rng_state()}
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + '.tmp'); torch.save(payload, tmp); tmp.replace(out)


def load_training_state(model, optimizer, path):
    """Restore a state saved by :func:`save_training_state`."""
    import random
    import torch
    import numpy as np
    payload = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(payload['model']); optimizer.load_state_dict(payload['optimizer'])
    random.setstate(payload['python_rng']); np.random.set_state(payload['numpy_rng'])
    torch.set_rng_state(payload['torch_rng'])
    return {k: payload[k] for k in ('rows', 'transitions', 'updates')}

def episode(model, seed, max_steps, *, gamma=1.0, gae_lambda=.97, stochastic=True):
    case=generate_case(seed=seed, problem=4, field_kind='smooth', mode='formal'); eng=Engine(case); eng.enter()
    policy=TorchCandidatePolicy(model, legacy_evaluation_features, temperature=1.0, stochastic=stochastic)
    planner=CandidatePPOPlanner(__import__('way4.planner',fromlist=['RecedingHorizonPlanner']).RecedingHorizonPlanner(), policy)
    result=Way4Pipeline(Way3EngineAdapter(eng),n_channels=20,problem=4,max_steps=max_steps,
                        planner=planner, planner_mode='final_ppo').run()
    ts = timed_policy_transitions(
        policy.records, result.virtual_time_s,
        full_clear=case.cleared_count == case.total,
        n_unresolved=case.total - case.cleared_count, gamma=gamma, lam=gae_lambda)
    if planner.last_error: print('policy fallback:', planner.last_error, flush=True)
    return ts, result, case


def validate_model(model, seeds, max_steps):
    rows = []
    for seed in seeds:
        _, result, case = episode(model, seed, max_steps, stochastic=False)
        rows.append({'seed': seed, 'full_clear': case.cleared_count == case.total,
                     'time_s': result.virtual_time_s, 'greedy': True,
                     'error': result.error,
                     'illegal_clear': getattr(result, 'illegal_clear', None),
                     'safety_violation': getattr(result, 'safety_violation', None)})
    return rows, validation_gate(rows, seeds)

def main():
    p=argparse.ArgumentParser(description='Auditable Candidate-PPO training')
    p.add_argument('--profile', choices=('formal','smoke'), default='formal')
    p.add_argument('--seeds', default=None, help='training seeds, comma separated')
    p.add_argument('--validation-seeds', default=None)
    p.add_argument('--test-seeds', default='12000,12001,12002,12003,12004')
    p.add_argument('--gamma', type=float, choices=(1.0, .995, .99), default=1.0)
    p.add_argument('--gae-lambda', type=float, choices=(.95, .97, .99), default=.97)
    p.add_argument('--minibatch-size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--updates', type=int, default=None)
    p.add_argument('--episodes-per-update', type=int, default=None)
    p.add_argument('--transitions-per-update', type=int, default=None)
    p.add_argument('--time-budget-hours', type=float, default=3.0)
    p.add_argument('--max-steps', type=int, default=3000)
    p.add_argument('--out', default='SXJM/Way4/runs/candidate_ppo_p4_formal.pt')
    p.add_argument('--resume-state', default=None)
    p.add_argument('--state-out', default=None)
    a=p.parse_args()
    if a.profile == 'smoke':
        train_default, val_default, epochs, updates, epu = '2000,2001', '2040,2041', 2, 1, 2
    else:
        # Medium-size bounded run intended to fit in roughly three hours on dex.
        train_default = ','.join(str(x) for x in range(10000,10200))
        val_default = ','.join(str(x) for x in range(11000,11050))
        epochs, updates, epu = 4, 24, 4
    seeds=[int(x) for x in (a.seeds or train_default).split(',') if x.strip()]
    validation=[int(x) for x in (a.validation_seeds or val_default).split(',') if x.strip()]
    test=[int(x) for x in a.test_seeds.split(',') if x.strip()]
    epochs=a.epochs or epochs; updates=a.updates or updates; epu=a.episodes_per_update or epu
    target_transitions = (a.transitions_per_update if a.transitions_per_update is not None
                          else (1 if a.profile == 'smoke' else 2048))
    if not seeds or not validation:
        raise ValueError('training and validation splits must be nonempty')
    if min(epochs, updates, epu, target_transitions, a.minibatch_size) < 1:
        raise ValueError('training counts must be positive')
    if not test or set(seeds) & set(validation) or set(seeds) & set(test) or set(validation) & set(test):
        raise ValueError('training, validation and test seeds must be non-overlapping and nonempty')
    model=CandidateActorCritic(210)
    trainer=CandidatePPOTrainer(model,PPOConfig(epochs=epochs, gamma=a.gamma,
        gae_lambda=a.gae_lambda, minibatch_size=a.minibatch_size))
    total_transitions=0; rows=[]; history=[]; updates_done=0; started=time.monotonic()
    case_stats = {}
    rng = random.Random(0)
    if a.resume_state:
        restored = load_training_state(model, trainer.optimizer, a.resume_state)
        rows = list(restored["rows"]); total_transitions = int(restored["transitions"])
        updates_done = int(restored["updates"])
    best_score=(-1.0, float('inf')); best_state=None; best_update=0
    def persist(complete=False):
        save_candidate_checkpoint(model, a.out, rows=rows,
                                  transitions=total_transitions, updates=updates_done,
                                  history=history, best_score=list(best_score), complete=complete,
                                  safety_qualified=bool(complete and best_state is not None),
                                  selected_update=best_update if complete and best_state is not None else None,
                                  validation_seeds=validation, test_seeds=test)
    # Cycle through the fixed training split. Each update uses fresh rollouts,
    # then immediately updates the policy; this avoids the old one-update-only
    # smoke behavior while retaining every episode result in the checkpoint.
    for update in range(updates_done, updates):
        if time.monotonic()-started >= a.time_budget_hours*3600:
            print('time budget reached; stopping before next rollout', flush=True); break
        batch=[]; batch_start=len(rows); empty_rollouts=0
        while len(batch) < target_transitions or len(rows)-batch_start < epu:
            if time.monotonic()-started >= a.time_budget_hours*3600:
                break
            s = (seeds[len(rows) % len(seeds)] if not case_stats
                 else choose_seed(seeds, case_stats, rng))
            ts,r,c=episode(model,s,a.max_steps, gamma=a.gamma, gae_lambda=a.gae_lambda)
            batch.extend(ts); total_transitions += len(ts)
            lower_bound = getattr(c, "lower_bound_s", None)
            if lower_bound is None and hasattr(c, "jammers"):
                source_points = [(float(j.x), float(j.y)) for j in c.jammers]
                lower_bound = mission_lower_bound(
                    start=(0.0, 0.0), coverage_points=source_points,
                    source_points=source_points, clear_centers=source_points,
                    speed=5.0, n_measure_min=0, n_sources=len(source_points),
                    n_switch_min=0)
            row = {'seed':s,'cleared':c.cleared_count,'total':c.total,
                         'time_s':r.virtual_time_s,'full_clear':c.cleared_count==c.total,
                         'transitions':len(ts)}
            row['tags'] = _case_difficulty_tags(c)
            if lower_bound is not None:
                row['lower_bound_s'] = float(lower_bound)
            rows.append(row)
            if lower_bound is not None:
                case_stats[s] = {'time_s': float(r.virtual_time_s),
                                 'lower_bound_s': float(lower_bound),
                                 'tags': row['tags']}
            persist(False)
            empty_rollouts = empty_rollouts + 1 if not ts else 0
            if empty_rollouts >= len(seeds):
                raise RuntimeError('one training seed cycle produced no policy transitions')
        if len(batch) < target_transitions:
            print('time budget reached with incomplete transition batch', flush=True)
            break
        if batch:
            obs,act,old,ret,adv,mask=collate(batch)
            if hasattr(trainer, 'cfg'):
                trainer.cfg.entropy_coef = entropy_coefficient(trainer.cfg, update, updates)
            stats=trainer.update(obs,act,old,ret,adv,mask)
            recent=rows[batch_start:]
            clear=[x['time_s'] for x in recent if x['full_clear']]
            score=(sum(x['full_clear'] for x in recent)/len(recent),
                   sum(clear)/len(clear) if clear else float('inf'))
            validation_rows, gate = validate_model(model, validation, a.max_steps)
            if gate['passed'] and (best_state is None or gate['mean_time_s'] < best_score[1]):
                best_score=(1., gate['mean_time_s'])
                best_state=copy.deepcopy(model.state_dict()); best_update=update+1
            current_entropy = trainer.cfg.entropy_coef if hasattr(trainer, 'cfg') else None
            stats.update({'update':update+1,'episodes':len(recent),
                          'batch_transitions':len(batch),
                          'target_transitions':target_transitions,
                          'entropy_coef': current_entropy,
                          'validation_rows':validation_rows, 'validation_gate':gate,
                          'batch_full_clear':score[0], 'batch_clear_mean_time_s':score[1],
                          'best_update':best_update, 'best_score':list(best_score)})
            history.append(stats); updates_done += 1
        persist(False)
        if a.state_out:
            save_training_state(model, trainer.optimizer, a.state_out, rows=rows,
                                transitions=total_transitions, updates=updates_done)
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    persist(updates_done == updates)
    print(json.dumps({'checkpoint':a.out,'profile':a.profile,'train_seeds':seeds,
                      'validation_seeds':validation,'test_seeds':test,'episodes':len(rows),
                      'transitions':sum(x['transitions'] for x in rows),
                      'updates':updates_done,'full_clear':sum(x['full_clear'] for x in rows),
                      'best_update':best_update,'best_score':list(best_score),
                      'time_budget_hours':a.time_budget_hours,
                      'validation_not_run':not bool(history),
                      'checkpoint_safety_qualified':best_state is not None}, ensure_ascii=False))
if __name__=='__main__': main()
