import math
import torch
from way4.rl.candidate_ppo import CandidatePPOTrainer, PPOConfig


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(2, 1)
        self.critic = torch.nn.Linear(2, 1)
        self.seen = []

    def forward(self, x):
        self.seen.extend(x[:, 0, 0].tolist())
        return self.actor(x).squeeze(-1), self.critic(x.mean(1)).squeeze(-1)


def test_shuffled_minibatches_visit_each_transition_each_epoch_and_update():
    torch.manual_seed(19)
    model = TinyPolicy()
    original = model.actor.weight.detach().clone()
    trainer = CandidatePPOTrainer(model, PPOConfig(epochs=2, minibatch_size=2))
    observations = [[[i, 1.], [0., 1.]] for i in range(5)]
    result = trainer.update(observations, [0]*5, [-.69]*5,
                            [1.]*5, [1., 2., 3., 4., 5.], [[True, True]]*5)
    assert result['optimizer_steps'] == 6  # retains the final one-row minibatch
    assert math.isfinite(result['loss'])
    assert sorted(model.seen[:5]) == list(range(5))
    assert sorted(model.seen[5:]) == list(range(5))
    assert model.seen[:5] != list(range(5))
    assert not torch.equal(original, model.actor.weight)
