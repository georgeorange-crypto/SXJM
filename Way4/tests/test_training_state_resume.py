import torch
from scripts.train_candidate_ppo import load_training_state, save_training_state


def test_training_state_round_trips_optimizer_and_counters(tmp_path):
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.Adam(model.parameters(), lr=.01)
    loss = model(torch.ones(1, 2)).sum(); loss.backward(); opt.step()
    path = tmp_path / "state.pt"
    save_training_state(model, opt, path, rows=[{"seed": 1}], transitions=7, updates=2)
    restored = torch.nn.Linear(2, 1); restored_opt = torch.optim.Adam(restored.parameters(), lr=.01)
    meta = load_training_state(restored, restored_opt, path)
    assert meta["transitions"] == 7 and meta["updates"] == 2
    assert meta["rows"] == [{"seed": 1}]
