import json
from scripts.ppo_credit_ablation import run


def test_credit_ablation_emits_all_gamma_lambda_pairs(tmp_path, monkeypatch):
    out = tmp_path / 'credit.json'
    monkeypatch.setattr('sys.argv', ['credit', '--out', str(out)])
    run()
    payload = json.loads(out.read_text(encoding='utf-8'))
    assert len(payload['rows']) == 9
    assert {r['gamma'] for r in payload['rows']} == {1.0, .995, .99}
    assert {r['gae_lambda'] for r in payload['rows']} == {.95, .97, .99}
