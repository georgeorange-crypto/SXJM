from way4.rl import Decision, expert_dataset


def test_expert_dataset_preserves_all_counterfactual_candidates():
    decision = Decision(
        features=[[1.0, 2.0], [3.0, 4.0]],
        q_math=[10.0, 20.0], chosen=0, temperature=5.0,
    )
    row = expert_dataset([decision])[0]
    assert row["n_candidates"] == 2
    assert row["q_math"] == [10.0, 20.0]
    assert len(row["features"]) == 2
    assert row["chosen"] == 0
