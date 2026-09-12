from way4 import ALGORITHM_NAME, LEARNER_ROLE


def test_final_algorithm_identity_is_frozen():
    assert ALGORITHM_NAME == "Certificate-Guided Active Belief Planning"
    assert LEARNER_ROLE == "Learning-Augmented Planner"
