from way4.belief import BeliefState
from way4.belief.hypothesis import HypothesisLayer
from way4.certificate import CertificateManager
from way4.core import RobotState
from way4.planner.candidates import CandidateGenerator


def test_soft_hypothesis_reaches_spatial_candidate_interactions():
    belief = BeliefState(n_channels=1)
    hypotheses = HypothesisLayer()
    # Materialise the channel's soft population before candidate construction.
    hypotheses.channel(1)
    generator = CandidateGenerator(
        hypothesis_layer=hypotheses,
        explore_ring_radii=(600.0,),
        explore_ring_angles=4,
        max_explore=4,
    )
    candidates = generator.generate_spatial_candidates(
        belief, CertificateManager(n_channels=1), RobotState()
    )
    assert candidates
    interactions = [c.interactions[1] for c in candidates]
    assert any(x.directional_entropy > 0.0 for x in interactions)
    assert any(x.visibility_gain >= 0.0 for x in interactions)
