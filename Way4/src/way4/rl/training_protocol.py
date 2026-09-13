"""PPO discount/GAE sweep protocol (F01/F02)."""
from itertools import product


def gamma_lambda_sweep(gammas=(1.0, 0.995, 0.99), lambdas=(0.95, 0.97, 0.99)):
    """Return deterministic, validated gamma/lambda experiment combinations."""
    gs, ls = tuple(float(x) for x in gammas), tuple(float(x) for x in lambdas)
    if not gs or not ls or any(not 0.0 < x <= 1.0 for x in gs + ls):
        raise ValueError("gamma and lambda must lie in (0, 1]")
    return tuple({"gamma": g, "gae_lambda": l} for g, l in product(gs, ls))
