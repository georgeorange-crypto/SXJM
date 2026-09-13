"""Deterministic priority weights for PPO hard-case sampling (AC04/AC05)."""


HARD_CASE_TAGS = frozenset({"edge_directional", "outward_directional", "boundary",
                            "evasive", "far_pair", "multi_cluster", "certificate_hole",
                            "high_measurement", "long_route"})


def hard_case_weight(case, *, floor: float = 1.0) -> float:
    """Weight a case by explicit difficulty and its observed T/T_LB ratio."""
    if floor <= 0:
        raise ValueError("floor must be positive")
    tags = set(case.get("tags", ()))
    ratio = float(case.get("time_s", 0.0)) / max(float(case.get("lower_bound_s", 0.0)), 1e-9)
    if ratio < 0.0:
        raise ValueError("time/lower-bound ratio must be non-negative")
    return max(float(floor), 1.0 + max(0.0, ratio - 1.0) + 0.25 * len(tags & HARD_CASE_TAGS))


def prioritized_case_weights(cases):
    """Return normalized sampling probabilities, preserving input order."""
    weights = [hard_case_weight(case) for case in cases]
    total = sum(weights)
    return [weight / total for weight in weights] if total else []


def choose_seed(seeds, case_stats, rng):
    """Sample one seed using its observed hard-case weight.

    ``case_stats`` maps seed to the same fields accepted by
    :func:`hard_case_weight`; missing seeds retain the neutral floor.
    """
    values = tuple(seeds)
    if not values:
        raise ValueError("at least one seed is required")
    weights = [hard_case_weight(case_stats.get(seed, {})) for seed in values]
    return rng.choices(values, weights=weights, k=1)[0]
