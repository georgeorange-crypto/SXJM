"""Small information-theoretic scoring helpers for selective sensing."""
from __future__ import annotations
from math import log
from typing import Iterable

def entropy(probabilities: Iterable[float]) -> float:
    return -sum(p * log(p) for p in probabilities if p > 0.0)

def information_gain(prior: Iterable[float], posteriors: Iterable[tuple[float, Iterable[float]]]) -> float:
    """Prior entropy minus probability-weighted posterior entropy."""
    return max(0.0, entropy(prior) - sum(float(w) * entropy(p) for w, p in posteriors))

def value_of_information(prior, posteriors, action_cost: float) -> float:
    if action_cost <= 0:
        return 0.0
    return information_gain(prior, posteriors) / float(action_cost)
