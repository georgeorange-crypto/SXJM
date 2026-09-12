"""Dependency-free soft-belief algorithms.  These never certify absence."""
from __future__ import annotations
from dataclasses import dataclass
from math import exp, hypot, pi, sqrt
from random import Random
from typing import Callable, Iterable, List, Sequence, Tuple

Point = Tuple[float, float]

def bayesian_update(prior: Sequence[float], likelihood: Sequence[float]) -> List[float]:
    if len(prior) != len(likelihood): raise ValueError("length mismatch")
    weights = [max(0.0, float(a)) * max(0.0, float(b)) for a, b in zip(prior, likelihood)]
    z = sum(weights)
    return [w / z for w in weights] if z > 0 else list(prior)

def gaussian_mixture(points: Sequence[Point], weights=None, bandwidth: float = 1.0,
                    samples: Sequence[Point] = ()) -> List[float]:
    """Evaluate an isotropic Gaussian mixture at query points."""
    if bandwidth <= 0: raise ValueError("bandwidth must be positive")
    ws = list(weights) if weights is not None else [1.0] * len(points)
    z = sum(ws) or 1.0; scale = 1.0 / (2.0 * pi * bandwidth * bandwidth)
    out = []
    for q in samples:
        out.append(sum((w / z) * scale * exp(-((q[0]-p[0])**2 + (q[1]-p[1])**2) / (2*bandwidth**2))
                       for p, w in zip(points, ws)))
    return out

def kde(points: Sequence[Point], queries: Sequence[Point], bandwidth: float = 1.0) -> List[float]:
    return gaussian_mixture(points, bandwidth=bandwidth, samples=queries)

@dataclass
class Particle:
    point: Point
    weight: float = 1.0

class ParticleFilter:
    def __init__(self, particles: Iterable[Particle], seed: int = 0):
        self.particles = list(particles); self.rng = Random(seed)
    def update(self, likelihood: Callable[[Point], float]) -> None:
        ws = [max(0.0, float(likelihood(p.point))) * p.weight for p in self.particles]
        z = sum(ws)
        if z > 0:
            for p, w in zip(self.particles, ws): p.weight = w / z
    def resample(self, n: int | None = None) -> None:
        n = n or len(self.particles)
        if not self.particles: return
        population = [p for p in self.particles]
        weights = [max(0.0, p.weight) for p in population]
        self.particles = [Particle(self.rng.choices(population, weights=weights)[0].point, 1.0/n) for _ in range(n)]
    def mean(self) -> Point:
        z = sum(p.weight for p in self.particles) or 1.0
        return (sum(p.point[0]*p.weight for p in self.particles)/z,
                sum(p.point[1]*p.weight for p in self.particles)/z)
