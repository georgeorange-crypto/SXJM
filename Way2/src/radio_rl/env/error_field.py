"""Bearing-error fields — ported verbatim (physics-identical) from the reference
oracle and wired into the ``ERROR_FIELDS`` registry.

The problem statement fixes only three hard constraints on the direction-finding
error and gives *no* probability distribution:

  (i)   globally bounded: error in [-1 deg, +1 deg];
  (ii)  location-determined: the electromagnetic environment at a point is fixed,
        so repeating a measurement at the same (x, y) yields the *same* error;
  (iii) spatially statistical: the error only varies statistically across
        different locations / environments.

Crucially, the error is therefore a deterministic function of (x, y) — never
per-measurement RNG. That single fact is what makes "wiggle in place and average"
either work (iid field) or fail (smooth field), and it is the property the whole
localization geometry leans on. Several fields all satisfying (i)-(iii) are
provided for robustness / worst-case testing.
"""

from __future__ import annotations

import hashlib
import math
import random
import struct
from typing import Optional

from ..core.registry import ERROR_FIELDS


def _clamp1(v: float) -> float:
    if v > 1.0:
        return 1.0
    if v < -1.0:
        return -1.0
    return v


class BaseErrorField:
    """Base class. Subclasses implement ``error_deg(x, y)`` in [-1, 1], constant
    for a given coordinate."""

    kind = "base"

    def __init__(self, seed: int, **params):
        self.seed = int(seed)
        self.params = dict(params)

    def error_deg(self, x: float, y: float) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def config(self) -> dict:
        return {"kind": self.kind, "seed": self.seed, "params": dict(self.params)}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(seed={self.seed}, params={self.params})"


@ERROR_FIELDS.register("smooth")
class SmoothErrorField(BaseErrorField):
    """Spatially-correlated field: a sum of sinusoids whose amplitudes are
    normalised so Sum|amp| = amplitude <= 1 (hence strictly bounded). Nearby
    points see almost the same error, so "wiggle and average" barely helps.
    This is the realistic default."""

    kind = "smooth"

    def __init__(self, seed: int, length_scale: float = 300.0,
                 n_components: int = 12, amplitude: float = 1.0):
        super().__init__(seed, length_scale=float(length_scale),
                         n_components=int(n_components), amplitude=float(amplitude))
        self.length_scale = float(length_scale)
        self.n_components = int(n_components)
        self.amplitude = float(amplitude)
        rng = random.Random(self.seed ^ 0x9E3779B9)
        amps = [rng.random() for _ in range(self.n_components)]
        s = sum(amps) or 1.0
        amps = [a / s * self.amplitude for a in amps]   # Sum|amp| = amplitude
        self._comps = []
        for a in amps:
            wavelength = self.length_scale * math.exp(rng.gauss(0.0, 0.5))
            wavelength = max(5.0, wavelength)
            k = 2.0 * math.pi / wavelength
            theta = rng.uniform(0.0, 2.0 * math.pi)
            phase = rng.uniform(0.0, 2.0 * math.pi)
            self._comps.append((a, k * math.cos(theta), k * math.sin(theta), phase))

    def error_deg(self, x: float, y: float) -> float:
        e = 0.0
        for a, kx, ky, ph in self._comps:
            e += a * math.sin(kx * x + ky * y + ph)
        return _clamp1(e)


@ERROR_FIELDS.register("iid")
class IIDErrorField(BaseErrorField):
    """Location-independent field: each distinct (x, y) draws independently from
    U[-magnitude, magnitude], constant for that coordinate (hash-determined).
    A "most ordinary random" baseline on which averaging repeated measurements is
    unusually effective — useful to expose strategies that rely on that."""

    kind = "iid"

    def __init__(self, seed: int, magnitude: float = 1.0):
        super().__init__(seed, magnitude=float(magnitude))
        self.magnitude = _clamp1(abs(float(magnitude)))

    def error_deg(self, x: float, y: float) -> float:
        b = struct.pack("<ddq", float(x), float(y), self.seed & 0x7FFFFFFFFFFFFFFF)
        h = hashlib.blake2b(b, digest_size=8).digest()
        u = int.from_bytes(h, "little") / float(1 << 64)   # [0,1)
        return _clamp1((2.0 * u - 1.0) * self.magnitude)


@ERROR_FIELDS.register("biased")
class BiasedErrorField(BaseErrorField):
    """Biased field: eps = clamp(bias + correlated noise), mean ~ bias. Defeats
    zero-mean least-squares / averaging denoisers."""

    kind = "biased"

    def __init__(self, seed: int, bias: float = 0.6, noise_amp: float = 0.4,
                 length_scale: float = 300.0, n_components: int = 12):
        super().__init__(seed, bias=float(bias), noise_amp=float(noise_amp),
                         length_scale=float(length_scale), n_components=int(n_components))
        self.bias = _clamp1(float(bias))
        self._noise = SmoothErrorField(seed, length_scale=length_scale,
                                       n_components=n_components,
                                       amplitude=abs(float(noise_amp)))

    def error_deg(self, x: float, y: float) -> float:
        return _clamp1(self.bias + self._noise.error_deg(x, y))


@ERROR_FIELDS.register("adversarial")
class AdversarialErrorField(BaseErrorField):
    """Worst-case field: error pinned near +/-magnitude.
      - mode="boundary": +mag / -mag by spatial block (constant within a block,
        flipping at block borders);
      - mode="constant": +magnitude everywhere (worst constant bias)."""

    kind = "adversarial"

    def __init__(self, seed: int, magnitude: float = 1.0,
                 length_scale: float = 150.0, mode: str = "boundary"):
        super().__init__(seed, magnitude=float(magnitude),
                         length_scale=float(length_scale), mode=str(mode))
        self.magnitude = _clamp1(abs(float(magnitude)))
        self.mode = str(mode)
        self._sign = SmoothErrorField(seed, length_scale=length_scale,
                                      n_components=6, amplitude=1.0)

    def error_deg(self, x: float, y: float) -> float:
        if self.mode == "constant":
            return self.magnitude
        v = self._sign.error_deg(x, y)
        s = 1.0 if v >= 0.0 else -1.0
        return _clamp1(s * self.magnitude)


@ERROR_FIELDS.register("piecewise")
class PiecewiseErrorField(BaseErrorField):
    """Zoned environment: the arena is split into n_blocks Voronoi cells (random
    centres), each with its own correlated bias b_k in [-block_bias, block_bias]
    plus small correlated noise. Error is discontinuous across cells, correlated
    within them."""

    kind = "piecewise"

    def __init__(self, seed: int, n_blocks: int = 8, block_bias: float = 0.7,
                 noise_amp: float = 0.2, length_scale: float = 120.0,
                 radius: float = 1800.0):
        super().__init__(seed, n_blocks=int(n_blocks), block_bias=float(block_bias),
                         noise_amp=float(noise_amp), length_scale=float(length_scale),
                         radius=float(radius))
        self.n_blocks = int(n_blocks)
        rng = random.Random(self.seed ^ 0x2545F491)
        self._centers = []
        for _ in range(self.n_blocks):
            r = float(radius) * math.sqrt(rng.random())
            t = rng.uniform(0.0, 2.0 * math.pi)
            self._centers.append((r * math.cos(t), r * math.sin(t)))
        self._biases = [rng.uniform(-abs(block_bias), abs(block_bias))
                        for _ in range(self.n_blocks)]
        self._noise = SmoothErrorField(self.seed + 1, length_scale=length_scale,
                                       n_components=6, amplitude=abs(float(noise_amp)))

    def error_deg(self, x: float, y: float) -> float:
        best_i, best_d = 0, float("inf")
        for i, (cx, cy) in enumerate(self._centers):
            d = (x - cx) * (x - cx) + (y - cy) * (y - cy)
            if d < best_d:
                best_d, best_i = d, i
        return _clamp1(self._biases[best_i] + self._noise.error_deg(x, y))


FIELD_KINDS = tuple(ERROR_FIELDS.available())


def make_error_field(kind: str = "smooth", seed: int = 0,
                     params: Optional[dict] = None, **kwargs) -> BaseErrorField:
    """Factory: build a field by kind via the registry. ``params`` (from
    ``config()``) and ``kwargs`` are merged into the constructor."""
    merged = dict(params or {})
    merged.update(kwargs)
    if kind not in ERROR_FIELDS:
        raise ValueError(
            f"unknown error field kind: {kind!r}; valid: {ERROR_FIELDS.available()}"
        )
    return ERROR_FIELDS.create(kind, seed=seed, **merged)
