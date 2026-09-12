"""Frozen evaluation split and paired-comparison protocol for Way4."""
from dataclasses import dataclass
from typing import Tuple
import math

@dataclass(frozen=True)
class EvaluationSplits:
    train: Tuple[int, ...] = tuple(range(2000, 2040))
    validation: Tuple[int, ...] = tuple(range(2040, 2050))
    test: Tuple[int, ...] = tuple(range(2050, 2060))
    stress: Tuple[int, ...] = tuple(range(3000, 3020))
    # Hand-selected geometry/regime probes; never used for tuning.
    adversarial: Tuple[int, ...] = tuple(range(4000, 4010))
    def assert_disjoint(self):
        sets = [set(self.train), set(self.validation), set(self.test), set(self.stress), set(self.adversarial)]
        assert all(not (sets[i] & sets[j]) for i in range(4) for j in range(i+1,4))
    def test_is_locked(self, seed):
        return seed in self.test and seed not in self.train and seed not in self.validation

    def split_of(self, seed):
        """Return the sole frozen split containing ``seed``, or ``None``."""
        for name in ("train", "validation", "test", "stress", "adversarial"):
            if seed in getattr(self, name):
                return name
        return None

DEFAULT_SPLITS = EvaluationSplits()
DEFAULT_SPLITS.assert_disjoint()


@dataclass(frozen=True)
class AblationSpec:
    """One reproducible deterministic/learning ablation cell."""

    name: str
    no_signal_localization: bool = True
    cardinality: bool = True
    adaptive_batch: bool = False
    stop: bool = False
    nbv: str = "minimax"
    coverage: str = "active_fallback"
    routing: str = "tspn"
    horizon: int = 1


DETERMINISTIC_ABLATIONS = (
    AblationSpec("way4_full"),
    AblationSpec("minus_no_signal", no_signal_localization=False),
    AblationSpec("minus_cardinality", cardinality=False),
    AblationSpec("adaptive_rerank", adaptive_batch=True),
    AblationSpec("adaptive_rerank_stop", adaptive_batch=True, stop=True),
    AblationSpec("nbv_greedy", nbv="greedy"),
    AblationSpec("coverage_backbone_only", coverage="backbone_only"),
    AblationSpec("routing_nearest", routing="nearest"),
)


def assert_ablation_matrix_valid(specs=DETERMINISTIC_ABLATIONS):
    names = [s.name for s in specs]
    assert len(names) == len(set(names)) and "way4_full" in names


def summarize_ablation_results(specs, results):
    """Aggregate only supplied real episode rows; never invent missing values.

    ``results`` maps cell name to rows containing ``success`` and optionally
    ``virtual_time_s``/``error``.  Missing cells and failed episodes remain
    explicit in the returned audit payload.
    """
    expected = {s.name for s in specs}
    out = []
    for spec in specs:
        rows = list(results.get(spec.name, ()))
        valid = [r for r in rows if r.get("error") is None and r.get("success") is not None]
        successes = sum(bool(r["success"]) for r in valid)
        times = sorted(float(r["virtual_time_s"]) for r in valid
                       if r.get("virtual_time_s") is not None)
        p90 = None
        if times:
            idx = 0.9 * (len(times) - 1)
            lo, hi = math.floor(idx), math.ceil(idx)
            p90 = times[lo] if lo == hi else times[lo] + (times[hi] - times[lo]) * (idx - lo)
        if not rows:
            status = "missing"
        elif not valid:
            status = "failed"
        elif len(valid) == len(rows):
            status = "complete"
        else:
            status = "partial"
        out.append({
            "name": spec.name,
            "status": status,
            "n_rows": len(rows),
            "n_valid": len(valid),
            "n_success": successes,
            "full_clear_rate": successes / len(valid) if valid else None,
            "time_mean": sum(times) / len(times) if times else None,
            "time_p90": p90,
            "errors": [r.get("error") for r in rows if r.get("error")],
        })
    return {"expected_cells": sorted(expected), "cells": out,
            "complete": all(c["status"] == "complete" for c in out)}
