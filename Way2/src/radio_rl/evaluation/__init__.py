"""Evaluation layer — score any configured pipeline over many seeds."""

from __future__ import annotations

from .evaluator import EvalConfig, Evaluator, evaluate
from .metrics import MetricsSummary, summarize
from .stress import run_stress, stress_types_for

__all__ = [
    "Evaluator",
    "EvalConfig",
    "evaluate",
    "MetricsSummary",
    "summarize",
    "run_stress",
    "stress_types_for",
]
