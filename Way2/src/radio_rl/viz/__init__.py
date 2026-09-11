"""Run visualization: capture an episode as a trace and render it as a
self-contained web page.

This package is a *thin observer* on top of the frozen pipeline. It never
changes how an episode runs; it re-runs ``Pipeline.run_episode`` with
``collect_trace=True`` and turns the resulting :class:`StepTrace` list (plus the
practice-mode ground truth from ``env.reveal()``) into a plain dict that a
browser can draw. Nothing here imports torch — recording a PPO run pulls the
tensor stack in only through the agent, exactly as a normal evaluation would.
"""

from __future__ import annotations

from .recorder import RunRecord, record_run, record_runs
from .report import render_html, write_report

__all__ = [
    "RunRecord",
    "record_run",
    "record_runs",
    "render_html",
    "write_report",
]
