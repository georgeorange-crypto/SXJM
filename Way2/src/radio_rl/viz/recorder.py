"""Capture an episode as a JSON-serializable trace for the web viewer.

The recorder owns no policy and no physics: it builds a :class:`Pipeline` from a
config, runs one episode with ``collect_trace=True``, and reshapes the frozen
:class:`StepTrace` list into the flat dict the browser draws from. Ground truth
(jammer positions, kinds, headings, effective radii) comes from
``env.reveal()``; it is available only in *practice* mode and is used purely for
rendering the "true" markers — the agent never sees it.

A recorded frame is one executed action. For each frame we keep:

* ``pose``      the robot pose *after* the action (x, y) — the route vertex;
* ``action``    type / channel / target of the instruction issued;
* ``result``    the observation outcome (signal / no_signal / near / clear_*);
* ``bearing``   the noisy svd (deg) when a SIGNAL was returned, else null;
* ``vt``        accumulated virtual (task) time in seconds after the action.

The viewer reconstructs everything else — the detection fan is drawn from the
scan pose along ``bearing`` with the +/-1 deg error wedge; the route is the
poly-line through successive poses; a source turns from grey to its channel
colour when a CLEAR_SUCCESS names its channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.constants import CONSTANTS
from ..core.datatypes import ActionType, ObservationType
from ..pipeline import Pipeline

# Human-readable names the browser keys on (kept out of the JS so the two
# layers agree on one spelling of each enum).
_ACTION_NAME = {
    int(ActionType.SCAN): "scan",
    int(ActionType.CLEAR): "clear",
    int(ActionType.EXIT): "exit",
}
_RESULT_NAME = {
    int(ObservationType.RESET): "reset",
    int(ObservationType.SIGNAL): "signal",
    int(ObservationType.NO_SIGNAL): "no_signal",
    int(ObservationType.TOO_STRONG): "near",
    int(ObservationType.CLEAR_SUCCESS): "clear_success",
    int(ObservationType.CLEAR_FAILURE): "clear_failure",
}


@dataclass
class RunRecord:
    """Everything the viewer needs for one algorithm on one case."""

    label: str
    problem: int
    seed: Optional[int]
    arena_radius: float
    bearing_error_deg: float
    frames: list[dict]
    truth: Optional[dict]           # None outside practice mode
    summary: dict
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "problem": self.problem,
            "seed": self.seed,
            "arena_radius": self.arena_radius,
            "bearing_error_deg": self.bearing_error_deg,
            "frames": self.frames,
            "truth": self.truth,
            "summary": self.summary,
            "meta": self.meta,
        }


def _truth_from_env(env: Any) -> Optional[dict]:
    """Pull ground-truth jammers from a practice-mode env, if exposed."""
    reveal = getattr(env, "reveal", None)
    if reveal is None:
        return None
    try:
        info = reveal()
    except Exception:
        return None
    jammers = []
    for j in info.get("jammers", []):
        jammers.append(
            {
                "channel": int(j["channel"]),
                "x": float(j["x"]),
                "y": float(j["y"]),
                "r_eff": float(j["r_eff"]),
                "kind": str(j["kind"]),
                "direction_deg": (
                    None if j.get("direction_deg") is None
                    else float(j["direction_deg"])
                ),
                "cleared": bool(j.get("cleared", False)),
            }
        )
    return {
        "total": int(info.get("total", len(jammers))),
        "n_omni": int(info.get("n_omni", 0)),
        "n_dir": int(info.get("n_dir", 0)),
        "jammers": jammers,
    }


def _frame_from_trace(step: Any) -> dict:
    """Reshape one StepTrace into a compact JSON frame."""
    a = step.action
    o = step.observation
    at = int(a.action_type)
    rt = int(o.result_type)
    frame = {
        "type": _ACTION_NAME.get(at, str(at)),
        "channel": int(a.channel),
        # target the instruction aimed at (pre-clamp); pose is where we ended up
        "target": [round(float(a.target_x), 3), round(float(a.target_y), 3)],
        "pose": [round(float(o.position_x), 3), round(float(o.position_y), 3)],
        "result": _RESULT_NAME.get(rt, str(rt)),
        "vt": round(float(step.virtual_time_s), 6),
        "n_candidates": int(step.n_candidates),
    }
    if o.bearing_deg is not None:
        frame["bearing"] = round(float(o.bearing_deg), 2)
    if o.clear_success is not None:
        frame["clear_success"] = bool(o.clear_success)
    return frame


def record_run(
    cfg: Any,
    *,
    seed: Optional[int] = None,
    label: Optional[str] = None,
    max_steps: int = 100_000,
) -> RunRecord:
    """Run one episode of the configured pipeline and capture its trace."""
    pipe = Pipeline(cfg)
    stats, trace = pipe.run_episode(
        seed=seed, max_steps=max_steps, collect_trace=True
    )

    frames = [_frame_from_trace(s) for s in trace]
    truth = _truth_from_env(pipe.env)

    algo_cfg = getattr(cfg, "algorithm", None)
    algo_name = None
    if algo_cfg is not None:
        get = getattr(algo_cfg, "get", None)
        if get is not None:
            algo_name = algo_cfg.get("name", algo_cfg.get("type"))
    label = label or (str(algo_name) if algo_name else "run")

    summary = {
        "sources_total": stats.sources_total,
        "sources_cleared": stats.sources_cleared,
        "clear_ratio": round(stats.clear_ratio, 4),
        "virtual_time_s": round(stats.virtual_time, 3),
        "avg_clear_time_s": (
            None if stats.avg_clear_time == float("inf")
            else round(stats.avg_clear_time, 3)
        ),
        "route_distance_m": round(stats.route_distance, 2),
        "num_scans": stats.num_scans,
        "num_clears": stats.num_clears,
        "failed_clears": stats.failed_clears,
        "num_switches": stats.num_switches,
        "steps": stats.steps,
        "wall_time_s": round(stats.wall_time, 4),
    }

    return RunRecord(
        label=label,
        problem=int(getattr(pipe, "problem", 3)),
        seed=seed,
        arena_radius=float(CONSTANTS.region_radius),
        bearing_error_deg=float(CONSTANTS.bearing_error_deg),
        frames=frames,
        truth=truth,
        summary=summary,
        meta={
            "clear_radius_m": float(CONSTANTS.clear_radius),
            "near_radius_m": float(CONSTANTS.too_strong_radius),
            "directional_half_angle_deg": float(CONSTANTS.directional_half_angle_deg),
            "n_frames": len(frames),
        },
    )


def record_runs(
    specs: list[dict],
    *,
    seed: Optional[int] = None,
    max_steps: int = 100_000,
) -> list[RunRecord]:
    """Record several algorithms on the *same* case for side-by-side viewing.

    Each spec is ``{"cfg": <composed cfg>, "label": <str>}``. Passing the same
    ``seed`` to every run means every algorithm faces an identical jammer layout,
    which is the only fair way to compare their routes on one canvas.
    """
    out: list[RunRecord] = []
    for spec in specs:
        out.append(
            record_run(
                spec["cfg"],
                seed=seed,
                label=spec.get("label"),
                max_steps=max_steps,
            )
        )
    return out
