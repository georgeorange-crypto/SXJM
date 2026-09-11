"""
配置加载：从 configs/*.yaml 读入并构造 ProblemConstants / RobotConstants，
其余参数以嵌套 dict 形式暴露给各模块。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml

from .types import ProblemConstants, RobotConstants

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Config:
    raw: Dict[str, Any]
    problem: ProblemConstants
    robot: RobotConstants

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


def load_config(problem: str = "q3", config_dir: str | None = None) -> Config:
    """载入 common.yaml + q{3,4}.yaml，深合并。problem ∈ {'q3','q4','common'}。"""
    cdir = config_dir or _CONFIG_DIR
    with open(os.path.join(cdir, "common.yaml"), "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if problem in ("q3", "q4"):
        with open(os.path.join(cdir, f"{problem}.yaml"), "r", encoding="utf-8") as f:
            raw = _deep_merge(raw, yaml.safe_load(f) or {})

    p = raw.get("problem", {})
    r = raw.get("radio", {})
    rob = raw.get("robot", {})
    loc = raw.get("localization", {})

    problem_c = ProblemConstants(
        area_radius=float(p.get("area_radius", 1800.0)),
        channels=int(p.get("channels", 20)),
        source_count_min=int(p.get("source_count_min", 10)),
        source_count_max=int(p.get("source_count_max", 16)),
        bearing_error_deg=float(r.get("bearing_error_deg", 1.0)),
        bearing_margin_deg=float(r.get("bearing_margin_deg", 0.1)),
        receive_radius_min=float(r.get("receive_radius_min", 1000.0)),
        receive_radius_max=float(r.get("receive_radius_max", 1500.0)),
        directional_half_angle_deg=float(r.get("directional_half_angle_deg", 90.0)),
    )
    robot_c = RobotConstants(
        speed=float(rob.get("speed", 5.0)),
        switch_time=float(rob.get("switch_time", 1.0)),
        detection_time=float(rob.get("detection_time", 5.0)),
        clear_hit_time=float(rob.get("clear_hit_time", 5.0)),
        clear_miss_time=float(rob.get("clear_miss_time", 3.0)),
        optical_radius=float(rob.get("optical_radius", 20.0)),
        too_strong_radius=float(rob.get("too_strong_radius", 5.0)),
        clear_mec_radius=float(loc.get("clear_mec_radius", 19.0)),
        radius_delete_margin=float(loc.get("radius_delete_margin", 1e-3)),
    )
    return Config(raw=raw, problem=problem_c, robot=robot_c)
