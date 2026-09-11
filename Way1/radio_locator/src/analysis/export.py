"""
导出：把运行结果（日志、指标、频道最终状态）落成 JSON / CSV，供论文表格与复盘。
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from typing import Dict, List, Optional, Sequence

from ..domain.world_state import WorldState


def export_log_csv(log: Sequence, out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "kind", "channel", "x", "y", "virtual_time", "mec_radius", "note"])
        for e in log:
            loc = getattr(e, "location", None)
            w.writerow([
                getattr(e, "step", ""), getattr(e, "kind", ""), getattr(e, "channel", ""),
                f"{loc[0]:.3f}" if loc else "", f"{loc[1]:.3f}" if loc else "",
                f"{getattr(e, 'virtual_time', 0.0):.4f}",
                f"{getattr(e, 'mec_radius', float('nan')):.4f}" if getattr(e, "mec_radius", None) is not None else "",
                getattr(e, "note", ""),
            ])


def export_summary_json(summary: Dict, metrics_obj, out_path: str) -> None:
    payload = dict(summary)
    if metrics_obj is not None:
        payload["metrics"] = _to_dict(metrics_obj)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def export_channels_json(world: WorldState, out_path: str) -> None:
    data = []
    for c, cs in sorted(world.channels.items()):
        data.append({
            "channel": c,
            "status": cs.status.value,
            "mec_center": list(cs.mec_center) if cs.mec_center else None,
            "mec_radius": None if cs.mec_radius == float("inf") else round(cs.mec_radius, 4),
            "n_observations": len(cs.observations),
            "source_type_possible_omni": cs.source_type_possible_omni,
            "source_type_possible_directional": cs.source_type_possible_directional,
        })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _to_dict(obj):
    if is_dataclass(obj):
        return {k: _to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if obj == float("inf"):
        return None
    return obj
