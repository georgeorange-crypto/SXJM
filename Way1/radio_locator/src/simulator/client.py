"""
真实模拟器 HTTP 客户端：/enter /measure /clear /exit。

要点（附件2 §5）：
- 每个新动作用新 request_id；仅网络重试同一动作时复用原 id 与原体。
- 同时判 HTTP 状态与 accepted；accepted=false 时 virtual_time_s=0 不作为当前时刻。
- 串行、逐个等待；用 remaining_real_duration_s 控现实时长。

本类与 MockSimulator 接口一致，可被 runtime 无感替换。
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .protocol import ClearResponse, EnterResponse, ExitResponse, MeasureResponse


class SimulatorClient:
    def __init__(self, base_url: str, robot_id: str, arena_id: str = "default", timeout_s: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.robot_id = robot_id
        self.arena_id = arena_id
        self.timeout_s = timeout_s
        self._last_vt = 0.0

    def _new_id(self, prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex[:16]}"

    def _post(self, path: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        data = json.dumps(payload).encode("utf-8")
        req = Request(self.base_url + path, data=data,
                      headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            try:
                return json.loads(e.read().decode("utf-8"))
            except Exception:
                return None
        except (URLError, TimeoutError, ConnectionError):
            return None  # 连不上：上层需同时处理

    def _base(self, prefix: str) -> Dict[str, Any]:
        return {"arena_id": self.arena_id, "robot_id": self.robot_id, "request_id": self._new_id(prefix)}

    def enter(self) -> EnterResponse:
        r = self._post("/enter", self._base("enter"))
        if not r or not r.get("accepted"):
            return EnterResponse(accepted=False)
        self._last_vt = r.get("virtual_time_s", 0.0)
        return EnterResponse(
            accepted=True,
            virtual_time_s=r.get("virtual_time_s", 0.0),
            max_virtual_duration_s=r.get("max_virtual_duration_s", 360000.0),
            max_real_duration_s=r.get("max_real_duration_s", 1200.0),
            remaining_real_duration_s=int(r.get("remaining_real_duration_s", 1200)),
        )

    def measure(self, x: float, y: float, channel: int) -> MeasureResponse:
        payload = self._base("measure")
        payload["position"] = {"x": x, "y": y}
        payload["channel"] = channel
        r = self._post("/measure", payload)
        if not r or not r.get("accepted"):
            return MeasureResponse(accepted=False)
        self._last_vt = r.get("virtual_time_s", self._last_vt)
        return MeasureResponse(accepted=True, virtual_time_s=r.get("virtual_time_s", 0.0),
                               measure_result=r.get("measure_result"), svd_deg=r.get("svd_deg"))

    def clear(self, x: float, y: float, channel: int) -> ClearResponse:
        payload = self._base("clear")
        payload["position"] = {"x": x, "y": y}
        payload["channel"] = channel
        r = self._post("/clear", payload)
        if not r or not r.get("accepted"):
            return ClearResponse(accepted=False)
        self._last_vt = r.get("virtual_time_s", self._last_vt)
        return ClearResponse(accepted=True, virtual_time_s=r.get("virtual_time_s", 0.0),
                             clear_result=r.get("clear_result"))

    def exit(self) -> ExitResponse:
        r = self._post("/exit", self._base("exit"))
        if not r or not r.get("accepted"):
            return ExitResponse(accepted=False)
        return ExitResponse(accepted=True, virtual_time_s=r.get("virtual_time_s", 0.0),
                            exit_reason=r.get("exit_reason"))
