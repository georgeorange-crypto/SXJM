"""
轻量客户端（可选）：封装 4 条指令，处理 request_id 自增、accepted 校验、网络重试复用。

与真实模拟器接口一致，机器狗策略可直接复用；也用于本包自测。
仅依赖标准库。
"""

from __future__ import annotations

import json
import itertools
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


class SimClient:
    def __init__(self, base_url: str, robot_id: str, timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.robot_id = robot_id
        self.timeout = timeout
        self._seq = itertools.count(1)
        self.last_virtual_time_s = 0.0     # 仅记录 accepted=true 的虚拟时刻

    def _new_request_id(self, tag: str) -> str:
        return f"{tag}-{next(self._seq)}"

    def _post(self, path: str, payload: dict) -> tuple[int, dict | None]:
        """
        返回 (http_status, json_or_None)。
        连接被直接关闭（无 JSON 体）时返回 (0, None)。
        """
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return resp.status, body
        except HTTPError as e:
            try:
                body = json.loads(e.read().decode("utf-8"))
            except Exception:
                body = None
            return e.code, body
        except (URLError, ConnectionError, OSError):
            return 0, None

    def _base(self, request_id: str) -> dict:
        return {"arena_id": "default", "robot_id": self.robot_id, "request_id": request_id}

    def _track(self, body: dict | None) -> None:
        if body and body.get("accepted") is True and "virtual_time_s" in body:
            self.last_virtual_time_s = body["virtual_time_s"]

    # ---- 四条指令 ----
    def enter(self) -> tuple[int, dict | None]:
        s, b = self._post("/enter", self._base(self._new_request_id("enter")))
        self._track(b)
        return s, b

    def measure(self, x: float, y: float, channel: int) -> tuple[int, dict | None]:
        p = self._base(self._new_request_id("measure"))
        p["position"] = {"x": x, "y": y}
        p["channel"] = channel
        s, b = self._post("/measure", p)
        self._track(b)
        return s, b

    def clear(self, x: float, y: float, channel: int) -> tuple[int, dict | None]:
        p = self._base(self._new_request_id("clear"))
        p["position"] = {"x": x, "y": y}
        p["channel"] = channel
        s, b = self._post("/clear", p)
        self._track(b)
        return s, b

    def exit(self) -> tuple[int, dict | None]:
        s, b = self._post("/exit", self._base(self._new_request_id("exit")))
        self._track(b)
        return s, b
