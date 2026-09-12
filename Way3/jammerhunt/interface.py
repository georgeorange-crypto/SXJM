"""
World 抽象层：让同一套 agent 策略在三种后端上运行，互不感知差异。

后端：
1) LocalWorld —— 包裹本包 environment.Engine，进程内、最快，用于蒙特卡洛主评测。
2) RunnerWorld —— 适配共享 offline_sim 的 EpisodeRunner，白嫖它丰富的压力/对抗案例
   与统计汇总（可选依赖；导入失败不影响 1、3）。
3) HttpWorld —— 通过 HTTP+JSON 连真实模拟器（或 offline_sim.SimServer 做集成测试）。
   自带 request_id 幂等 + 断线按原 id 重试（附件2 §5.3）。

agent 只依赖下面的统一契约：
    world.enter() -> bool
    world.measure(x,y,ch) -> MeasureObs      # .ok / .is_direction / .svd_deg / .is_near ...
    world.clear(x,y,ch)   -> ClearObs        # .ok / .success
    world.exit()
    world.position / world.channel / world.virtual_time_s / world.finished

坐标(0,0)起步、测向机初始频道 1（附件1 §1）。仅依赖标准库。
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

Point = tuple[float, float]


# --------------------------------------------------------------------------- #
# 统一观测对象
# --------------------------------------------------------------------------- #
@dataclass
class MeasureObs:
    ok: bool                                # False = 时限到/接口关闭，应结束本局
    result: Optional[str] = None            # "no_signal" | "near" | "direction"
    svd_deg: Optional[float] = None

    @property
    def is_direction(self) -> bool:
        return self.result == "direction"

    @property
    def is_near(self) -> bool:
        return self.result == "near"

    @property
    def is_no_signal(self) -> bool:
        return self.result == "no_signal"


@dataclass
class ClearObs:
    ok: bool
    result: Optional[str] = None            # "success" | "no_target_in_range"

    @property
    def success(self) -> bool:
        return self.result == "success"


# --------------------------------------------------------------------------- #
# 抽象基类
# --------------------------------------------------------------------------- #
class World:
    def enter(self) -> bool:
        raise NotImplementedError

    def measure(self, x: float, y: float, channel: int) -> MeasureObs:
        raise NotImplementedError

    def clear(self, x: float, y: float, channel: int) -> ClearObs:
        raise NotImplementedError

    def exit(self) -> None:
        raise NotImplementedError

    @property
    def position(self) -> Point:
        raise NotImplementedError

    @property
    def channel(self) -> int:
        raise NotImplementedError

    @property
    def virtual_time_s(self) -> float:
        raise NotImplementedError

    @property
    def finished(self) -> bool:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# 1) 本包 Engine
# --------------------------------------------------------------------------- #
class LocalWorld(World):
    def __init__(self, engine):
        self.engine = engine

    def enter(self) -> bool:
        self.engine.enter()
        return True

    def measure(self, x, y, channel) -> MeasureObs:
        resp, out = self.engine.measure(x, y, channel)
        if resp.get("__finished__"):
            return MeasureObs(ok=False)
        return MeasureObs(ok=True, result=out.result, svd_deg=out.svd_deg)

    def clear(self, x, y, channel) -> ClearObs:
        resp, out = self.engine.clear(x, y, channel)
        if resp.get("__finished__"):
            return ClearObs(ok=False)
        return ClearObs(ok=True, result=out.result)

    def exit(self) -> None:
        self.engine.exit()

    @property
    def position(self) -> Point:
        return (self.engine.x, self.engine.y)

    @property
    def channel(self) -> int:
        return self.engine.channel

    @property
    def virtual_time_s(self) -> float:
        return self.engine.virtual_time_s

    @property
    def finished(self) -> bool:
        return self.engine.finished


# --------------------------------------------------------------------------- #
# 2) offline_sim.EpisodeRunner 适配器（可选）
# --------------------------------------------------------------------------- #
class RunnerWorld(World):
    """
    适配 offline_sim.harness.EpisodeRunner：其 measure/clear 返回 dict 或 None(时限到)。
    借此把本包 agent 直接投喂给 offline_sim 的 evaluate / evaluate_stress。
    """

    def __init__(self, runner):
        self.runner = runner
        self._finished = False

    def enter(self) -> bool:
        self.runner.enter()
        return True

    def measure(self, x, y, channel) -> MeasureObs:
        d = self.runner.measure(x, y, channel)
        if d is None:
            self._finished = True
            return MeasureObs(ok=False)
        return MeasureObs(ok=True, result=d.get("measure_result"), svd_deg=d.get("svd_deg"))

    def clear(self, x, y, channel) -> ClearObs:
        d = self.runner.clear(x, y, channel)
        if d is None:
            self._finished = True
            return ClearObs(ok=False)
        return ClearObs(ok=True, result=d.get("clear_result"))

    def exit(self) -> None:
        self.runner.exit()

    @property
    def position(self) -> Point:
        return self.runner.position

    @property
    def channel(self) -> int:
        return self.runner.current_channel

    @property
    def virtual_time_s(self) -> float:
        return self.runner.virtual_time_s

    @property
    def finished(self) -> bool:
        return self._finished


# --------------------------------------------------------------------------- #
# 3) HTTP 客户端 + HttpWorld（连真实模拟器）
# --------------------------------------------------------------------------- #
class HttpClient:
    """
    最小 HTTP+JSON 客户端（附件2 §5）：
    - 每个“新动作”用新的 request_id；网络失败按【同一 request_id】重试（幂等，§5.3）。
    - 返回 (http_status, body_or_None)。连接被直接关闭（无体）→ (0, None)。
    """

    def __init__(self, base_url: str, robot_id: str, arena_id: str = "default",
                 timeout: float = 5.0, retries: int = 4, retry_wait: float = 0.05):
        self.base_url = base_url.rstrip("/")
        self.robot_id = robot_id
        self.arena_id = arena_id
        self.timeout = timeout
        self.retries = retries
        self.retry_wait = retry_wait
        self._seq = itertools.count(1)

    def _post_once(self, path: str, payload: dict) -> tuple[int, Optional[dict]]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = Request(self.base_url + path, data=data,
                      headers={"Content-Type": "application/json; charset=utf-8"},
                      method="POST")
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            try:
                return e.code, json.loads(e.read().decode("utf-8"))
            except Exception:
                return e.code, None
        except (URLError, ConnectionError, OSError):
            return 0, None

    def call(self, path: str, extra: Optional[dict] = None) -> tuple[int, Optional[dict]]:
        """发起一个新动作，失败按同一 request_id 重试。extra 含 position/channel。"""
        rid = f"{path.strip('/')}-{next(self._seq)}"
        payload = {"arena_id": self.arena_id, "robot_id": self.robot_id, "request_id": rid}
        if extra:
            payload.update(extra)
        status, body = self._post_once(path, payload)
        attempt = 0
        # 网络类失败 / 429 / 500 → 同 id 重试（幂等安全）；连接关闭(0,None)也重试几次
        while attempt < self.retries and (status in (0, 429, 500)):
            time.sleep(self.retry_wait)
            status, body = self._post_once(path, payload)
            attempt += 1
        return status, body


class HttpWorld(World):
    """通过 HttpClient 连真实模拟器；本地推算位置/频道（服务器不返回位置）。"""

    def __init__(self, client: HttpClient):
        self.client = client
        self._x = 0.0
        self._y = 0.0
        self._ch = 1
        self._vt = 0.0
        self._finished = False

    def _accepted(self, body: Optional[dict]) -> bool:
        return bool(body) and body.get("accepted") is True

    def enter(self) -> bool:
        status, body = self.client.call("/enter")
        if not self._accepted(body):
            self._finished = True
            return False
        self._x, self._y, self._ch = 0.0, 0.0, 1
        self._vt = float(body.get("virtual_time_s", 0.0))
        return True

    def measure(self, x, y, channel) -> MeasureObs:
        status, body = self.client.call(
            "/measure", {"position": {"x": x, "y": y}, "channel": int(channel)})
        if not self._accepted(body):
            self._finished = True
            return MeasureObs(ok=False)
        self._x, self._y, self._ch = x, y, int(channel)
        self._vt = float(body.get("virtual_time_s", self._vt))
        return MeasureObs(ok=True, result=body.get("measure_result"),
                          svd_deg=body.get("svd_deg"))

    def clear(self, x, y, channel) -> ClearObs:
        status, body = self.client.call(
            "/clear", {"position": {"x": x, "y": y}, "channel": int(channel)})
        if not self._accepted(body):
            self._finished = True
            return ClearObs(ok=False)
        self._x, self._y = x, y                     # /clear 不改频道
        self._vt = float(body.get("virtual_time_s", self._vt))
        return ClearObs(ok=True, result=body.get("clear_result"))

    def exit(self) -> None:
        self.client.call("/exit")
        self._finished = True

    @property
    def position(self) -> Point:
        return (self._x, self._y)

    @property
    def channel(self) -> int:
        return self._ch

    @property
    def virtual_time_s(self) -> float:
        return self._vt

    @property
    def finished(self) -> bool:
        return self._finished
