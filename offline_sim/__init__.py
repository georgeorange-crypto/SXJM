"""
offline_sim —— 无线电干扰源模拟器的本地离线复现。

严格落实《附件1 模拟器使用说明》《附件2 通信接口说明》中的：
  · 物理规则（有效接收半径、全向/定向覆盖、示向度误差、近距/清除半径）
  · 计时规则（移动/切换/检测/清除的虚拟耗时、微秒累计、时限）
  · HTTP+JSON 通信协议（方法/路径/头/体校验、状态码、accepted 语义、request_id 幂等）

模块：
  case.py    案例与干扰源、按位置固定的示向度误差场、随机案例生成
  engine.py  纯逻辑引擎（物理 + 计时 + 状态推进）
  server.py  HTTP 层（http.server），把引擎暴露为 http://127.0.0.1:2026
  client.py  轻量客户端（可选，便于自测/给机器狗复用）
  run.py     命令行启动器
"""

from .case import Case, Jammer, ErrorField, generate_case, generate_stress_case, STRESS_TYPES
from .fields import (
    make_error_field, FIELD_KINDS,
    SmoothErrorField, IIDErrorField, BiasedErrorField,
    AdversarialErrorField, PiecewiseErrorField,
)
from .engine import Engine
from .server import SimServer, SimSession, serve
from .harness import (
    EpisodeRunner, EpisodeResult, Metrics,
    run_episode, aggregate, evaluate, evaluate_stress,
)

__all__ = [
    "Case", "Jammer", "ErrorField", "generate_case", "generate_stress_case", "STRESS_TYPES",
    "make_error_field", "FIELD_KINDS",
    "SmoothErrorField", "IIDErrorField", "BiasedErrorField",
    "AdversarialErrorField", "PiecewiseErrorField",
    "Engine", "SimServer", "SimSession", "serve",
    "EpisodeRunner", "EpisodeResult", "Metrics",
    "run_episode", "aggregate", "evaluate", "evaluate_stress",
]
