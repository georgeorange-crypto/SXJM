"""
把“方法 × 案例”跑成 EpisodeMetrics：底层用 Way3 environment 引擎（真值已知），
外面套 InstrumentedWorld 拆计时，再用真值证书填 下界/Oracle 倍率与归因。

方法注册表 METHODS：name → factory(problem, reveal, **kw) → 有 .run(world) 的策略对象。
  lb / oracle  需要真值 reveal；其余现实方法只用可观测信息。
  （lb 不是可运行“策略”，它的时间直接由证书解析给出，见 run_episode 特判。）

聚合按【字典序】：先比全清成功率，再比成功局时间——漏源的方法没有资格用低耗时刷分。
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Callable, Optional

from jammerhunt import environment as env

from . import lowerbound as lb
from .ablations import make_ablation
from .baselines import make_greedy_scan, make_reactive
from .interface_shim import LocalWorld
from .metrics import (
    EpisodeMetrics,
    InstrumentedWorld,
    TimeBreakdown,
    attach_certificate,
    oracle_time,
)
from .oracle import OracleHunter

Point = tuple[float, float]


# --------------------------------------------------------------------------- #
# 方法注册
# --------------------------------------------------------------------------- #
def _ours(problem, reveal, **kw):
    from jammerhunt.agent import Hunter
    return Hunter(problem=problem, **kw)


def _oracle(problem, reveal, **kw):
    return OracleHunter(reveal=reveal)


METHOD_FACTORIES: dict[str, Callable] = {
    "ours": _ours,
    "oracle": _oracle,
    "reactive": lambda problem, reveal, **kw: make_reactive(problem, **kw),
    "greedy_scan": lambda problem, reveal, **kw: make_greedy_scan(problem, **kw),
    "no_global_scan": lambda problem, reveal, **kw: make_ablation("no_global_scan", problem, **kw),
    "no_global_route": lambda problem, reveal, **kw: make_ablation("no_global_route", problem, **kw),
    "no_active_sensing": lambda problem, reveal, **kw: make_ablation("no_active_sensing", problem, **kw),
    "no_replanning": lambda problem, reveal, **kw: make_ablation("no_replanning", problem, **kw),
}

# "lb" 是解析层（无可运行策略），单独处理。
ALL_METHODS = ["lb", "oracle", "reactive", "greedy_scan",
               "no_global_scan", "no_global_route", "no_active_sensing", "no_replanning", "ours"]


def source_points(case: env.Case) -> list[Point]:
    return [(j.x, j.y) for j in case.jammers]


# --------------------------------------------------------------------------- #
# 单局
# --------------------------------------------------------------------------- #
def run_episode(method: str, seed: int, problem: int = 3,
                hunter_kwargs: Optional[dict] = None, **case_kwargs) -> EpisodeMetrics:
    """跑一个方法在一个可复现案例上，返回带证书的 EpisodeMetrics。"""
    case = env.generate_case(seed=seed, problem=problem, **case_kwargs)
    reveal = case.reveal()
    sources = [(j["x"], j["y"]) for j in reveal["jammers"]]
    cert = lb.certificate(sources)

    if method == "lb":
        # 下界“伪局”：时间 = T_abs_lb，视作完美全清；计时分解按下界含义填充。
        m = EpisodeMetrics(
            seed=seed, problem=problem, method="lb", total=case.total,
            cleared=case.total, success=True, n_omni=case.n_omni, n_dir=case.n_dir,
            finish_reason="lower_bound",
        )
        attach_certificate(m, cert)
        m.T = m.T_abs_lb
        m.bd = TimeBreakdown(
            move_s=cert.T_move_LB, detect_s=0.0, switch_s=0.0,
            clear_s=lb.PER_SOURCE_S * case.total, n_clear=case.total,
            n_clear_hit=case.total, move_m=cert.L_LB,
        )
        return m

    if method not in METHOD_FACTORIES:
        raise ValueError(f"unknown method {method!r}; choices={ALL_METHODS}")

    engine = env.Engine(case)
    world = InstrumentedWorld(LocalWorld(engine))
    strat = METHOD_FACTORIES[method](problem, reveal, **(hunter_kwargs or {}))
    err = None
    try:
        strat.run(world)
    except Exception as e:                       # 策略 bug 不该崩掉整批
        err = f"{type(e).__name__}: {e}"

    m = EpisodeMetrics(
        seed=seed, problem=problem, method=method, total=case.total,
        cleared=case.cleared_count, success=(case.cleared_count == case.total),
        n_omni=case.n_omni, n_dir=case.n_dir, finish_reason=engine.finish_reason, error=err,
        T=engine.virtual_time_s, bd=world.bd,
    )
    attach_certificate(m, cert)
    # 一致性：InstrumentedWorld 累加应等于引擎虚拟钟（浮点/微秒量级容差）。
    if abs(m.bd.total_s - engine.virtual_time_s) > 1e-3 and err is None:
        m.error = (m.error or "") + f" [timing mismatch bd={m.bd.total_s:.3f} eng={engine.virtual_time_s:.3f}]"
    return m


# --------------------------------------------------------------------------- #
# 汇总（字典序：清除率优先，再时间）
# --------------------------------------------------------------------------- #
def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    idx = q * (len(s) - 1)
    lo, hi = int(math.floor(idx)), int(math.ceil(idx))
    if lo == hi:
        return s[lo]
    return s[lo] * (hi - idx) + s[hi] * (idx - lo)


@dataclass
class MethodSummary:
    method: str
    n: int = 0
    n_success: int = 0
    n_error: int = 0
    success_rate: float = 0.0
    # 官方口径：定位清除总时间 / 被清除源数（对全部局求平均）
    time_per_cleared: float = 0.0
    # 成功局时间（次级排序键）
    time_mean_success: float = 0.0
    time_p50_success: float = 0.0
    time_p90_success: float = 0.0
    time_max_success: float = 0.0
    # 相对下界 / Oracle（仅成功局，避免失败局污染）
    R_LB_median: float = 0.0
    R_LB_mean: float = 0.0
    R_LB_p90: float = 0.0
    R_LB_max: float = 0.0
    R_oracle_median: float = 0.0
    R_oracle_mean: float = 0.0
    # 归因（成功局均值）
    dT_move_mean: float = 0.0
    dT_info_mean: float = 0.0
    C_move_mean: float = 0.0
    C_sense_mean: float = 0.0
    # 证书质量
    cert_gap_rel_mean: float = 0.0
    results: list = field(default_factory=list, repr=False)

    def sort_key(self) -> tuple[float, float]:
        """字典序排序键：成功率降序 → 成功局均值时间升序。（越小越好，成功率取负）"""
        return (-self.success_rate, self.time_mean_success if self.n_success else math.inf)

    def report(self) -> str:
        return "\n".join([
            f"[{self.method}]  n={self.n}  全清率={self.success_rate*100:.2f}% ({self.n_success}/{self.n})"
            f"  异常={self.n_error}",
            f"  官方口径 平均(总时间/清除源数) = {self.time_per_cleared:.2f} s/源",
            f"  成功局时间 均值/P50/P90/max = "
            f"{self.time_mean_success:.1f} / {self.time_p50_success:.1f} / "
            f"{self.time_p90_success:.1f} / {self.time_max_success:.1f} s",
            f"  R_LB(=T/T_abs_lb)   中位/均值/P90/max = "
            f"{self.R_LB_median:.3f} / {self.R_LB_mean:.3f} / {self.R_LB_p90:.3f} / {self.R_LB_max:.3f}",
            f"  R_oracle            中位/均值           = "
            f"{self.R_oracle_median:.3f} / {self.R_oracle_mean:.3f}",
            f"  归因(成功局均值)  ΔT_move={self.dT_move_mean:.1f}s  ΔT_info={self.dT_info_mean:.1f}s"
            f"  C_move={self.C_move_mean:.2f}  C_sense={self.C_sense_mean:.2f}",
        ])


def summarize(method: str, results: list[EpisodeMetrics]) -> MethodSummary:
    s = MethodSummary(method=method, results=results)
    s.n = len(results)
    if s.n == 0:
        return s
    s.n_success = sum(1 for r in results if r.success)
    s.n_error = sum(1 for r in results if r.error)
    s.success_rate = s.n_success / s.n

    # 官方口径：Σ时间 / Σ清除源数（对全部局）
    tot_time = sum(r.T for r in results)
    tot_cleared = sum(r.cleared for r in results)
    s.time_per_cleared = tot_time / tot_cleared if tot_cleared > 0 else math.inf

    succ = [r for r in results if r.success]
    if succ:
        ts = [r.T for r in succ]
        s.time_mean_success = statistics.fmean(ts)
        s.time_p50_success = _pct(ts, 0.50)
        s.time_p90_success = _pct(ts, 0.90)
        s.time_max_success = max(ts)
        rlb = [r.R_LB for r in succ if math.isfinite(r.R_LB)]
        ror = [r.R_oracle for r in succ if math.isfinite(r.R_oracle)]
        if rlb:
            s.R_LB_median = _pct(rlb, 0.50)
            s.R_LB_mean = statistics.fmean(rlb)
            s.R_LB_p90 = _pct(rlb, 0.90)
            s.R_LB_max = max(rlb)
        if ror:
            s.R_oracle_median = _pct(ror, 0.50)
            s.R_oracle_mean = statistics.fmean(ror)
        s.dT_move_mean = statistics.fmean([r.dT_move for r in succ])
        s.dT_info_mean = statistics.fmean([r.dT_info for r in succ])
        cm = [r.C_move for r in succ if math.isfinite(r.C_move)]
        cs = [r.C_sense for r in succ if math.isfinite(r.C_sense)]
        s.C_move_mean = statistics.fmean(cm) if cm else 0.0
        s.C_sense_mean = statistics.fmean(cs) if cs else 0.0
    s.cert_gap_rel_mean = statistics.fmean([r.cert_gap_rel for r in results])
    return s


def run_method(method: str, n: int, problem: int = 3, base_seed: int = 1000,
               hunter_kwargs: Optional[dict] = None, **case_kwargs) -> MethodSummary:
    results = [run_episode(method, base_seed + i, problem, hunter_kwargs, **case_kwargs)
               for i in range(n)]
    return summarize(method, results)


def run_all(methods: Optional[list[str]] = None, n: int = 100, problem: int = 3,
            base_seed: int = 1000, hunter_kwargs: Optional[dict] = None,
            **case_kwargs) -> dict[str, MethodSummary]:
    """在【同一批种子】上跑多方法，保证可比。返回 {method: summary}。"""
    methods = methods or ALL_METHODS
    out: dict[str, MethodSummary] = {}
    for mth in methods:
        out[mth] = run_method(mth, n, problem, base_seed, hunter_kwargs, **case_kwargs)
    return out
