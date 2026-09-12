"""
问题3 → 问题4 的定向性代价（共享同一物理下界）。

题面：光学定位/清除成败只与“清除点到源距离”有关，与信号覆盖角无关。因此若两案例源【同位置、
同频道、同 R_eff】，只是 Q3 全全向、Q4 部分改定向，则二者【绝对物理下界完全一样】。
故可构造配对案例，把“定向不可见性”造成的额外代价干净地隔离出来。

配对构造：先按 problem=3 生成基准案例（全全向），复制其 jammers，随机挑若干个改成 dir（随机朝向），
误差场与 R_eff 保持不变 → 两案例 LB 相同。指标：
  P_dir = T_Q4 / T_Q3               定向性额外困难倍率
  R3 = T_Q3 / T_LB, R4 = T_Q4 / T_LB
  R4 - R3                           定向不可见性对定位策略的额外归一化代价
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Optional

from jammerhunt import environment as env

from . import lowerbound as lb
from .interface_shim import LocalWorld
from .metrics import InstrumentedWorld, attach_certificate
from .runner import METHOD_FACTORIES


def make_paired_cases(seed: int, n_directional: Optional[int] = None,
                      **case_kwargs) -> tuple[env.Case, env.Case]:
    """
    返回 (caseA 全全向, caseB 同位置但部分定向)。两者源坐标/频道/R_eff/误差场完全一致。
    n_directional=None → 随机 1..m-1。
    """
    base = env.generate_case(seed=seed, problem=3, **case_kwargs)
    jam_a = [env.Jammer(j.channel, j.x, j.y, j.r_eff, "omni", None) for j in base.jammers]
    caseA = env.Case(jam_a, field=base.field, seed=seed, problem=3, mode=base.mode)

    rng = random.Random((seed << 1) ^ 0x9E3779B9)
    m = len(base.jammers)
    nd = n_directional if n_directional is not None else rng.randint(1, max(1, m - 1))
    nd = max(1, min(m - 1, nd))
    idx = set(rng.sample(range(m), nd))
    jam_b = []
    for i, j in enumerate(base.jammers):
        if i in idx:
            jam_b.append(env.Jammer(j.channel, j.x, j.y, j.r_eff, "dir", rng.uniform(0, 360)))
        else:
            jam_b.append(env.Jammer(j.channel, j.x, j.y, j.r_eff, "omni", None))
    caseB = env.Case(jam_b, field=base.field, seed=seed, problem=4, mode=base.mode)
    return caseA, caseB


def _run_case(method: str, case: env.Case, problem: int, hunter_kwargs: Optional[dict]):
    reveal = case.reveal()
    sources = [(j["x"], j["y"]) for j in reveal["jammers"]]
    cert = lb.certificate(sources)
    engine = env.Engine(case)
    world = InstrumentedWorld(LocalWorld(engine))
    strat = METHOD_FACTORIES[method](problem, reveal, **(hunter_kwargs or {}))
    err = None
    try:
        strat.run(world)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    return {
        "T": engine.virtual_time_s, "success": case.cleared_count == case.total,
        "cleared": case.cleared_count, "total": case.total,
        "T_abs_lb": cert.T_abs_lb(), "err": err,
    }


@dataclass
class DirectionalPairResult:
    seed: int
    T_q3: float
    T_q4: float
    T_lb: float
    ok3: bool
    ok4: bool

    @property
    def P_dir(self) -> float:
        return self.T_q4 / self.T_q3 if self.T_q3 > 1e-9 else float("inf")

    @property
    def R3(self) -> float:
        return self.T_q3 / self.T_lb if self.T_lb > 1e-9 else float("inf")

    @property
    def R4(self) -> float:
        return self.T_q4 / self.T_lb if self.T_lb > 1e-9 else float("inf")


def directional_cost(n: int = 100, base_seed: int = 5000, method: str = "ours",
                     hunter_kwargs: Optional[dict] = None, **case_kwargs):
    """在 n 组配对案例上比较 Q3(全向) 与 Q4(部分定向)，返回 (结果列表, 汇总 dict)。"""
    results: list[DirectionalPairResult] = []
    for i in range(n):
        seed = base_seed + i
        caseA, caseB = make_paired_cases(seed, **case_kwargs)
        ra = _run_case(method, caseA, 3, hunter_kwargs)
        rb = _run_case(method, caseB, 4, hunter_kwargs)
        # 同源位置 → LB 相同；取 A 的即可（数值上二者一致）
        results.append(DirectionalPairResult(
            seed=seed, T_q3=ra["T"], T_q4=rb["T"], T_lb=ra["T_abs_lb"],
            ok3=ra["success"], ok4=rb["success"]))
    both = [r for r in results if r.ok3 and r.ok4]
    summary = {
        "n": n,
        "n_both_success": len(both),
        "success3": sum(1 for r in results if r.ok3) / n if n else 0.0,
        "success4": sum(1 for r in results if r.ok4) / n if n else 0.0,
        "P_dir_median": statistics.median([r.P_dir for r in both]) if both else float("nan"),
        "P_dir_mean": statistics.fmean([r.P_dir for r in both]) if both else float("nan"),
        "R3_median": statistics.median([r.R3 for r in both]) if both else float("nan"),
        "R4_median": statistics.median([r.R4 for r in both]) if both else float("nan"),
        "dR_median": statistics.median([r.R4 - r.R3 for r in both]) if both else float("nan"),
    }
    return results, summary


def format_directional(summary: dict) -> str:
    return "\n".join([
        f"配对案例数 n = {summary['n']}   (两问都全清: {summary['n_both_success']})",
        f"全清率  Q3={summary['success3']*100:.1f}%   Q4={summary['success4']*100:.1f}%",
        f"P_dir = T_Q4/T_Q3   中位/均值 = {summary['P_dir_median']:.3f} / {summary['P_dir_mean']:.3f}",
        f"R3 = T_Q3/T_LB 中位 = {summary['R3_median']:.3f}",
        f"R4 = T_Q4/T_LB 中位 = {summary['R4_median']:.3f}",
        f"R4 - R3 中位 = {summary['dR_median']:.3f}   (定向不可见性的额外归一化代价)",
    ])
