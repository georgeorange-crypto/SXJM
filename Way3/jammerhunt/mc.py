"""
蒙特卡洛评测：全清成功率 + 平均定位-清除时间（第三/四问）。

与题目最重要的约束对齐——“确保清除所有干扰源”优先于速度：
  - success_rate        全清成功率（首要指标）
  - mean_time_success   成功局的平均定位-清除虚拟时间（=题目得分，其次）
另附时间分布 / 移动距离 / 测量次数 / 清除失败次数等诊断量。

两条评测通道：
  1) 本包 environment + LocalWorld —— 进程内、可复现、快，为主评测。
  2) offline_sim 桥（evaluate / evaluate_stress）—— 复用共享测试床的随机与【对抗】案例
     做独立复核（含把我方扫描点喂给对抗生成器，专挑覆盖缝隙）。best-effort：导入失败不影响 1。

运行：
  python -m jammerhunt.mc --problem 3 -n 200
  python -m jammerhunt.mc --problem 4 -n 100 --offline
"""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass, field as dc_field
from typing import Optional

from . import environment as env
from .agent import Hunter
from .interface import LocalWorld

Point = tuple[float, float]


# --------------------------------------------------------------------------- #
# 单局（本包 LocalWorld）
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeResult:
    seed: int
    problem: int
    total: int
    cleared: int
    success: bool
    virtual_time_s: float
    n_omni: int
    n_dir: int
    finish_reason: Optional[str] = None
    error: Optional[str] = None


def run_local_episode(seed: int, problem: int = 3,
                      hunter_kwargs: Optional[dict] = None,
                      **case_kwargs) -> EpisodeResult:
    """在本包 environment 上跑单局（真值来自 case，策略看不到）。"""
    case = env.generate_case(seed=seed, problem=problem, **case_kwargs)
    engine = env.Engine(case)
    err = None
    try:
        Hunter(problem=problem, **(hunter_kwargs or {})).run(LocalWorld(engine))
    except Exception as e:                       # 策略 bug 不该崩掉整批
        err = f"{type(e).__name__}: {e}"
    return EpisodeResult(
        seed=seed, problem=problem, total=case.total, cleared=case.cleared_count,
        success=(case.cleared_count == case.total), virtual_time_s=engine.virtual_time_s,
        n_omni=case.n_omni, n_dir=case.n_dir, finish_reason=engine.finish_reason, error=err,
    )


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
def _pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = q * (len(sorted_vals) - 1)
    lo, hi = int(math.floor(idx)), int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


@dataclass
class Summary:
    n: int = 0
    n_success: int = 0
    n_error: int = 0
    success_rate: float = 0.0
    # 时间（全部局）
    time_mean: float = 0.0
    time_p50: float = 0.0
    time_p90: float = 0.0
    time_max: float = 0.0
    # 时间（仅成功局，避免失败局污染“得分”）
    time_mean_success: float = 0.0
    time_p90_success: float = 0.0
    time_max_success: float = 0.0
    results: list = dc_field(default_factory=list, repr=False)

    def report(self) -> str:
        lines = [
            f"局数               : {self.n}",
            f"全清成功率         : {self.success_rate * 100:.2f}%  ({self.n_success}/{self.n})",
            f"策略异常           : {self.n_error}",
            f"虚拟时间(全部) 均值/P50/P90/max : "
            f"{self.time_mean:.1f} / {self.time_p50:.1f} / {self.time_p90:.1f} / {self.time_max:.1f} s",
            f"虚拟时间(成功) 均值/P90/max     : "
            f"{self.time_mean_success:.1f} / {self.time_p90_success:.1f} / {self.time_max_success:.1f} s",
        ]
        return "\n".join(lines)


def summarize(results: list[EpisodeResult]) -> Summary:
    s = Summary(results=results)
    s.n = len(results)
    if s.n == 0:
        return s
    s.n_success = sum(1 for r in results if r.success)
    s.n_error = sum(1 for r in results if r.error)
    s.success_rate = s.n_success / s.n
    times = sorted(r.virtual_time_s for r in results)
    s.time_mean = statistics.fmean(times)
    s.time_p50 = _pct(times, 0.50)
    s.time_p90 = _pct(times, 0.90)
    s.time_max = times[-1]
    succ = sorted(r.virtual_time_s for r in results if r.success)
    if succ:
        s.time_mean_success = statistics.fmean(succ)
        s.time_p90_success = _pct(succ, 0.90)
        s.time_max_success = succ[-1]
    return s


def monte_carlo(n: int = 200, problem: int = 3, base_seed: int = 1000,
                hunter_kwargs: Optional[dict] = None, **case_kwargs) -> Summary:
    """在 n 个可复现随机案例上评测本包策略（LocalWorld）。"""
    results = [run_local_episode(base_seed + i, problem, hunter_kwargs, **case_kwargs)
               for i in range(n)]
    return summarize(results)


# --------------------------------------------------------------------------- #
# offline_sim 桥（best-effort 独立复核）
# --------------------------------------------------------------------------- #
def _ensure_offline_on_path() -> bool:
    """把仓库根（Way3 的上级）加入 sys.path，以便 import offline_sim（共享只读测试床）。"""
    import os
    import sys
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import offline_sim  # noqa: F401
        return True
    except Exception:
        return False


def offline_evaluate(problem: int = 3, n_cases: int = 100, base_seed: int = 0,
                     field_kind: str = "smooth", hunter_kwargs: Optional[dict] = None):
    """用共享 offline_sim.harness.evaluate 复核本包策略（随机案例）。返回其 Metrics 或 None。"""
    if not _ensure_offline_on_path():
        return None
    from offline_sim import harness
    from .agent import make_runner_policy
    policy = make_runner_policy(problem, **(hunter_kwargs or {}))
    return harness.evaluate(policy, n_cases=n_cases, problem=problem,
                            base_seed=base_seed, field_kind=field_kind)


def offline_stress(stress_type: str, problem: int = 4, n_cases: int = 50, base_seed: int = 0,
                   field_kind: str = "adversarial", hunter_kwargs: Optional[dict] = None,
                   feed_scan_points: bool = True):
    """
    用 offline_sim.harness.evaluate_stress 在【对抗案例】上复核。
    feed_scan_points=True 时把我方扫描点喂给生成器，让它专门在覆盖缝隙布源——
    这是对“覆盖完备性”最狠的检验。返回其 Metrics 或 None。
    """
    if not _ensure_offline_on_path():
        return None
    from offline_sim import harness
    from .agent import make_runner_policy
    from .coverage import omni_scan_points, directional_scan_points
    policy = make_runner_policy(problem, **(hunter_kwargs or {}))
    sp = None
    if feed_scan_points:
        sp = directional_scan_points() if problem == 4 else omni_scan_points()
    return harness.evaluate_stress(policy, stress_type, n_cases=n_cases, problem=problem,
                                   base_seed=base_seed, field_kind=field_kind, scan_points=sp)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="jammerhunt 蒙特卡洛评测")
    ap.add_argument("--problem", type=int, default=3, choices=(3, 4))
    ap.add_argument("-n", "--n", type=int, default=200, help="案例数")
    ap.add_argument("--seed", type=int, default=1000, help="基准种子")
    ap.add_argument("--field", default="smooth", choices=("smooth", "constant"),
                    help="本包误差场类型（constant=最坏 ±1° 常数场）")
    ap.add_argument("--offline", action="store_true", help="额外用 offline_sim 复核")
    ap.add_argument("--stress", default=None, help="offline_sim 对抗案例类型（隐含 --offline）")
    args = ap.parse_args(argv)

    print(f"== 本包 LocalWorld 评测：problem={args.problem}  n={args.n}  field={args.field} ==")
    summ = monte_carlo(n=args.n, problem=args.problem, base_seed=args.seed,
                       field_kind=args.field)
    print(summ.report())
    bad = [r for r in summ.results if not r.success]
    if bad:
        print(f"\n未全清的局（前 10）：")
        for r in bad[:10]:
            print(f"  seed={r.seed} cleared={r.cleared}/{r.total} "
                  f"omni={r.n_omni} dir={r.n_dir} reason={r.finish_reason} err={r.error}")

    if args.offline or args.stress:
        print("\n== offline_sim 复核 ==")
        if args.stress:
            m = offline_stress(args.stress, problem=args.problem, n_cases=min(args.n, 50),
                               base_seed=args.seed)
        else:
            m = offline_evaluate(problem=args.problem, n_cases=args.n, base_seed=args.seed)
        if m is None:
            print("  （无法导入 offline_sim，跳过）")
        else:
            print(m.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
