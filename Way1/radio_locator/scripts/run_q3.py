"""
Q3 全流程：在 mock 上用覆盖布点 + 集员定位 + 调度清除，跑完一局并输出指标。
用法：python scripts/run_q3.py [--seed N] [--plot]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.metrics import compute_metrics
from src.coverage.q3_hex_cover import hex_cover
from src.domain.config import load_config
from src.domain.types import ChannelStatus
from src.domain.world_state import WorldState
from src.runtime.controller import Controller, ControllerConfig
from src.simulator.mock_simulator import MockSimulator, random_case, make_smooth_field


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--coarse", type=float, default=40.0)
    ap.add_argument("--fine", type=float, default=15.0)
    ap.add_argument("--max-steps", type=int, default=3000)
    args = ap.parse_args()

    cfg = load_config("q3")
    problem, robot = cfg.problem, cfg.robot

    case = random_case(seed=args.seed, problem=problem, q4=False,
                       error_field=make_smooth_field(args.seed))
    sim = MockSimulator(case, problem, robot)
    world = WorldState(problem=problem, robot=robot)

    r_safe = cfg.get("q3_coverage", "safe_receive_radius", default=980.0)
    m = cfg.get("q3_coverage", "outer_points", default=6)
    plan = hex_cover(problem.area_radius, r_safe=r_safe, m=m)

    ctrl = Controller(sim, world, plan.points,
                      ControllerConfig(problem_mode="q3", coarse_size=args.coarse,
                                       fine_size=args.fine, max_steps=args.max_steps,
                                       lam=cfg.get("planner", "lookahead_lambda", default=1.0)))
    ctrl.run()

    metrics = compute_metrics(ctrl.log, world, robot)
    n_true = len(case.jammers)
    print(f"=== Q3  seed={args.seed} ===")
    print(f"真源数         : {n_true}")
    print(f"已清除         : {world.cleared_count()}")
    print(f"判定无源       : {world.confirmed_absent()}")
    print(f"总虚拟时间     : {world.virtual_time:.1f} s")
    print(f"  移动         : {metrics.breakdown.move:.1f} s")
    print(f"  切频         : {metrics.breakdown.switch:.1f} s")
    print(f"  探测(5s×{metrics.n_measure}): {metrics.breakdown.detection:.1f} s")
    print(f"  清除命中     : {metrics.breakdown.clear_hit:.1f} s ({metrics.n_clear_success} 次)")
    print(f"  清除未命中   : {metrics.breakdown.clear_miss:.1f} s ({metrics.n_clear_miss} 次)")
    print(f"账本一致       : {metrics.ledger_consistent}")
    print(f"不变量违规     : {[v.code for v in ctrl.monitor.violations] or '无'}")
    print(f"任务完成       : {world.mission_complete()}")

    # 交叉校验：所有 CLEARED 均真实命中
    bad = [c for c, cs in world.channels.items()
           if cs.status == ChannelStatus.CLEARED and not (case.jammer_on(c) and case.jammer_on(c).cleared)]
    print(f"误清频道       : {bad or '无'}")

    if args.plot:
        from src.analysis.plots import plot_trajectory, plot_cover
        out = os.path.join(os.path.dirname(__file__), "..", "out")
        os.makedirs(out, exist_ok=True)
        plot_cover(plan.points, problem.area_radius, r_safe, os.path.join(out, "q3_cover.png"))
        plot_trajectory(ctrl.log, problem.area_radius, os.path.join(out, "q3_traj.png"))
        print(f"图件输出到 {out}")


if __name__ == "__main__":
    main()
