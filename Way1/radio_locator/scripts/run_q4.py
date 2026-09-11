"""
Q4 全流程：源型未知（全向/定向混合），用三角网覆盖 + 组合可行集跑完一局。
用法：python scripts/run_q4.py [--seed N] [--plot]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.metrics import compute_metrics
from src.coverage.q4_triangular_mesh import q4_mesh_cover
from src.domain.config import load_config
from src.domain.types import ChannelStatus
from src.domain.world_state import WorldState
from src.runtime.controller import Controller, ControllerConfig
from src.simulator.mock_simulator import MockSimulator, random_case, make_smooth_field


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--coarse", type=float, default=50.0)
    ap.add_argument("--fine", type=float, default=20.0)
    ap.add_argument("--max-steps", type=int, default=4000)
    args = ap.parse_args()

    cfg = load_config("q4")
    problem, robot = cfg.problem, cfg.robot

    case = random_case(seed=args.seed, problem=problem, q4=True,
                       error_field=make_smooth_field(args.seed))
    sim = MockSimulator(case, problem, robot)
    world = WorldState(problem=problem, robot=robot)

    edge = cfg.get("q4_coverage", "triangle_edge", default=950.0)
    plan = q4_mesh_cover(problem.area_radius, edge=edge)

    ctrl = Controller(sim, world, plan.points,
                      ControllerConfig(problem_mode="q4", coarse_size=args.coarse,
                                       fine_size=args.fine, max_steps=args.max_steps,
                                       lam=cfg.get("planner", "lookahead_lambda", default=1.0)))
    ctrl.run()

    metrics = compute_metrics(ctrl.log, world, robot)
    n_true = len(case.jammers)
    n_dir = sum(1 for j in case.jammers if j.kind.value == "directional")
    print(f"=== Q4  seed={args.seed} ===")
    print(f"真源数         : {n_true}  (定向 {n_dir} / 全向 {n_true - n_dir})")
    print(f"已清除         : {world.cleared_count()}")
    print(f"判定无源       : {world.confirmed_absent()}")
    print(f"总虚拟时间     : {world.virtual_time:.1f} s")
    print(f"  移动         : {metrics.breakdown.move:.1f} s")
    print(f"  切频         : {metrics.breakdown.switch:.1f} s")
    print(f"  探测         : {metrics.breakdown.detection:.1f} s ({metrics.n_measure} 次)")
    print(f"  清除命中/未中: {metrics.breakdown.clear_hit:.1f}/{metrics.breakdown.clear_miss:.1f} s")
    print(f"账本一致       : {metrics.ledger_consistent}")
    print(f"不变量违规     : {[v.code for v in ctrl.monitor.violations] or '无'}")
    print(f"任务完成       : {world.mission_complete()}")

    bad = [c for c, cs in world.channels.items()
           if cs.status == ChannelStatus.CLEARED and not (case.jammer_on(c) and case.jammer_on(c).cleared)]
    print(f"误清频道       : {bad or '无'}")

    if args.plot:
        from src.analysis.plots import plot_trajectory, plot_cover
        out = os.path.join(os.path.dirname(__file__), "..", "out")
        os.makedirs(out, exist_ok=True)
        plot_cover(plan.points, problem.area_radius, edge / (3 ** 0.5),
                   os.path.join(out, "q4_cover.png"))
        plot_trajectory(ctrl.log, problem.area_radius, os.path.join(out, "q4_traj.png"))
        print(f"图件输出到 {out}")


if __name__ == "__main__":
    main()
