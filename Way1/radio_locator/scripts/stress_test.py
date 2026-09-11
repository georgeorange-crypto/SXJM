"""
压力测试：多局随机布局批量跑，统计完成率、误清率、时间分布、不变量违规。
用法：python scripts/stress_test.py --n 50 --problem q3
"""
import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.metrics import compute_metrics
from src.coverage.q3_hex_cover import hex_cover
from src.coverage.q4_triangular_mesh import q4_mesh_cover
from src.domain.config import load_config
from src.domain.types import ChannelStatus
from src.domain.world_state import WorldState
from src.runtime.controller import Controller, ControllerConfig
from src.simulator.mock_simulator import MockSimulator, random_case, make_smooth_field


def run_one(seed, problem, robot, cfg, q4):
    case = random_case(seed=seed, problem=problem, q4=q4, error_field=make_smooth_field(seed))
    sim = MockSimulator(case, problem, robot)
    world = WorldState(problem=problem, robot=robot)
    if q4:
        plan = q4_mesh_cover(problem.area_radius, edge=cfg.get("q4_coverage", "triangle_edge", default=950.0))
        mode, coarse, fine, steps = "q4", 60.0, 25.0, 4000
    else:
        plan = hex_cover(problem.area_radius, r_safe=cfg.get("q3_coverage", "safe_receive_radius", default=980.0))
        mode, coarse, fine, steps = "q3", 50.0, 20.0, 3000
    ctrl = Controller(sim, world, plan.points,
                      ControllerConfig(problem_mode=mode, coarse_size=coarse, fine_size=fine,
                                       max_steps=steps, lam=1.0))
    ctrl.run()
    metrics = compute_metrics(ctrl.log, world, robot)
    n_true = len(case.jammers)
    false_clear = any(cs.status == ChannelStatus.CLEARED and
                      not (case.jammer_on(c) and case.jammer_on(c).cleared)
                      for c, cs in world.channels.items())
    # 误判缺失（同样严重）：某频道被判 ABSENT，但该频道确有真源 → 真源被漏。
    false_absent = [c for c, cs in world.channels.items()
                    if cs.status == ChannelStatus.ABSENT and case.jammer_on(c) is not None]
    # 漏源总判：真源既未清除也未在完成态被承认（present/cleared）。
    missed = [j.channel for j in case.jammers
              if world.channels[j.channel].status in (ChannelStatus.ABSENT, ChannelStatus.UNKNOWN)]
    return {
        "seed": seed, "n_true": n_true, "cleared": world.cleared_count(),
        "time": world.virtual_time, "complete": world.mission_complete(),
        "false_clear": false_clear, "false_absent": false_absent,
        "missed": missed, "ledger_ok": metrics.ledger_consistent,
        "violations": [v.code for v in ctrl.monitor.violations],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--problem", choices=["q3", "q4"], default="q3")
    ap.add_argument("--start-seed", type=int, default=1000)
    args = ap.parse_args()

    cfg = load_config(args.problem)
    problem, robot = cfg.problem, cfg.robot
    q4 = args.problem == "q4"

    results = []
    t0 = time.time()
    for i in range(args.n):
        seed = args.start_seed + i
        r = run_one(seed, problem, robot, cfg, q4)
        results.append(r)
        ok_row = (r["complete"] and not r["false_clear"] and not r["false_absent"]
                  and r["ledger_ok"] and not r["violations"])
        flag = "OK" if ok_row else "!!"
        print(f"[{flag}] seed={seed:5d} true={r['n_true']:2d} cleared={r['cleared']:2d} "
              f"t={r['time']:8.1f} complete={r['complete']} false_clear={r['false_clear']} "
              f"false_absent={r['false_absent']} viol={r['violations']}")
    dt = time.time() - t0

    n = len(results)
    n_complete = sum(1 for r in results if r["complete"])
    n_false = sum(1 for r in results if r["false_clear"])
    n_false_absent = sum(1 for r in results if r["false_absent"])
    n_missed = sum(1 for r in results if r["missed"])
    n_ledger_bad = sum(1 for r in results if not r["ledger_ok"])
    n_viol = sum(1 for r in results if r["violations"])
    times = [r["time"] for r in results if r["complete"]]
    print("\n========== 汇总 ==========")
    print(f"局数           : {n}   用时 {dt:.1f}s")
    print(f"任务完成       : {n_complete}/{n}  ({100*n_complete/n:.0f}%)")
    print(f"误清（严重）   : {n_false}")
    print(f"误判缺失（严重）: {n_false_absent}")
    print(f"漏源局         : {n_missed}")
    print(f"账本不一致     : {n_ledger_bad}")
    print(f"不变量违规局   : {n_viol}")
    if times:
        print(f"完成局时间     : 均值 {statistics.mean(times):.0f}s  "
              f"中位 {statistics.median(times):.0f}s  最大 {max(times):.0f}s")
    ok = (n_false == 0 and n_false_absent == 0 and n_ledger_bad == 0 and n_viol == 0)
    print(f"正确性总判      : {'通过' if ok else '存在问题'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
