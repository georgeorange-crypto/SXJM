"""
基准：测量各核心算法的耗时（几何 / 可行集重建 / 路径规划），供效率评估。
用法：python scripts/benchmark.py
"""
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.domain.config import load_config
from src.domain.observation import Observation
from src.domain.types import ObservationType
from src.geometry.mec import min_enclosing_circle
from src.optimization.held_karp import held_karp_open
from src.optimization.two_opt import solve_route
from src.set_estimation.q3_feasible_set import Q3FeasibleSet
from src.simulator.mock_simulator import MockCase, MockJammer, MockSimulator, make_smooth_field


def bench(name, fn, reps=1):
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    dt = (time.perf_counter() - t0) / reps
    print(f"{name:40s}: {dt*1000:8.2f} ms")
    return out


def main():
    cfg = load_config("q3")
    problem, robot = cfg.problem, cfg.robot
    rng = random.Random(0)

    # MEC
    pts = [(rng.uniform(-1800, 1800), rng.uniform(-1800, 1800)) for _ in range(500)]
    bench("MEC (500 pts)", lambda: min_enclosing_circle(pts), reps=20)

    # Held-Karp n=14
    nodes = [(rng.uniform(0, 1800), rng.uniform(0, 1800)) for _ in range(14)]
    bench("Held-Karp open TSP (n=14)", lambda: held_karp_open(nodes, 0), reps=3)

    # 2-opt n=60
    big = [(rng.uniform(0, 1800), rng.uniform(0, 1800)) for _ in range(60)]
    bench("2-opt route (n=60)", lambda: solve_route(big, 0), reps=3)

    # Q3 可行集重建（多观测）
    sx, sy = 300.0, -200.0
    case = MockCase(jammers=[MockJammer(1, sx, sy, 1200.0)], error_field=make_smooth_field(0))
    sim = MockSimulator(case, problem, robot)
    sim.enter()
    obs = []
    for k in range(6):
        ang = 2 * 3.14159 * k / 6
        px, py = sx + 700 * __import__("math").cos(ang), sy + 700 * __import__("math").sin(ang)
        r = sim.measure(px, py, 1)
        ot = r.to_observation_type()
        obs.append(Observation((px, py), 1, ot,
                               bearing_deg=r.svd_deg if ot == ObservationType.BEARING else None, step=k))

    fs = Q3FeasibleSet(problem=problem, robot=robot, coarse_size=20.0, fine_size=5.0)
    bench("Q3 rebuild coarse=20 (6 obs)", lambda: fs.rebuild(obs, min_size=20.0), reps=5)
    bench("Q3 rebuild fine=2 (6 obs)", lambda: fs.rebuild(obs, min_size=2.0), reps=2)
    print(f"\n最终 MEC 半径: {fs.mec_radius:.2f} m  叶盒数: {len(fs.leaf_boxes())}")


if __name__ == "__main__":
    main()
