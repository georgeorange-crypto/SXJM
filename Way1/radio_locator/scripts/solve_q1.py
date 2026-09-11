"""
Q1 演示脚本：给定若干示向度观测，用楔形三角交会定位单个源。
用法：python scripts/solve_q1.py   （用内置示例）
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.domain.config import load_config
from src.domain.observation import Observation
from src.domain.types import ObservationType
from src.localization.q1_solver import solve_q1
from src.simulator.mock_simulator import MockCase, MockJammer, MockSimulator, make_smooth_field


def main():
    cfg = load_config("common")
    problem, robot = cfg.problem, cfg.robot

    sx, sy, r_eff = 520.0, -280.0, 1250.0
    case = MockCase(jammers=[MockJammer(1, sx, sy, r_eff)], error_field=make_smooth_field(0))
    sim = MockSimulator(case, problem, robot)
    sim.enter()

    viewpoints = [(-300.0, 0.0), (500.0, -900.0), (1100.0, -200.0)]
    obs = []
    for k, (x, y) in enumerate(viewpoints):
        r = sim.measure(x, y, 1)
        ot = r.to_observation_type()
        obs.append(Observation((x, y), 1, ot,
                               bearing_deg=r.svd_deg if ot == ObservationType.BEARING else None, step=k))
        print(f"viewpoint {k}: ({x:.0f},{y:.0f}) -> {r.measure_result} svd={r.svd_deg}")

    res = solve_q1(obs, problem, robot)
    if res.is_empty():
        print("可行域为空（观测异常）")
        return
    print(f"\n真源     : ({sx:.1f}, {sy:.1f})")
    print(f"估计中心 : ({res.center[0]:.1f}, {res.center[1]:.1f})")
    print(f"定位半径 : {res.radius:.2f} m   直径: {res.diameter:.2f} m   面积: {res.area:.1f} m^2")
    err = math.hypot(sx - res.center[0], sy - res.center[1])
    print(f"中心误差 : {err:.2f} m   真源在可行域内: {err <= res.radius + 1e-6}")


if __name__ == "__main__":
    main()
