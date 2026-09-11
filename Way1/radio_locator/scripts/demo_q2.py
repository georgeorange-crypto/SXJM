"""
Q2 演示：给定首次观测后的可行带，用 Minimax 选择第二观测点，并打印各候选评估。
用法：python scripts/demo_q2.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.domain.config import load_config
from src.localization.second_viewpoint import choose_second_viewpoint


def main():
    cfg = load_config("common")
    problem, robot = cfg.problem, cfg.robot

    # 首测在 (0,-1000) 得到指向北的楔形，可行带近似沿 y 轴一列点
    band = [(0.0, y) for y in range(200, 1400, 100)]
    dec = choose_second_viewpoint(band, problem, robot, from_pos=(0.0, -1000.0),
                                  first_viewpoint=(0.0, -1000.0), lam=1.0)

    print("Minimax 第二观测点选择")
    print(f"候选数: {len(dec.all_evals)}")
    print(f"最优点: ({dec.best[0]:.1f}, {dec.best[1]:.1f})  "
          f"最坏残差: {dec.evaluation.worst_residual:.1f} m  "
          f"最坏结果: {dec.evaluation.worst_outcome}  score={dec.evaluation.score:.2f}")
    print("\n前 8 个候选（按 score 升序）:")
    for e in sorted(dec.all_evals, key=lambda e: e.score)[:8]:
        print(f"  ({e.location[0]:7.1f},{e.location[1]:7.1f})  residual={e.worst_residual:7.1f}  "
              f"move={e.move_cost:6.1f}  worst={e.worst_outcome:11s}  score={e.score:7.2f}")


if __name__ == "__main__":
    main()
