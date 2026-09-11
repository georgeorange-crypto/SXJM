"""
CLI：跑四层评价体系并出表/图。

  python -m eval_suite --problem 3 -n 100                    全方法梯队 + 榜单 + 分解表 + 主图
  python -m eval_suite --problem 4 -n 60 --methods ours,oracle,lb
  python -m eval_suite --directional -n 80                   Q3↔Q4 定向性代价（配对案例）
  python -m eval_suite --problem 3 -n 50 --no-fig            跳过画图

结果表打印到 stdout；主图存到 evaluate/out/figs/。
"""

from __future__ import annotations

import argparse
import os

from . import report
from .directional import directional_cost, format_directional
from .runner import ALL_METHODS, run_all


def _out_dir() -> str:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # evaluate/
    d = os.path.join(here, "out", "figs")
    os.makedirs(d, exist_ok=True)
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="四层评价体系（绝对下界/Oracle/基线/OURS/消融）")
    ap.add_argument("--problem", type=int, default=3, choices=(3, 4))
    ap.add_argument("-n", "--n", type=int, default=100, help="案例数")
    ap.add_argument("--seed", type=int, default=1000, help="基准种子")
    ap.add_argument("--methods", default=None,
                    help="逗号分隔的方法子集；默认全部：" + ",".join(ALL_METHODS))
    ap.add_argument("--directional", action="store_true",
                    help="改跑 Q3↔Q4 定向性代价（配对案例，用 --methods 的第一个方法，默认 ours）")
    ap.add_argument("--no-fig", action="store_true", help="不画主图")
    args = ap.parse_args(argv)

    if args.directional:
        method = (args.methods.split(",")[0] if args.methods else "ours")
        print(f"== 定向性代价（Q3 全向 ↔ Q4 部分定向配对）  method={method}  n={args.n} ==")
        _, summ = directional_cost(n=args.n, base_seed=args.seed, method=method)
        print(format_directional(summ))
        return 0

    methods = args.methods.split(",") if args.methods else ALL_METHODS
    print(f"== 四层评价：problem={args.problem}  n={args.n}  methods={methods} ==\n")
    summaries = run_all(methods=methods, n=args.n, problem=args.problem, base_seed=args.seed)

    print("---- 各方法明细 ----")
    for mth in methods:
        if mth in summaries:
            print(summaries[mth].report())
            print()

    print("---- 榜单（字典序：清除率↓, 成功局均值时间↑）----")
    print(report.leaderboard(summaries))
    print()
    print("---- 计时分解（成功局均值，秒）----")
    print(report.stacked_breakdown_table(summaries))

    if not args.no_fig:
        out = os.path.join(_out_dir(), f"breakdown_p{args.problem}.png")
        path = report.make_figure(summaries, out, problem=args.problem)
        print(f"\n主图: {path if path else '（matplotlib 不可用，已跳过）'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
