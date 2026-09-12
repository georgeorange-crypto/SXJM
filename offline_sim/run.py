"""
命令行启动器：启动本地离线模拟器，监听 http://127.0.0.1:2026，供机器狗程序连接。

用法示例：
  python -m offline_sim.run --robot-id 12345 --problem 3 --seed 42
  python -m offline_sim.run --robot-id 12345 --problem 4 --port 2026
  python -m offline_sim.run --case case.json           # 从固定案例文件加载复现

退出：Ctrl+C。演练模式下退出时打印案例真值（total / omni / dir / 各源）。
"""

from __future__ import annotations

import argparse
import sys

from .case import Case, generate_case
from .server import SimSession, SimServer


def main(argv=None):
    ap = argparse.ArgumentParser(description="无线电干扰源离线模拟器")
    ap.add_argument("--robot-id", required=True, help="参赛队号（robot_id 必须逐字节匹配）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2026)
    ap.add_argument("--problem", type=int, default=3, choices=(3, 4),
                    help="3=全为全向；4=全向+定向混合")
    ap.add_argument("--seed", type=int, default=None, help="随机种子（复现同一案例）")
    ap.add_argument("--n-jammers", type=int, default=None, help="干扰源个数 10..16（默认随机）")
    ap.add_argument("--n-dir", type=int, default=None, help="定向源个数（仅 problem=4）")
    ap.add_argument("--hard", action="store_true", help="P4 最难基准：全部干扰源均为定向源")
    ap.add_argument("--mode", default="practice", choices=("practice", "formal"),
                    help="practice 演练（退出揭示真值）/ formal 正式")
    ap.add_argument("--field", default="smooth",
                    choices=("smooth", "iid", "biased", "adversarial", "piecewise"),
                    help="示向度误差场类型（见 fields.py）")
    ap.add_argument("--case", default=None, help="从 JSON 案例文件加载（忽略随机参数）")
    ap.add_argument("--verbose", action="store_true", help="打印 HTTP 访问日志")
    args = ap.parse_args(argv)

    if args.case:
        with open(args.case, "r", encoding="utf-8") as f:
            case = Case.from_json(f.read())
    else:
        case = generate_case(
            seed=args.seed, problem=args.problem,
            n_jammers=args.n_jammers, n_directional=args.n_dir, mode=args.mode,
            hard=args.hard,
            field_kind=args.field,
        )

    session = SimSession(case, robot_id=args.robot_id)
    server = SimServer(session, host=args.host, port=args.port, verbose=args.verbose)

    print(f"[offline_sim] 监听 http://{args.host}:{args.port}")
    print(f"[offline_sim] robot_id={args.robot_id}  problem={args.problem}  mode={case.mode}")
    print(f"[offline_sim] 干扰源总数={case.total}（omni={case.n_omni}, dir={case.n_dir}）"
          + ("  << 正式模式下机器狗不可知此真值" if case.mode == "formal" else ""))
    print("[offline_sim] 机器狗程序把 BASE_URL 指向上面地址即可。Ctrl+C 结束。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[offline_sim] 收到中断，关闭。")
    finally:
        server.shutdown()
        server.server_close()
        eng = session.engine
        print(f"[offline_sim] 结束原因={eng.st.finish_reason}  "
              f"虚拟时刻={eng.st.virtual_time_s:.6f}s  已清除={case.cleared_count}/{case.total}")
        if case.mode == "practice":
            rev = case.reveal()
            print(f"[offline_sim] 案例真值：total={rev['total']} omni={rev['n_omni']} dir={rev['n_dir']}")
            for j in rev["jammers"]:
                d = "omni" if j["kind"] == "omni" else f"dir@{j['direction_deg']:.1f}°"
                mark = "✓已清" if j["cleared"] else "×未清"
                print(f"    ch{j['channel']:>2}  ({j['x']:8.1f},{j['y']:8.1f})  "
                      f"R_eff={j['r_eff']:.0f}  {d}  {mark}")


if __name__ == "__main__":
    sys.exit(main())
