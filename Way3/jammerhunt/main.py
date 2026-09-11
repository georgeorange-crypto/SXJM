"""
比赛入口：通过 HTTP+JSON 连真实模拟器，跑一局“定位并清除全部干扰源”。

用法（先启动模拟器，再运行本入口）：
    # 终端1：启动参考模拟器（或换成赛方给的地址）
    python -m offline_sim.run --robot-id TEAM1 --problem 3 --seed 42 --port 2026
    # 终端2：
    python -m jammerhunt.main --robot-id TEAM1 --problem 3 --host 127.0.0.1 --port 2026

第三问用全向 1-覆盖扫描点，第四问用定向 3-覆盖扫描点（见 coverage.py / 思路.md）。
策略看不到真值，只能报告“自认为”的结果：已清除数、各频道判定（有源已清 / 判无源）。
真实模拟器在 practice 模式 /exit 后才揭示真值，可另行核对。

仅依赖标准库 + 本包。
"""

from __future__ import annotations

import argparse
import sys

from .agent import Hunter
from .interface import HttpClient, HttpWorld


def build_world(robot_id: str, host: str, port: int, arena_id: str,
                url: str | None, timeout: float) -> HttpWorld:
    base_url = url if url else f"http://{host}:{port}"
    client = HttpClient(base_url, robot_id=robot_id, arena_id=arena_id, timeout=timeout)
    return HttpWorld(client)


def run(robot_id: str, problem: int, host: str = "127.0.0.1", port: int = 2026,
        arena_id: str = "default", url: str | None = None, timeout: float = 5.0,
        src_radius: float | None = None) -> Hunter:
    """连模拟器跑一局，返回运行后的 Hunter（含各频道状态）。"""
    world = build_world(robot_id, host, port, arena_id, url, timeout)
    kw = {} if src_radius is None else {"src_radius_m": src_radius}
    hunter = Hunter(problem=problem, **kw)
    hunter.run(world)
    return hunter


def _report(hunter: Hunter, world: HttpWorld) -> str:
    lines = [
        "== 运行结束（策略自报，非真值）==",
        f"problem            : {hunter.problem}",
        f"已清除频道数        : {hunter.cleared_count}",
        f"最终虚拟时间(自报)  : {world.virtual_time_s:.1f} s",
        f"接口是否已结束      : {world.finished}",
    ]
    cleared = sorted(c for c, s in hunter.chans.items() if s.cleared)
    present = sorted(c for c, s in hunter.chans.items() if s.detected and not s.cleared)
    absent = sorted(c for c, s in hunter.chans.items() if s.absent)
    lines.append(f"已清除频道          : {cleared}")
    if present:
        lines.append(f"有源但未清(告警!)   : {present}")
    lines.append(f"判定无源频道        : {absent}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="无线电干扰源快速自动定位与清除（第三/四问入口）")
    ap.add_argument("--robot-id", required=True, help="登录队号（= robot_id）")
    ap.add_argument("--problem", type=int, default=3, choices=(3, 4),
                    help="3=全向；4=定向+全向混合")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2026)
    ap.add_argument("--arena-id", default="default")
    ap.add_argument("--url", default=None, help="完整 base URL（给定则忽略 host/port）")
    ap.add_argument("--timeout", type=float, default=5.0, help="单次 HTTP 超时(s)")
    ap.add_argument("--src-radius", type=float, default=None,
                    help="第四问假定源半径上限(m)；默认 1800 全盘（最稳）")
    args = ap.parse_args(argv)

    world = build_world(args.robot_id, args.host, args.port, args.arena_id,
                        args.url, args.timeout)
    kw = {} if args.src_radius is None else {"src_radius_m": args.src_radius}
    hunter = Hunter(problem=args.problem, **kw)
    hunter.run(world)
    print(_report(hunter, world))
    # 若存在“有源但未清”的频道，以非零码退出，便于脚本发现异常
    unresolved = any(s.detected and not s.cleared for s in hunter.chans.values())
    return 1 if unresolved else 0


if __name__ == "__main__":
    sys.exit(main())
