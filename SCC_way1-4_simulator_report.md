# SX Way1–Way4 模拟器测试与 evaluate 基线对比报告

## 1. 测试范围与口径

测试主机为 `dex`，实际项目目录为 `D:\\George\\SX`（用户给出的 `D:Georege/SX` 路径不存在，已按实际目录纠正）。测试使用仓库自带的离线模拟器/本地虚拟世界，不涉及正式盲测或真实 HTTP 竞赛服务。

统一优先级为：

1. 全清成功率；
2. 成功局虚拟时间；
3. 测量次数、移动距离及异常/停滞信息。

误差场主要使用 `smooth`；Way3/Way4 使用相同固定种子逐局对比。`evaluate` 的 P3 使用 10 局、P4 使用 8 局；Way3/Way4 对比使用 P3 的 1000–1009、P4 的 2000–2007。

## 2. Way1–Way3 结果

| 方法 | 场景 | 局数 | 全清 | 成功局虚拟时间均值 | 备注 |
|---|---:|---:|---:|---:|---|
| Way1 | 算法基准 | — | — | MEC 1.32 ms；Held–Karp(n=14) 100.99 ms；2-opt(n=60) 7.42 ms | 123 个 pytest 测试通过 |
| Way2 | P3 | 10 | 10/10 (100%) | 8905.0 s | greedy_math，平均 113.3 steps/局 |
| Way3 | P3 | 10 | 10/10 (100%) | 5180.6 s | P50 5172.6 s，P90 5798.6 s，最大 6442.8 s |
| Way3 | P4 | 10 | 10/10 (100%) | 10026.0 s | P50 10052.5 s，P90 10570.5 s，最大 10820.4 s |

Way2 P4 的 10 局默认评测在长时间无输出后中止，未将其当作完成结果；这与 Way2 P4 已有的长步数/停滞风险一致。

## 3. evaluate 基线

### P3（10 局）

| 方法 | 全清率 | 成功局均值 | R_LB 中位数 | 移动超支 ΔT_move | 信息超支 ΔT_info |
|---|---:|---:|---:|---:|---:|
| lb | 100% | 1765.6 s | 1.000 | 0 s | -119 s |
| oracle | 100% | 1945.7 s | 1.100 | 61 s | 0 s |
| ours / Way3 | 100% | 5180.6 s | 2.786 | 2590 s | 706 s |
| greedy_scan | 100% | 5226.5 s | 2.816 | 2635 s | 706 s |
| reactive | 100% | 20674.2 s | 12.546 | 18303 s | 487 s |

P3 中 `ours` 比 `greedy_scan` 快约 0.9%，比 `reactive` 快约 74.9%；`ours` 的总超支约 79% 来自额外移动。

### P4（8 局）

| 方法 | 全清率 | 成功局均值 | R_LB 中位数 | 移动超支 ΔT_move | 信息超支 ΔT_info |
|---|---:|---:|---:|---:|---:|
| lb | 100% | 1727.9 s | 1.000 | 0 s | -119 s |
| oracle | 100% | 1906.8 s | 1.108 | 60 s | 0 s |
| ours / Way3 | 100% | 9911.0 s | 5.619 | 5874 s | 2190 s |
| greedy_scan | 100% | 10113.7 s | 5.772 | 6077 s | 2190 s |
| reactive | 87.5% | 56369.6 s | 32.775 | 52846 s | 1683 s |

P4 中 `ours` 比 `greedy_scan` 快约 2.0%；`reactive` 出现 1 局未全清，且成功局平均时间约为 `ours` 的 5.7 倍。

## 4. Way4 与 Way3 同引擎对比

### P3：通过

- Way3：10/10 全清，均值 5180.6 s。
- Way4：10/10 全清，均值 4983.9 s。
- Way4/Way3 时间比 0.962，约快 3.8%。
- M7 全清率门禁：PASS。

### P4：不通过，存在明确回归

- Way3：8/8 全清，成功局均值 10022.6 s。
- Way4：3/8 全清，成功局均值 6493.3 s（仅对 3 个成功局计算）。
- Way4/Way3 成功交集时间比 0.668，但不能以此宣称更快，因为 Way4 漏清 5 个 Way3 已成功的种子。
- 回归种子：`2000, 2003, 2004, 2005, 2007`。
- `2004` 另报告 `no_progress_stall`。
- M7 全清率门禁：FAIL（Way4 37.5% < Way3 100%）。

## 5. 结论与后续建议

1. Way1 的几何/规划基础和测试闭环正常；Way2 P3 可运行但明显慢于 Way3，Way2 P4 存在长时间停滞风险。
2. Way3 在本次 P3/P4 固定种子上均保持 100% 全清，是当前可靠基线。
3. Way4 P3 已达到“不降低全清率且略有提速”的阶段性目标。
4. Way4 P4 尚未达到验收标准，当前首要任务不是继续优化时间，而是定位 `2000/2003/2004/2005/2007` 的漏清路径和 `no_progress_stall`，优先恢复 100% 全清，再重新比较时间。
5. `evaluate` 的 `ours`/Way3 结果与 Way4 管线结果必须分开报告：前者用于统一方法学基线，后者用于 Way4 回归门禁；两者不能直接当作同一实现的性能数字。

## 6. 可复现实验命令

```text
cd /d D:\\George\\SX
python Way1\\radio_locator\\scripts\\benchmark.py
cd Way1\\radio_locator && python -m pytest -q
cd ..\\..\\Way3 && python -m jammerhunt.mc --problem 3 -n 10 --seed 1000
python -m jammerhunt.mc --problem 4 -n 10 --seed 2000
cd ..\\evaluate && python -m eval_suite --problem 3 -n 10 --methods lb,oracle,reactive,greedy_scan,ours --no-fig
python -m eval_suite --problem 4 -n 8 --methods lb,oracle,reactive,greedy_scan,ours --no-fig
cd ..
python Way4\\scripts\\compare_way3_way4.py --problem 3 -n 10 --seed 1000 --max-steps 3000
python Way4\\scripts\\compare_way3_way4.py --problem 4 -n 8 --seed 2000 --max-steps 3000
```
