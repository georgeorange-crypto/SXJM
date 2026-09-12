# Way4 checklist acceptance status

更新时间：2026-09-13。权威源码：`SXJM/Way4/`。

## 已有当前证据

- P0 interface/routing regression：STOP schema、checkpoint schema guard、joint
  route wiring、WAIT task bookkeeping、viewpoint opportunities、Pareto ranking、
  route synergy、unified route pipeline 均有 targeted tests。
- Full Way4 regression：261 passed、12 skipped、0 failed。
- P4 test gate 2050–2059：10/10 full-clear；2053–2059 的新增 gate artifact 为
  7/7 full-clear、0 error。

## 实验状态

- Math vs Candidate-PPO test split 2050–2059 的 paired job 仍未完成；当前
  partial artifact 明确为 `complete=false`，Math 仅部分写入，PPO 尚无可用
  paired 结果。
- 已有 2040–2049 paired 结果：Math/PPO 均 10/10 full-clear；PPO mean
  16217.38s、Math mean 16232.24s，但 paired win rate 0.5、最大 regression
  3432.04s，不能宣称 PPO 稳定优于 Math。
- A0–A12 完整消融、Lower Bound/Oracle/Hybrid 全套真实结果、长训练统计与
  最终发布报告仍未完成。

## 验收规则

仅当 artifact 同时具备完整 seed 集、full-clear 统计、时间分位数、paired
delta/win-rate 和无错误记录，才将对应实验项从 `[~]` 改为 `[x]`。Smoke、
partial artifact、unsupported cell 和未完成进程不计为完成。
