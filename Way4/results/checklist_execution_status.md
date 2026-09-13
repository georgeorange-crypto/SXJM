# Way4 checklist execution status

更新时间：2026-09-13。状态只依据当前工作树、测试输出和结果文件；`implemented` 不等于 `benchmark_accepted`。

## 已有代码与专项测试证据

| 阶段 | 范围 | 当前状态 | 证据 |
|---|---|---|---|
| B–E | 时间审计、progress、dense time reward、potential shaping、checkpoint gate | implemented + tested | `src/way4/pipeline.py`, `src/way4/rl/candidate_rollout.py`, `src/way4/rl/objectives.py`; 全量回归通过 |
| G–L | adaptive scan、batch stop、value-per-second、Pareto、2-opt、FOCUS、SpatialStop | implemented + tested | `src/way4/channels/scheduler.py`, `src/way4/planner/receding_horizon.py`, `src/way4/planner/spatial.py`, 专项 tests |
| M–V | task pool、route cleanup、beam successor、time debt、Feature v3、backbone、piggyback、候选语义 | implemented + tested | `src/way4/planner/`, `src/way4/rl/features.py`; 专项 tests |
| W–AB | route regret、endgame、watchdog、entropy/greedy、critic pooling、hierarchical PPO | implemented + tested | `src/way4/rl/`, `src/way4/planner/endgame.py`; 专项 tests |
| AC–AF | PPO rollout/update、hard-case mining、seed split、全局效率及 batch 指标 | implemented + smoke/tested | `scripts/train_candidate_ppo.py`, `src/way4/rl/training_protocol.py`; smoke 训练已执行 |

## 真实执行证据

| 实验 | 结果 |
|---|---|
| M0 vs M1，paired seeds 2000–2002 | 两者均 full-clear 20/20；M1 平均总时间 14451.740879 s，M0 为 15052.808616 s；M1 平均改善 3.99%，但 1/3 seed 回归 |
| M2，seeds 2000/2001 | 两者均 full-clear 20/20；结果与 M1 完全一致，未观察到 TSPN 增量 |
| M3，seeds 2000/2001 | 两者均 full-clear 20/20；相对 M1 平均慢 131.106995 s（0.95%） |
| M1 development set，seeds 2000–2009 | 10/10 full-clear；mean 748.808737、median 759.370146、P90 791.599517、max 792.325970 s/target；G1 pass，G2/G3/FINAL fail |
| 独立 soundness | 500/500 checks passed |
| Feature/PPO smoke | 2 episodes、2 transitions、1 update；截断验证被 gate 正确拒绝 |

## 尚不能标记最终完成

最新增量：M0 seed 2004 已完成并并入 `ablation_M0_seeds2000_2004_current.json`；5/5 full-clear，均无 illegal_clear 或 safety_violation。M0 审计结果为 mean 733.343、median 706.884、P90 813.655、max 816.584 s/target；G1–FINAL 均未通过（审计要求 certificate_sound=true，而本次未附 soundness 证据）。

1. M0–M3 中 M1 已完成 2000–2009 development set；M0/M2/M3 仍只有 2 seed，P0–P4 PPO cells 尚未完成正式多 seed/full-clear 性能矩阵。
2. PPO 尚未完成正式长训练、多 seed、独立 train/val/test 的性能证明。
3. 当前 M0–M3 平均约 688 s/target，只通过 G1/G2，未通过 G3（500）和 FINAL（400）。
4. 真实实验尚未满足 checklist 要求的完整 2000–2009 development regression set 与最终统计报告。

## 主要结果文件

- `ablation_M0_seeds2000_2001_current.json`
- `ablation_M1_seeds2000_2001_current.json`
- `ablation_M2_seeds2000_2001_current.json`
- `ablation_M3_seeds2000_2001_current.json`
- `paired_M0_M1_seeds2000_2001_current.json`
- `performance_gate_M1_seeds2000_2009_current.json`
- `performance_gate_M2_seeds2000_2001_current.json`
