# Way4 冻结方案 checklist（实时审计）

更新时间：2026-09-12

## 已完成且有当前证据

- [x] GPU/CUDA 环境可用：`torch 2.1.2+cu121`、CUDA 12.1、RTX 4060、CUDA 矩阵运算成功。
- [x] Candidate-PPO 网络、GAE、PPO trainer、padding/mask、checkpoint 已接入。
- [x] P4 训练集与 validation/test/stress seed 已隔离，并有不相交断言。
- [x] P4 validation：`candidate_ppo_p4_large.pt` 在 2040–2049 全部 full-clear。
- [x] P4 Math validation：2040–2049 全部 full-clear，并已生成 same-seed paired 汇总。
- [x] Same-seed validation paired：10/10 均有配对记录，full-clear 前提满足；性能差异已保留，未宣称 PPO 胜出。
- [x] Candidate-PPO CUDA scaling：small/medium/large 均完成 forward 基准，结果已打印并记录此前基准文件。
- [x] PPO 温度采样的 `old_log_prob` 已修正为与实际采样分布一致。
- [x] 编译验证通过。
- [x] RL trainer 定向测试：7 passed。
- [x] 评测协议/指标定向测试：15 passed，1 skipped（保留跳过原因）。

## 正在运行/尚未闭环

- [x] 独立 test gate 2050–2059：2050–2059 均已完成且 full-clear（2053–2059 新增 7/7，0 error）。
- [ ] full test same-seed paired Math vs PPO：脚本已修正为与 test gate 同口径 `max_steps=300`，并支持每个 seed/mode 原子增量写盘；首次重启在加载阶段因现有 checkpoint 均缺少当前 `way4-relational-candidate-ppo-v2` schema 而失败（`ValueError: checkpoint schema/problem mismatch`），后续兼容 checkpoint 生成尝试在当前并发负载下未留下产物。当前没有可计入的 paired 结果；重跑前必须先生成并验证兼容 checkpoint。
- [x] checkpoint 前置校验：新增 `scripts/validate_candidate_checkpoint.py`，对 schema、input_dim、problem、state_dict 逐项输出 JSON；旧 `candidate_ppo_p4_large.pt` 已实测报告 schema 缺失；3 个验证测试通过。
- [x] checkpoint 架构核验：对旧 `candidate_ppo_p4_large.pt` 执行当前 `CandidateActorCritic(210)` 的 strict `state_dict` 加载，确认缺少 relational Transformer/GNN 参数；不能仅补写 metadata，必须重新训练当前架构 checkpoint。
- [x] paired 启动失败落盘：`scripts/paired_p4_eval.py` 现在在 checkpoint 加载失败时原子写出 `complete=false`；`results/paired_startup_failure.json` 已实测记录 `ValueError: checkpoint schema/problem mismatch`。
- [x] 训练中断可审计：`scripts/train_candidate_ppo.py` 现在每个 episode 后原子写入当前 relational-v2 checkpoint，并记录 `complete=false`、已完成 seeds 与 transition 数；最终写入 `complete=true`。已提取 `save_candidate_checkpoint()` 并以临时模型测试 partial/complete 元数据与临时文件清理（1 passed）。
- [ ] 8-cell 论文消融：合并报告 [ablation_full_2000_2002_merged.json](ablation_full_2000_2002_merged.json) 已覆盖全部 8 格；3 格为 3/3 success/full-clear，4 格为 seed 2000 单次 success/full-clear，`minus_no_signal` 为真实失败（7/20、`no_progress_stall`）。跨场景/多 seed 统计仍未完成。
- [ ] Lower Bound / Oracle / Hybrid 与 `R_LB`：尚未形成 P4 全套真实结果。
- [x] 几何 outcome predictor：真实 feasible-region sampled geometry 已接入主预测路径；相关 planner/outcome 测试已通过。
- [x] Soft hypothesis 特征路径：`CandidateGenerator.generate_spatial_candidates()` 已将 `HypothesisLayer.elimination_gain()` 与 `alive_fraction()` 写入 `CandidateChannelFeatures.visibility_gain`/`directional_entropy`；新增 `tests/test_soft_hypothesis_e2e.py`，真实候选生成路径通过（1 passed）。完整 PPO rollout/训练效果仍另行保持未完成。
- [ ] 多 seed 大规模训练：已有 10-seed checkpoint，但训练更新策略与正式统计/超参选择审计仍需补齐。
- [ ] 最终发布报告：必须等 test、paired、消融、LB/Oracle/Hybrid 和统计证据齐全后生成。

## 口径

截至 2026-09-13，test seeds 2050–2059 均已分别验证为 100% full-clear；2059 汇总记录见 `results/p4_test_gate_2053_2059.json`，2050–2052 见各自单 seed artifact。

未实际运行或没有可检查 artifact 的条目保持 `[ ]`；smoke、dry-run、unsupported 配置和未完成进程不计为完成。

本轮回归：checkpoint 校验/增量保存、soft hypothesis E2E、消融协议、task pool、FutureCost、Pareto、service/synergy 共 `19 passed, 1 warning`。
