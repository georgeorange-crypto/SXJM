# 时间优化清单：训练路径增量验收

2026-09-13 | M0 baseline seed 2003: full-clear 20/20, illegal_clear=0, safety_violation=0, 14137.672165 s, 112 macros. Four-seed M0/M1 paired mean: M0 14824.024503 s vs M1 14397.136073 s; M1 delta -426.888430 s (-2.88%), with 2/4 regressions. Artifact: `results/paired_M0_M1_seeds2000_2003_current.json`.
2026-09-13 | M0 baseline extended to seed 2002: full-clear 20/20, illegal_clear=0, safety_violation=0, 16185.244684 s, 91 macros. Three-seed M0/M1 paired comparison: mean M0 15052.808616 s vs M1 14451.740879 s; M1 delta -601.067736 s (-3.99%), with 1/3 regressions. Artifact: `results/paired_M0_M1_seeds2000_2002_current.json`.
2026-09-13 | M1 development regression complete for seeds 2000–2009: all 10 full-clear 20/20, illegal_clear=0, safety_violation=0. Aggregate mean 748.808737, median 759.370146, P90 791.599517, max 792.325970 s/target; G1 passes, G2/G3/FINAL fail. Artifact: `results/performance_gate_M1_seeds2000_2009_current.json`.
2026-09-13 | M1 development regression seed 2008: full-clear 20/20, illegal_clear=0, safety_violation=0, 15027.447166 s (751.372358 s/target), 121 macros. Aggregate seeds 2000–2008: full-clear 100%, mean 745.951926, median 755.860851, P90 791.680234, max 792.325970 s/target; G1 pass, G2/G3/FINAL fail. Artifact: `results/performance_gate_M1_seeds2000_2008_current.json`.
2026-09-13 | M1 development regression seed 2007: full-clear 20/20, illegal_clear=0, safety_violation=0, 15434.030014 s (771.701501 s/target), 101 macros. Aggregate seeds 2000–2007: full-clear 100%, mean 745.274372, median 759.370146, P90 791.760951, max 792.325970 s/target; G1 pass, G2/G3/FINAL fail. Artifact: `results/performance_gate_M1_seeds2000_2007_current.json`.
2026-09-13 | M1 development regression seed 2006: full-clear 20/20, illegal_clear=0, safety_violation=0, 15257.588814 s (762.879441 s/target), 134 macros. Aggregate seeds 2000–2006: full-clear 100%, mean 741.499068, median 755.860851, P90 791.841668, max 792.325970 s/target; G1 pass, G2/G3/FINAL fail. Artifact: `results/performance_gate_M1_seeds2000_2006_current.json`.
2026-09-13 | M1 development regression seed 2005: full-clear 20/20, illegal_clear=0, safety_violation=0, 15117.217012 s (755.860851 s/target), 141 macros. Aggregate seeds 2000–2005: full-clear 100%, mean 737.935673, median 733.763467, P90 791.922385, max 792.325970 s/target; G1 pass, G2/G3/FINAL fail. Artifact: `results/performance_gate_M1_seeds2000_2005_current.json`.
2026-09-13 | M1 development regression seed 2004: full-clear 20/20, illegal_clear=0, safety_violation=0, 15846.519400 s (792.325970 s/target), 129 macros. Aggregate seeds 2000–2004: full-clear 100%, mean 734.350637, median 711.666083, P90 792.003102, max 792.325970 s/target; G1 pass, G2/G3/FINAL fail. Artifact: `results/performance_gate_M1_seeds2000_2004_current.json`.
2026-09-13 | M1 development regression seed 2003: full-clear 20/20, illegal_clear=0, safety_violation=0, 14233.321654 s (711.666083 s/target), 105 macros. Aggregate seeds 2000–2003: full-clear 100%, mean 719.856804, median 706.422306, P90 767.562985, max 791.518800 s/target; G1 pass, G2/G3/FINAL fail. Artifact: `results/performance_gate_M1_seeds2000_2003_current.json`.
2026-09-13 | M1 development regression seed 2002: full-clear 20/20, illegal_clear=0, safety_violation=0, 15830.375994 s (791.518800 s/target), 137 macros. Three-seed aggregate (2000–2002): full-clear rate 100%, mean 722.587044 s/target, median 701.178530, P90 773.450746, max 791.518800; G1 passes, G2/G3/FINAL fail. Artifact: `results/performance_gate_M1_seeds2000_2002_current.json`.
2026-09-13 | AI Gate statistics: `audit_performance_gates.py` now emits mean/median/P90/max seconds per target (inclusive quantile; max for n<2), while preserving full-clear and correctness gates. M1 two-seed audit rerun: mean/median 688.121166 s, P90 698.567057 s, max 701.178530 s; tests 2 passed.
2026-09-13 | E03/AC validation execution: `validate_candidate_ppo.py` ran smoke checkpoint on seeds 2040,2041 with max_steps=1; both rows emitted illegal_clear=0/safety_violation=0 and greedy=true, but full_clear_rate=0 due intentionally truncated episodes, so unified validation gate rejected with `full_clear_failed`. This confirms rejection behavior, not model performance.
2026-09-13 | M2 two-seed completion: seeds 2000/2001 both full-clear 20/20 with illegal_clear=0 and safety_violation=0; times are exactly M1's 13501.276052 s and 14023.570592 s. Mean 688.121166 s/target; G1/G2 pass, G3/FINAL fail. TSPN switch has no observed increment on these two cases. Artifact: `results/performance_gate_M2_seeds2000_2001_current.json`.
2026-09-13 | M3 two-seed comparison: seed 2001 full-clear 20/20, 13989.169558 s, 74 macros, zero illegal/safety violations. Against M1 on seeds 2000/2001, M3 mean 13893.530317 s vs M1 13762.423322 s: +131.106995 s (+0.95%), with 1/2 regressions. Backbone-only remains correctness-safe but not accepted as time improvement. Artifact: `results/paired_M1_M3_seeds2000_2001_current.json`.
2026-09-13 | M3 real run: adaptive scan + TSPN + backbone-only coverage, seed 2000, full-clear 20/20, illegal_clear=0, safety_violation=0, 13797.891076 s / 75 macros. Against M1 same seed, M3 is +296.615024 s slower despite fewer macros (75 vs 109), so backbone-only is safe on this case but not time-superior. Artifact: `results/paired_M1_M3_seed2000_current.json`.
2026-09-13 | M2 real run: adaptive scan + TSPN, seed 2000, full-clear 20/20, illegal_clear=0, safety_violation=0, 13501.276052 s / 109 macros. Same-seed M1 comparison is exactly equal (delta 0), so this routing switch is wired and safe but shows no incremental effect on this case. Artifact: `results/paired_M1_M2_seed2000_current.json`.
2026-09-13 | M0/M1 two-seed paired comparison complete: both configurations full-clear on seeds 2000/2001 with zero illegal/safety violations. M0 mean total time 14486.590582 s; M1 13762.423322 s; mean delta M1-M0 = -724.167260 s (-5.00%). Seed 2000 improves by 2830.409 s, seed 2001 regresses by 1382.074 s; 1/2 severe directional regressions, so adaptive scan is promising but not accepted as universally better. Artifact: `results/paired_M0_M1_seeds2000_2001_current.json`.
2026-09-13 | M1 two-seed completion: seeds 2000/2001 both full-clear 20/20, illegal_clear=0, safety_violation=0; times 13501.276052 s and 14023.570592 s, mean 13762.423322 s total = 688.121166 s/target. Unified audit: G1/G2 passed, G3/FINAL failed on time. Artifacts: `results/ablation_M1_seeds2000_2001_current.json`, `results/performance_gate_M1_seeds2000_2001_current.json`.
2026-09-13 | AH paired result: M0 (adaptive_scan=False) vs M1 (adaptive_scan=True), same seed 2000; both full-clear 20/20 with zero illegal/safety violations. M1 reduces virtual time from 16331.685016 s to 13501.276052 s, delta -2830.408964 s (-17.33%), and macros 117 to 109. Artifact: `results/paired_M0_M1_seed2000_current.json`. Single-seed paired evidence only.
2026-09-13 | AI Gate re-run: M1 seed 2000 regenerated with `illegal_clear=0`, `safety_violation=0`; paired with independent negative-soundness v2 (`all_passed=true`, 500 checks), audit marks G1 and G2 passed (675.064 s/target), G3 and FINAL failed on time threshold. This is one seed, not the required multi-seed final claim.
2026-09-13 | AI gate evidence wiring: deterministic ablation rows now export `illegal_clear` and `safety_violation`; gate auditor accepts these only when both are zero and separately requires an independent negative-soundness audit (`--soundness`, current v2: 500/500 checks passed). Tests 4 passed. Existing M1 artifact predates the new fields and remains `insufficient_evidence` until rerun.
2026-09-13 | AH/AI real run: recommended cell M1 (adaptive scan), seed 2000, max_steps=3000 completed with `full_clear=true`, `resolved=20/20`, 109 macros, virtual time 13501.276052 s (675.063803 s/target). Performance Gate audit remains rejected because the row lacks independent clear-correct/certificate-sound evidence; no correctness fields were fabricated.
2026-09-13 | AI performance-gate audit: added `scripts/audit_performance_gates.py`, which reads real ablation rows, emits G1/G2/G3/FINAL only when all rows are full-clear and independently marked `clear_correct`/`certificate_sound`, and otherwise returns `insufficient_evidence`. Unit tests 2 passed; current five-cell probe is correctly rejected as insufficient evidence. Artifact: `results/performance_gate_audit_current.json`.
2026-09-13 | AH ablation runner repair verification: previously stale audit rows marked several deterministic cells unsupported; current runner executes `routing_nearest`, `nbv_greedy`, `coverage_backbone_only`, `minus_no_signal`, and `minus_cardinality` on seed 2000 (max_steps=1), preserving incomplete outcomes rather than fabricating success. Artifact: `results/ablation_cells_probe_current.json`; this is wiring/smoke evidence, not a performance comparison.
2026-09-13 | F01/F02 credit-horizon contract rerun: `scripts/ppo_credit_ablation.py` generated 9 gamma/lambda rows and `tests/test_ppo_credit_ablation.py` plus CLI flag tests passed (2 passed). This validates parameter plumbing/GAE arithmetic only; environment-level multi-seed comparison remains pending.
2026-09-13 | N02/O04 Feature v3 regression: `evaluation_features_v3` now has an explicit test proving the frozen base schema is preserved and time-debt/future-route blocks are finite and appended; `tests/test_service_features.py` 2 passed. This is API/contract evidence, not a trained-policy performance result.
本记录仅覆盖已经执行的训练工程修改，不代表总清单完成。

| 项目 | 当前证据 | 未完成部分 |
|---|---|---|
| C01/C02 | `timed_policy_transitions` 对每个策略决策区间计入 -ΔT/1000；终局追加失败惩罚；末区间包含数学回退耗时；四时间桶审计可挂接，势能塑形已接入 rollout | 真实 rollout 逐秒对账待执行 |
| C03/C04/C06 | bounded reward 已接入 no-progress/repeat；pipeline 记录真实标记，倍率上限 3 | 真实长程 rollout 的逐秒统计与 ablation 待执行 |
| F01/F02 | CLI gamma/GAE 参数实际传入 rollout | 三组 gamma、三组 lambda 实验未执行 |
| AC02 | formal 默认累计至少 2048 transitions 后 update；保留整局；每局保存；超时未足批次不 update；支持 optimizer/Python/NumPy/Torch RNG、rows/transitions/updates 恢复 | 正式大规模训练未执行；性能验收待执行 |
| AC03 | 每 epoch 随机 minibatch、末批保留、梯度裁剪；GNN/attention/critic 屏蔽 padding | 真实训练性能验收待执行 |
| AD01/AD02 | formal 默认 train 10000–10199、val 11000–11049、test 12000–12049；训练脚本拒绝 seed 重叠并写入 checkpoint/summary | 正式大规模训练待执行 |
| E03 | 固定 validation seeds 的 greedy 评估、统一 validation gate、checkpoint safety_qualified，以及执行前 clear proof/clear_audit/illegal_clear/safety_violation 均已接入 | 真实大规模 validation 尚未执行；未通过 gate 的 smoke checkpoint 不得晋级 |

验证环境：`D:/Anaconda3/envs/gesture_env/python.exe`，PYTHONPATH 包含
`SXJM/Way4`、`SXJM/Way4/src`、`SXJM`。

本轮验证：transition-batch/checkpoint/minibatch 3 passed；
relational-policy/minibatch 5 passed。测试包含真实参数更新、每 epoch 样本全覆盖、
末批保留，以及补零前后 valid logits/critic 相等。

下一依赖：完整时间/progress 审计与安全 validation gate；之后才进行正式训练和消融。

2026-09-13 端到端运行记录：seed 2000 的 `route_math + adaptive_scan` 两次运行均超过
30 分钟且无结果文件，随后终止；因此该配置当前只有可运行性阻塞证据，没有 full-clear、
安全计数或性能结论。后续需先优化规划计算预算/缓存，再做大样本 gate。

CLEAR 审计增量：pipeline 已启用 strict clear safety；无执行前证明的 CLEAR 会被拒绝且不推进时间，
并写入 rejected audit。executor/homing 回归 16 passed、3 skipped。证明在调用环境前计算；实际命中
不能反向使无证明清除合法；near 证书按实际目标距离检查。

AE01–AE07：新增统一 `compute_efficiency_metrics()` 与 episode 输出字段；当前
no-progress/unnecessary-return 仍使用保守零值，直到 primitive progress/route-regret
数据接通后才可作为正式性能统计。

AE08：新增 `backbone_pruning_rate(initial_count, visited_count)`，并将其纳入
`EfficiencyMetrics` 与 `compute_efficiency_metrics()`；校验 `0 <= visited <= initial`，
避免把被远程偿还的节点误计为实际访问。效率指标回归 5 passed。

W01–W03：新增 `route_regret(delta_time, plan_before, plan_after)`，并接入 pipeline
每个 macro 的 FutureCost 前后估计；decision audit 记录 `route_plan_before_s`、
`route_plan_after_s` 和 `route_regret_s`。相关 planner/pipeline 回归 14 passed。

回归修复：补回 `EpisodeResult.decision_audit` 公共字段，允许效率函数接收 pipeline
传入的交叉诊断参数，并将秒制 `route_cost_seconds/route_score` 放回
`UnifiedRoutePlanner`。小 stall-limit 下跳过额外恢复窗口以保持明确的终止上界；相关失败
回归测试现为 2 passed。全量项目测试曾达到 410 passed/12 skipped，但此前收集阶段有 4
项契约失败，修复后的全量重跑仍需继续确认。

AH01–AH04：补充 `RECOMMENDED_ABLATIONS`（M0–M3、P0–P4）与
`compare_ablation_steps()`；逐步比较仅在两侧均有完整真实结果时计算，否则明确返回
`insufficient_data`，不把配置矩阵误报为实验结论。协议回归 2 passed。

AI01–AI04：新增 `performance_gate()` 与 G1/G2/G3/FINAL 阈值（1000/700/500/400 s per
target）；只有时间达标、Full-clear=100%、clear correctness 与 certificate soundness
同时成立才通过，最终 Gate 允许 `<=400`，其余 Gate 使用严格 `<`。门禁回归 2 passed。

全量回归复核：在 Way4 实际项目根目录执行 `python -m pytest -q tests`，结果为
`414 passed, 12 skipped`（8 个可选 Torch/环境警告）；此前 4 个契约失败已全部修复。

AC01–AC03：复核 `scripts/train_candidate_ppo.py`：smoke profile 明确为 2 个训练 seed、
1 update；正式 profile 按 `transitions_per_update`（默认 2048）收集后立即 update，
trainer 使用 shuffle、可配置 4 epoch minibatch 与 gradient clipping，并逐 episode
持久化 checkpoint。transition threshold、checkpoint 持久化与 minibatch 回归覆盖通过。

AF01/AF02：新增 `compute_batch_metrics()`，输出 scan batch 数量、均值、中位数、最大值、
提前中断次数及原始 `batch_stop_reason` 序列；不从缺失 trace 推断原因。批次指标回归
1 passed。

AF01/AF02 接线：`EpisodeResult` 新增 batch size 与 early-stop 字段，`Way4Pipeline._result()`
从真实 `decision_audit.measure_count` 和显式 stop reason 汇总并写入结果；无记录时保持
零值/空列表。pipeline batch 结果与事件回归 4 passed。

AG01–AG03：新增 `build_hard_trace_audit()`，统一输出 route、measurement、time 三个
稳定字段组，覆盖移动/新路线/回头/重复边/certificate-only/无必要返回/可消交叉，测量
总数/有效/无进展/重复/证书/定位，以及总时间、各时间桶和 route-regret waste。审计回归
1 passed；真实 trace 的细粒度 useful 标签仍依赖执行器提供对应字段。
该指标已经可用于 trace，仍需多 seed 统计后才能形成性能结论。

B01：decision audit 新增逐步 `delta_accounting_error_s`；`TransitionAudit.validate()`
会拒绝未能由移动/测量/切频/清除四桶解释的时间。新增对账回归通过。

新增 `audit_from_record()`：读取 pipeline decision trace 时会实例化并校验
`TransitionAudit`，错误时间记录会直接失败。审计解析与 dense rollout 回归 12 passed。

C01/C02/C03/C04/C06 补强：`timed_policy_transitions()` 在输入包含四个真实时间桶时
附加经过校验的 `TransitionAudit`，并继续使用真实 `no_progress`/`repeat_measure` 标志
生成 dense 时间 reward；缺失桶时保持兼容但不伪造 audit。相关 rollout/parser 回归
6 passed。

A01/A02：新增机器可读架构契约 `architecture_contract.py`，并写入 DESIGN：总时间
最小化受 FullClear/IllegalClear/SafetyViolation 硬约束，PPO 只能返回合法候选索引。
契约测试 2 passed。包含 planner mode 的扩展回归再次长时间无输出，已单独停止，
不能据此宣称该运行通过。

Y03：ProgressWatchdog 新增最近四个动作签名的 A→B→A→B 检测；pipeline 将宏类型和
目标坐标传入 watchdog，命中后进入既有数学回退。watchdog 回归新增 2 个测试。

G01/G02：`Way4Pipeline` 默认已切换为 `adaptive_scan=True`、`batch_stop=True`；
两者仍可显式关闭以复现 legacy ablation。默认值契约新增测试。

E01/E02：Candidate-PPO `RewardConfig` 已改为 full-clear 额外奖励 0、失败终局
`-100-N_unresolved`；13 个 RL objective/train/rollout 回归通过。旧 `TabularSMDP`
配置仍保留独立兼容参数，尚未作为 Candidate-PPO 正式训练路径使用。

H03：CandidateGenerator 的正式默认 `min_sensing_efficiency=1e-3` 已启用，按硬覆盖
增益/完整批量扫描时间过滤低收益 EXPLORE/VERIFY；`0.0` 仅作为显式消融配置保留，
并拒绝负阈值。新增回归 2 passed。该项完成的是门控接线与单元证据，仍需正式多 seed
训练统计确认阈值对 full-clear 与总时间的影响。

I01：`compute_route_metrics()` 的 `backtrack_m` 改为按最近最多 3 个 waypoint 的主运动
方向计算，使用 `cos(theta)<-0.5` 判定；短轨迹使用可用历史长度，且暴露 `backtrack_k`
与阈值参数并校验范围。反向腿与重复腿诊断回归 3 passed。

I02：`repeated_edge_m` 改为基于完整历史 trajectory polyline 的共线线段重叠长度，
每条新腿的重复长度封顶于该腿长度，避免重复累计失控；既有 `compute_efficiency_metrics()`
输出 `repeated_edge_ratio = repeated_edge_m / move_distance_m`。新增部分重叠轨迹回归，
路由诊断 2 passed。

I03：新增独立 `self_intersection_count` 字段并导出到标准 episode row；它仅统计非相邻
轨迹线段的几何交叉，不进入规划代价或 PPO 惩罚。新增交叉轨迹回归，路由诊断 3 passed。

I04：对每个非相邻交叉按 checklist 的 2-opt 重接条件比较原始长度
`d(A,B)+d(C,D)` 与 `d(A,C)+d(B,D)`，新增 `avoidable_crossing_count` 和
`avoidable_crossing_m`，并导出到标准 episode row；仅作诊断，不直接处罚所有交叉。
交叉/2-opt 回归 3 passed。

I05：修正 `unnecessary_return_m` 的证据边界：新增
`compute_unnecessary_return_m(points, ready_tasks)`，仅在显式提供
`(task_point, first_ready_index, execution_index)` 且存在首次接近后再远离时计量；
无 ready 证据时不再把普通 revisit 冒充不必要返回，Route Regret 仍由 decision audit
独立记录。I05/Route Regret 回归 6 passed。

I06：`EfficiencyMetrics` 新增独立 `coverage_overlap_ratio` 与 `route_overlap_ratio`。
前者由“已覆盖重叠增益/总覆盖增益”计算，后者由“重复轨迹长度/移动长度”计算，并在
标准 episode row 导出；输入一致性校验已加入，回归 7 passed。当前 pipeline 尚未产生
逐次 coverage-overlap 原始增益，因此该字段在未提供原始证据时保持 0，不能据此宣称
已完成多 seed 的覆盖重叠统计。

J01：`UnifiedRoutePlanner` 新增 `route_cost_seconds()`，将路线长度、长跳和交叉惩罚
按 cost-model speed 折算为秒；兼容入口 `route_score()` 改为委托该秒制目标，避免米、
常数分数与秒直接相加。相关路由/效率回归 20 passed。

J02：新增 `route_waste_penalty_seconds()`，按任务类型 mask 轨迹浪费：Explore 强罚、
Verify 中罚、Refine 按 info/sec 衰减、Pursue/Clear 低罚；所有项先按速度折算为秒，
仅作为排序信号，不进入安全门。任务权重与单位回归 9 passed。

K01：新增并导出 `FocusController/FocusState`，按
`P_completion / T_expected_to_finish` 超阈值进入 `FOCUS(channel)`，并提供 no-progress、
hypothesis invalid、reacquire failed、better task、watchdog 五类退出原因接口；非法概率、
时间和阈值会拒绝。单元回归 2 passed。当前控制器尚未接入 pipeline 的候选选择循环，
因此 K01 仍为“机制已实现、端到端启用待完成”，不能宣称任务承诺已改变实际规划行为。

K01/K02 接入：pipeline 现在为 REFINE 候选写入 `p_completion`，在承诺分数超过阈值后
进入 FOCUS(channel)，后续候选仅保留该 channel（及合法 EXIT）；连续 3 次无进展或
planner watchdog fallback 时释放 FOCUS。completion-oriented REFINE 候选显式标记
`p_completion=1.0`。Focus、pipeline 默认值和路由回归 8 passed。

L01：`SpatialStopGenerator` 已将同一空间 stop 的多 channel scan/refine/verify 服务
合并，并把 cluster 半径内的 CLEAR 折叠到同一 stop；同时保留全部 legacy candidates
作为安全回退。`SpatialStop.primitives()` 按 measures-then-clears 展开，空间 stop 回归
8 passed。

L02/L03：空间聚类使用可配置半径（正式默认 1000 m）与 DBSCAN，允许 100–200 m 级
空间近邻任务合并而非要求完全同坐标；每个 `SpatialStop` 新增可供 math/PPO/route
使用的 `task_cluster_size`，并同步写入 metadata。空间聚类回归 9 passed。

M01：`RemainingTaskPool` 新增统一 `pending_tasks` 注册表，覆盖 EXPLORE、INITIALIZE、
REFINE、REACQUIRE、PURSUE、CLEAR、VERIFY、BACKBONE；同步仍只读 belief，且支持外部
任务注册，不改变安全状态。任务池与既有 route assignment 回归 6 passed。

M02：`UnifiedRoutePlanner` 新增 `insert_at_cheapest()`，严格按每个相邻路段及尾部
插入位置计算最小 Δ，不重建既有路线或隐式执行 2-opt；route cleanup 留给独立周期。
新增插入回归，routing/opportunity/task-pool 回归 21 passed。

M03：新增 `insert_clear_if_cheap()`，clear-ready task 复用现有路线的最小插入增量，
仅当 Δ 不超过阈值时插入，否则保留原路线；用于避免扫描后专程 clear 再原路返回。
clear insertion 与任务池回归 19 passed。pipeline 的持续路线维护仍由 M04 完成。

M04：新增 `RouteCleanupScheduler`，按默认每 5 decisions、750 m 或重大 belief 变化
触发 cleanup；`UnifiedRoutePlanner.cleanup_route()` 执行确定性严格改进的 2-opt 路径
修复，并仅在 ΔT<0 时接受结果。route cleanup/cadence 回归 16 passed。

M05：`RecedingHorizonPlanner` 新增 `beam_search()`，支持 H=3–5 和 beam width B，
逐层保留 top-B 序列并以 terminal `J_hat` 排序；`plan()` 在 horizon>=3 时使用该 beam
首动作。当前后续层使用不可变候选池，属于保守静态 beam，待 simulator successor
generator 接入后再升级为状态转移 beam。beam/cache 回归 2 passed。

N01：新增并导出 `time_debt(estimated_remaining_s, remaining_lower_bound_s)`，按
`max(0, T_hat_remaining - T_remaining_LB)` 计算并校验非负时间；lower-bound/future-cost
回归 15 passed。尚未将该指标接入 candidate feature 与实时 trace，N02 待执行。

N02：新增兼容的 v3 `time_debt_block(before_est_s, after_est_s, remaining_lb_s)`，输出
`time_debt_before/after/delta` 的归一化三元组；保留 v2 feature 维度以避免旧 checkpoint
失配。当前 pipeline 尚未提供逐决策可信 remaining-LB，因此未伪造实时 debt 数值；Time
Debt block 回归 2 passed，完整 online 接线仍待 lower-bound 来源接入。

O01：新增兼容的 v3 `candidate_time_block()`，输出
`predicted_move/measure/switch/service/total_delta_time` 五个归一化即时真实时间字段；
优先读取 candidate metadata，缺失子项保持 0，仅 total 使用 `expected_time`，不从总量
伪造分解。旧 v2 feature 维度不变，时间特征回归 4 passed。

O02：新增兼容的 v3 `route_efficiency_block()`，覆盖 route_delta、route_rank、detour、
backtrack、repeated edge 及 ratio、unnecessary return、avoidable crossing gain、
corridor distance、next-task distance；米制字段按 ARENA_R 归一化，比例字段保持独立，
旧 v2 feature 维度不变。路线特征回归 5 passed。

O03：新增兼容的 v3 `progress_per_second_block()`，输出 coverage、certificate、
information、clear probability、localization 五类 progress/sec 特征；统一以候选
实际时间归一化，并拒绝非正时间或负收益。Progress feature 回归 6 passed。

O04：新增兼容的 v3 `future_route_block()`，覆盖 remaining route LB before/after、
future cost delta、time debt delta、nearest next task、task cluster size；秒制和米制
分别归一化，允许有符号 delta，旧 v2 feature 维度不变。未来路线特征回归 6 passed。

O05：新增兼容的 v3 `history_behavior_block()`，覆盖 region/channel 历史访问次数、
time/distance since last progress、recent route overlap ratio；时间/距离归一化并校验
计数与比例范围，旧 v2 feature 维度不变。历史行为特征回归 7 passed。

P01–P04：新增 planner-side `BackboneManager` 及 `BackboneNode/Edge`、
`CoverageResponsibility`、`DirectionalTriangle`、`BoundaryCap`、`CoverageDebt` 数据结构。
节点状态与“必须访问”解耦，支持 visit 和 remote observation 将未来节点标记为
`SATISFIED_BY_OTHER_OBSERVATION` 并从剩余路线移除；不修改 belief/certificate。backbone
manager 回归 9 passed。

P05：`BackboneManager.apply_observation()` 现在接受动态 Refine/Clear 观测覆盖的
coverage cells、directional triangles、boundary caps、certificate holes 和 channels，
逐项偿还对应 responsibility；部分覆盖标记 `PARTIALLY_SATISFIED`，全部覆盖标记
`SATISFIED_BY_OTHER_OBSERVATION` 并移出剩余 route。backbone manager 回归 3 passed。

Q01/Q02：新增离线 `optimize_p3_backbone()` 与 `P3BackboneResult`，以确定性 arena grid
执行圆覆盖 greedy 近优选择，并联合输出点数、route length、overlap proxy、outside
proxy 和加权 objective；新增 V6 风格 `center_ring_baseline()`。该版本的覆盖可行性是
grid-level proxy，尚未替代 HardDiscCoverVerifier 的保守证明；P3 backbone 回归 4 passed。

R01：扩展工程化 `DirectionalTriangle` 对象，记录 detector A/B/C、covered region、
directional guarantee 和 certificate status，并保留现有 directional certificate 函数
用于几何检验。对象与原 directional certificate 回归 5 passed。

R02：新增 `build_p4_triangle_mesh()`，构造确定性的中心—环形三角网格，输出 12 个
三角责任区域与 13 个去重 detector 点；每个 triangle 记录 covered region。当前是
几何 backbone 的规划网格，圆盘边界的正式全覆盖/方向证明仍需 R03 Boundary Cap 与
保守采样 verifier 联合确认。三角网格回归 2 passed。

R03：扩展一等公民 `BoundaryCap`，新增 detector_points、covered_arc、outward directions
和 certificate status；新增 `build_boundary_caps()`，允许 detector 位于 R=1800 外侧，
避免将外向定向源错误限制在 arena 内。BoundaryCap 回归 3 passed；正式 cap coverage
验证仍需 R04 几何解析检查。

R04：新增 `BackboneGeometryAudit`/`audit_backbone_geometry()`，离线检查 side length、
boundary support、triangle/cap coverage、worst outward direction 与 supplied samples 的
full-arena inclusion；解析检查与三角网格回归 2 passed。该检查是离散/结构审计，仍不
替代 HardDiscCoverVerifier 或完整连续域定理证明。

S01：`BackboneManager.initialize_open_route()` 现在在初始化时对所有未满足节点构造
确定性的全局 open route，并保存 node id 顺序与按 speed 折算的 `route_cost_s`；已满足
节点不会进入初始路线。Backbone route 回归 4 passed。

S02：新增 `BackboneManager.prune_route()`，observation 后仅从既有 route 移除已满足、
替代、冗余等节点，保留其余节点原有顺序并重新计算剩余秒制 open-route 成本；不从零
生成新路线。Backbone prune 回归 5 passed。

S03：新增 `accept_shortened_route()`，动态信息驱动的计划只有在旧 route 的有序子序列
且新成本不增加时才接受；隐式新增节点或成本上升均拒绝，保证“完整安全计划→逐步剪短”
而不是无限追加 candidate。Backbone policy 回归 6 passed。

T01：新增 `CandidateGenerator.refine_with_piggyback()`，对每个 Dedicated REFINE
 保留原候选，并在给定 backbone 点上生成带同一 refine channel 的 Piggyback 变体，
 标注 `piggyback_backbone_index`。接口回归 2 passed；当前尚未把 backbone 点自动注入
 默认 `generate()`，因此正常管线接入仍是后续项。

T02：新增 `RefineVariantEvaluation`、`evaluate_refine_variant()` 与
`choose_refine_variant()`，按 `J(q)=ΔT_route+T_measure+E[T_remaining]` 比较
Dedicated/Piggyback，而不是只按 `p_clear` 或 `info_gain`；联合目标回归 2 passed。

T03：新增 `WAIT_FOR_BACKBONE(index)` 状态常量与 `mark_wait_for_backbone()`；仅当
 额外绕行不超过显式阈值时标注等待状态，允许执行层避免专门折返。状态回归 2 passed；
当前是候选元数据/策略原语，尚未接入默认 route scheduler 的自动等待决策。

U01/U02/U03：新增 `CandidateGenerator.backbone_candidates()`，输出沿既有 route 的
`BackboneNext`、责任已替代后的 `BackboneSkip`，以及按当前位置选择最近未完成节点的
`BackboneBridge`；三类候选保持执行器兼容的 `EXPLORE` action，并以 metadata 表达
backbone 语义。候选回归 2 passed；尚未接入默认 `_choose_macro()` 的 backbone 优先级。

U04：新增 `backbone_closure_candidates()`，按剩余 `certificate_holes` 与方向责任债务
排序，优先提出最后缺失几何区域的 `BackboneClosure` 候选。U05：已有
`refine_with_piggyback()`，现统一标注 `PiggybackRefine` 等价元数据。U06：新增
`piggyback_clear_candidates()`，当 clear 点与未来 backbone 点的 detour 不超过阈值时
生成 `PiggybackClear`。回归 2 passed；三类候选尚未接入默认 macro 选择优先级。

AC04/AC05：新增 `hard_case_mining.py`，对 boundary、directional、evasive、far-pair、
multi-cluster、certificate-hole、high-measurement、long-route 等显式困难标签提高采样
权重，并叠加 `T/T_LB` 超额比例；输出归一化概率且保持输入顺序。困难样本回归 1 passed。

AB01/AB02/AB03：新增可选 `HierarchicalPolicy`，分别输出 waypoint spatial logits 与
channel/STOP logits，并提供 `log π_spatial + log π_channel` 联合概率接口；不改变现有
扁平 CandidatePPO。层次策略回归 1 passed。

V01/V02：新增 `BACKBONE_DECISIONS=(FOLLOW,SKIP,INSERT,DETOUR,RETURN)` 与
`backbone_feature_block()`，将候选是否属于 backbone、索引、绕行/重入成本、debt、
跳过收益与 piggyback 收益编码为独立特征块，使 PPO 学习“何时偏离 backbone”；不改变
既有冻结 `FEATURE_DIM`，由 backbone-aware 导出器显式追加。特征回归 6 passed。
X01/X02：新增 `EndgamePlan` 与 `solve_endgame()`；小规模 unresolved 点集进入确定性
Held–Karp open-route solver，输出顺序、路线长度与 `used_exact`，空集直接完成，避免残局
继续交给自由 PPO。残局 solver 回归 2 passed；尚未由 pipeline 按 unresolved 阈值自动切换。

X02 补强：新增 `EndgameController`，当 unresolved 不超过配置阈值时将候选 waypoint
交给 exact open-route solver，并选择精确路线首点对应候选；超过阈值或无候选时保持旁路。
控制器/solver 回归 4 passed；默认 pipeline 自动切换仍需显式接线与端到端验证。

X01/X02 接线：`Way4Pipeline(planner_mode="final_ppo")` 在 unresolved≤3 时调用
`EndgameController`，由 exact open-route 首点直接选择候选并标注 `endgame_solver`；
其他模式不改变。pipeline endgame 回归 1 passed。
AA01/AA02：Critic 新增可选 `mean_max` pooling，并支持 `global_time_dim` 输入；新增
`critic_time_state_block()` 编码 elapsed_time、unresolved/cleared count、certificate
coverage、remaining route lower bound、time debt 与 phase。保持默认模型兼容，时间状态
回归 2 passed。
AD01/AD02：新增 `SeedSplits`，明确 2000–2009 仅为 development regression，默认 train=
10000–10199、val=11000–11049、test=12000–12049，并提供无重复/无交叉校验。seed split
回归 2 passed；既有旧 smoke seed 不再作为 unseen test 声称。

AD02 补强：`train_candidate_ppo.py` 新增 `--test-seeds`，训练、validation、test 三组
均要求非空且两两不重叠，checkpoint 与最终摘要保存 validation/test seed 清单；split
回归与 checkpoint/transition 回归共 3 passed。

本轮验收：D02 的 `phi_before/phi_after` 已由 `timed_policy_transitions` 实际传入
dense-time reward，回归 15 passed；G06 将 `ChannelScheduler` 默认 `early_cap` 从 20
降为 5，adaptive-scan/scheduler 回归 14 passed。`git diff --check` 通过。

继续验收：pipeline 每个 decision audit 现记录 B03-B05 的 `progress_before/after`、
status transition、delta_clear、delta_certificate、面积/MEC 缩减及无进展时间/距离/决策数；
同时修复 route-regret 缺失值汇总。pipeline/trace 回归 3 passed、2 skipped。

G03-G06 接线复核：adaptive scan 已通过 `MacroExecutor.execute_adaptive` 在每次
measurement 后 fold belief/certificate 并重新选择；pipeline 改为复用已配置的
`self.channel_scheduler`，不再丢失 scheduler 权重与 early_cap。相关回归 17 passed、
2 skipped。

G07：`AdaptiveScanSession.choose()` 现在按单频道的 value-per-second（价值除以
measure+必要 switch 时间）重排频道；batch `select()` 仍保留批量扫描的 sunk-move
经济性。scheduler/adaptive/executor/pipeline 回归 17 passed、2 skipped。

T/U 补强：Refine 的 piggyback 变体现在明确标记 `backbone_kind=PiggybackRefine`，
与 BackboneNext/Skip/Bridge/Closure/PiggybackClear 元数据体系一致；相关 backbone、
closure、等待与 piggyback 回归 7 passed。

B 链路补强：`audit_from_record()` 现在保留 `delta_clear`、certificate、localization
area、MEC、entropy 与 hypothesis 增量，pipeline decision audit 到 PPO transition 不再
丢失 progress 证据；审计/rollout 回归 13 passed。

AF02 补强：pipeline 按真实 primitive observation、状态迁移、coverage gain、clear
结果与 batch 截断情况记录 `near`、`strong_bearing`、`localized`、`certificate_gain`、
`clear_ready`、`replan`、`low_marginal_value` 等 batch stop reason；相关 pipeline/trace
回归 4 passed、2 skipped。

E03 补强：独立 `validate_candidate_ppo.py` 现输出每个 seed 的 `illegal_clear`、
`safety_violation`，并调用统一 `validation_gate`；验证脚本只有在 full-clear、两项安全
计数为零、greedy 与时间字段均有效时返回成功。Gate/算法回归 10 passed。

AC04/AC05 补强：训练脚本从实际 case 的定向属性、边界半径、近距离源对与源数量推导
hard-case tags，并与实测 `time_s/lower_bound_s` 一并写入 `case_stats`；后续
`choose_seed()` 的 tags 与 T/LB 权重不再是空输入。训练脚本编译及采样/集成回归 4 passed。

C05：`dense_time_reward`、`TransitionAudit` 与 `timed_policy_transitions` 新增可选
`detour_time_s`，明确按 expected extra seconds 计入奖励；未提供时保持旧行为。奖励、
rollout、审计回归 11 passed。

C05 主流程补强：pipeline 将候选 `meta["detour_time_s"]` 写入每步 decision audit，
使绕行估计可继续进入 rollout reward；相关审计、batch 与 reward 回归 10 passed。

全量回归：当前 Way4 工作树 `437 passed, 12 skipped, 8 warnings`，耗时约 467 秒；
未发现由本轮 B/C/E/G/AC 改动引入的回归失败。

C05 训练记录补强：`TorchCandidatePolicy` 将实际选中候选的 `meta["detour_time_s"]`
写入 PPO rollout record，令 detour penalty 从候选、策略选择到 reward 的数据链路闭合；
候选策略/rollout 回归 12 passed、5 warnings。

2026-09-13 M0 seed 2004 增量完成：`ablation_M0_seed2004_retry.json` 正常退出，
full_clear=true、resolved=20、illegal_clear=0、safety_violation=0、virtual_time_s=14038.199124。
已与 seeds 2000–2003 合并为 `ablation_M0_seeds2000_2004_current.json`；5-seed 审计
mean=733.342971、median=706.883608、P90=813.655444、max=816.584251 s/target。
G1–FINAL 均未通过：G1 受 certificate_sound 未附证据影响，G2–FINAL 同时超过时间阈值。

AC01/AC02 运行证据：使用 `--profile smoke --updates 1 --max-steps 1` 实际完成 2 个
training episodes、2 transitions、1 PPO update，并生成 `results/candidate_ppo_smoke_current.pt`
与 `.state`；validation 运行且因 FullClear 未达标正确拒绝 checkpoint，illegal_clear=0、
safety_violation=0。此前默认 max_steps 的长运行已停止，未将其误记为成功训练。

F01/F02 运行证据：执行 `scripts/ppo_credit_ablation.py` 生成
`results/ppo_credit_ablation_current.json`，覆盖 gamma={1,.995,.99} 与
GAE lambda={.95,.97,.99} 的全部 9 组合；协议/消融回归 3 passed。

AH runner 补强：`run_deterministic_ablations.py` 新增 `--matrix recommended`，可直接
按 checklist 的 M0-M3/P0-P4 名称运行；P0-P4 在 deterministic-only runner 中保留为
显式 `unsupported_configuration`，不会以数学结果冒充 PPO。编译通过，并生成
`results/ablation_recommended_probe.json` 作为 P4 显式未执行证据。

AH 实证审计：汇总 `results/ablation_matrix_real_2000.json` 得到整体 `complete=false`；
仅 `way4_full`、`adaptive_rerank`、`adaptive_rerank_stop` 有真实完整行，其余旧 cell
为 `unsupported_configuration`。因此 M0-M3/P0-P4 的完整多 seed 消融仍明确未完成。

C05 审计约束：`TransitionAudit.validate()` 现对 `detour_time_s` 执行 finite/non-negative
校验，与 move/measure/switch/clear 四个时间桶一致，拒绝坏 detour 数据进入训练；相关
transition/reward 回归 14 passed。

B04 补强：watchdog 的 `_progress_key()` 现纳入有限 belief feasible-area 总量，显著面积
缩减会被认定为真实 progress，不再只依赖 resolved/coverage/MEC；pipeline/homing/progress
回归 7 passed、5 skipped。

B04 再补强：P4 启用 `HypothesisLayer` 时，alive omni+directional hypothesis mass 也纳入
`_progress_key()`；真实假设消减会被 watchdog 识别为 progress。相关 pipeline、homing、
hypothesis 回归 17 passed、5 skipped（约 322 秒）。

B01 字段契约补强：`TransitionAudit` 增加 `delta_virtual_time`（seconds）别名，并在
`to_record()` 同时序列化 `delta_virtual_time_s` 与 `delta_virtual_time`；时间审计/rollout
回归 9 passed。

B02 补强：pipeline decision audit 现在写入真实 `position_before` 与 `position_after`
坐标，可独立复算 `delta_move_distance_m`；相关 decision/pipeline 回归 3 passed、2 skipped。

B02 路线诊断补强：每个 decision audit 现在保存累计 route metrics 的前后快照差值，
并输出本决策的 `backtrack_m`、`repeated_edge_m`、`unnecessary_return_m`，不再只在
episode 末端提供不可分解总量；相关 pipeline 回归 3 passed、2 skipped。

B03 字段契约补强：decision audit 新增顶层 `channel_status_before` 与
`channel_status_after`，同时保留完整 `progress_before/after` 嵌套快照；审计/rollout
回归 4 passed。

B03 hypothesis 补强：P4 decision audit 记录 hypothesis alive mass 的前后值，并输出
正值表示假设减少的 `delta_hypothesis`；与 watchdog 使用同一真实来源。pipeline/hypothesis
回归 13 passed（约 334 秒）。

AE05/I06 补强：pipeline 对扫描 decision 记录 CoverageGainMap 的预测/实际新增覆盖，
并输出 heuristic `coverage_overlap_ratio`；该诊断明确不参与 hard certificate 或安全判断。
pipeline/trace/efficiency 回归 8 passed、2 skipped。

AE05/I06 episode 汇总补强：`_result()` 汇总各 decision 的 predicted/actual coverage，
将 coverage overlap 传入 `compute_efficiency_metrics()` 生成 episode 级 ratio；仍明确是
heuristic 诊断，不改变安全 gate。效率/pipeline 回归 8 passed、2 skipped。

M04 接线：pipeline 新增 `RouteCleanupScheduler(decision_period=5, distance_period_m=750)`；
到期时调用严格降成本的 `UnifiedRoutePlanner.cleanup_route()`，并在候选 metadata 标记
`route_cleanup_applied`。routing/unified-route/pipeline 回归 18 passed、2 skipped。

M05 补强：`RecedingHorizonPlanner.beam_search()` 新增可选动态 `successor(sequence,
base_view)` 展开接口；提供时每层基于 partial sequence 重新取得候选，否则保持静态池兼容。
receding-horizon/beam 回归 12 passed。

N02/O04 补强：新增并公开显式 `evaluation_features_v3()`，在冻结基础特征后追加
time-debt 三元组与 future-route block；旧 `evaluation_features()` 及 checkpoint 维度
保持不变。time-debt/relational policy 回归 13 passed、4 warnings。

AH runner 当前验证：用新 `--matrix recommended --cells M0,M1,M2,M3 --seeds 2000
--max-steps 1` 生成 `results/ablation_M0_M3_probe_current.json`；四个新 cell 均实际
执行并保留真实的 incomplete 单步结果（无 unsupported 替代）。该 probe 仅证明入口和
配置映射，不作为性能/FullClear 结论。
候选策略/rollout 回归 12 passed、5 warnings。
