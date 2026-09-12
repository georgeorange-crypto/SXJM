# Way4 implementation checklist — evidence audit

审计日期：2026-09-12。判定规则：`[x]` 有代码+测试/运行证据；`[~]` 有部分实现但未闭环或缺验收证据；`[ ]` 未发现实现；`[?]` 因环境/证据不足无法确认。路径均相对于本项目根目录。

## 总结

- P0 主骨架：`[~]`。P0-000～P0-025 已有代码、测试或运行证据，包含 effective geometry、cardinality lifecycle、INITIALIZED/PRESENT_UNOBSERVED、统一 action mask、adaptive scan、事件总线和时间模型；剩余主要是 P0-026 的独立 SMDP learner 大规模效果证据及 P0-001 的全 consumer 收敛。
- P1：`[~]`。NBV/VOI、routing/K_i、FutureCost、deterministic expert release 和 reward/value 契约已补齐；仍缺大规模消融、稳定训练和跨场景统计。
- P2：`[~]`。特征向量、residual/candidate PPO、lower-bound 基础已实现；仍缺 LP/凸松弛、理论证明及稳定样本效率实验。
- P3：`[~]`。`gesture_env` 下完整 pytest 曾通过（205 passed，12 skipped），并新增 soundness、seed audit、expert trace 与 P4 gate 结果；仍缺大样本 paired fair comparison 和全矩阵消融。
- P0 Gate：多个 gate 已完成或有强证据；端到端大样本比较及若干理论/训练 gate 仍保持 `[~]`，不把局部测试等同于最终研究完成。

## P0 — deterministic hard skeleton

| 状态 | ID | 证据与结论 |
|---|---|---|
| [x] | P0-000 | 已创建 Git freeze commits `399d6ad`、`8f09be4`、`25f9020`；`FREEZE_MANIFEST.md` 与 `results/way4_source_fingerprint.json` 记录权威源码根、环境、逐文件 SHA-256 和 tree fingerprint。 |
| [~] | P0-001 | 已有统一 effective 查询、`EffectiveRegion`、保守栅格面积/直径/采样；NBV 的 hypotheses 与 planning diameter、scheduler refine proxy 均优先使用 effective geometry（无 native sampler 才 fallback 到 `F_c`），相关 soundness/NBV/scheduler targeted regression 5 passed。仍需审计所有 legacy duck-typed consumers，故保持 partial。 |
| [x] | P0-002 | `record_no_signal()` 立即更新 exclusion；NBV/native hypothesis sampling 使用 effective samples，安全几何仍明确使用 outer `F_c`，避免把近似域误作 certificate。 |
| [x] | P0-003 | 已新增 `EffectiveRegion` 显式保守栅格表示，支持 holes/multiplicity 的连通分量统计，并与安全 outer `F_c` 分层。 |
| [x] | P0-004 | `CardinalityState` 计算并传播 `p,z,u,q_min,q_max`；pipeline 每轮执行 cardinality closure，并将 `q_min == unknown` 的未知频道转为 `PRESENT_UNOBSERVED`。 |
| [x] | P0-005 | 已加入 `PRESENT_UNOBSERVED`，只在 cardinality closure 证明时传播，不伪造 bearing。 |
| [x] | P0-006 | `INITIALIZED` 要求至少两条相差 10° 以上 bearing；candidate、scheduler、future-cost 与 receding-horizon 均识别 INITIALIZED/PRESENT_UNOBSERVED。 |
| [x] | P0-007 | 生产代码中的 `.status =` 已收敛到 `belief/channel.py`；certificate/executor 通过受控 transition 方法调用，新增静态写点审计测试。测试夹具中的直接赋值不属于生产路径。 |
| [x] | P0-008 | `certificate/manager.py`、`belief/channel.py`、`tests/test_certificate_manager.py` 覆盖 MEC/硬 certificate、18m 阈值和 absent 只由证书层写入。完整测试受环境错误阻断。 |
| [x] | P0-009 | readiness score 已综合面积、直径、`kappa`、连通分支、bearing diversity、coverage debt 与 robot 距离，并有边界/惩罚测试；硬状态门槛仍由 `initialization_ready` 控制。 |
| [x] | P0-010 | scheduler 已调用统一 `core.actions.get_allowed_actions()`，覆盖 UNKNOWN/DETECTED/INITIALIZED/PRESENT_UNOBSERVED/LOCALIZED/CLEARED/ABSENT，并有回归测试。 |
| [x] | P0-011 | `CandidateGenerator._explore_candidates()` 生成正式 `MacroActionType.EXPLORE`，由 scheduler 选 batch 并由 executor 执行。 |
| [x] | P0-012 | `CandidateGenerator._initialize_candidates()` 为 `PRESENT_UNOBSERVED` 生成独立 `INITIALIZE` macro，并保留 readiness/几何后续 refinement。 |
| [x] | P0-013 | `MinimaxNBV.initialization_point()` 按 bearing crossing quality 选点，candidate generator 对 `INITIALIZED` 优先使用，并有测试。 |
| [x] | P0-014 | `MacroActionType.VERIFY`、completion-anchor fallback、verification scheduler mode 和 executor 路径均已存在。 |
| [x] | P0-015 | `certificate/manager.py`、`executor/macro_executor.py`、`tests/test_certificate_manager.py` 和 `test_macro_executor.py` 有 clear certificate/clear execution 证据。 |
| [x] | P0-016 | candidate generator 已覆盖 UNKNOWN Explore/Verify、PRESENT_UNOBSERVED Initialize、DETECTED/INITIALIZED Refine、LOCALIZED Clear。 |
| [x] | P0-017 | `certificate/hard_disc_cover.py`、`fallback.py`、`tests/test_p3_disc_cover_verifier.py` 提供 quadtree/backbone 验证；缺本次完整 pytest 通过记录。 |
| [x] | P0-018 | `coverage_debt`/`uncovered_area` 已持久化，并由 `coverage_snapshot()` 统一导出；hard absence 仍明确由独立 verifier 负责。 |
| [x] | P0-019 | scheduler 已加入 coverage debt、skip promotion、连续 scan debt，并通过 40-step long-trajectory property test 证明 unresolved channels 不会永久 starvation。 |
| [x] | P0-020 | `channels/scheduler.py` 与 `planner/candidates.py` 按 waypoint/状态筛选 channel，不默认每个 waypoint 扫 1..20；有 scheduler tests。 |
| [x] | P0-021 | scheduler 已提供统一 `value_components()`（`V_E,V_I,V_R,V_C,V_D`）和按 measure+switch 归一化的 `eta()`，并由 `channel_value()` 使用。 |
| [x] | P0-022 | `AdaptiveScanSession`、`MacroExecutor.execute_adaptive()` 已接入 `Way4Pipeline(adaptive_scan=True)`；每次 measurement 后重新选择，测试覆盖 STOP/重排。 |
| [x] | P0-023 | `ScanPlan` 已记录 `stop_reason`/`stop_value`/dwell cost；无正 VOI 时返回显式 STOP，且有测试。 |
| [x] | P0-024 | `core/cost.py`、`tests/test_cost_model_gold.py`、`test_way4_gold_timing_vs_env.py` 覆盖速度 5、measure 5、switch 1 等时间模型。 |
| [x] | P0-025 | 已有正式事件类型、pipeline 事件记录与 `EventBus` 全局/按类型订阅接口；新增 `Way4Pipeline(event_bus=...)` 外部只读发布通道，pipeline 外部订阅回归测试通过。 |
| [~] | P0-026 | `OptionTransition` 支持 `transition_dataset()` JSON-ready 导出，且新增独立 `TabularSMDPLearner`，按 option 聚合 semi-Markov elapsed-time return，并覆盖 terminal/full-clear 奖励；真实环境 rollout 训练曲线与效果证据仍缺失。 |
| [x] | P0-027 | `core/actions.py` 使用 macro action；`rl` 有 candidate/residual planner，未见 learner 直接输出 dx/dy/channel 的主路径。 |
| [x] | P0-028 | `planner/candidates.py` 生成候选，scheduler/certificate 过滤，executor 执行；有 `test_candidate_generator.py`、`test_planner_mode.py` 等 mask 证据。 |

## P1 — planning quality and routing

| 状态 | ID | 证据与结论 |
|---|---|---|
| [~] | P1-001/P1-002 | `sensing/nbv.py` 已优先调用 `ChannelBelief.sample_effective_hypotheses()`，并有 NO_SIGNAL 排除测试；仍保留 legacy polygon fallback，且 objective 仍是 minimax diameter。 |
| [~] | P1-003/P1-004/P1-005 | `kappa` 横切候选、`completion_point()`、`verification_point()` 已接入候选池并有测试；仍缺真实观测分布下的 full-clear/empty 概率统计校准。 |
| [~] | P1-006/P1-007/P1-008/P1-009 | 已有 mission-time `value_of_information()`、`sensing_trigger()` 与 `should_replan()` 事件策略并有测试；pipeline 仍默认每个 macro 重新规划，尚缺运行轨迹中的 trigger-vs-always-replan 对比评测。 |
| [x] | P1-010/P1-011 | H=1/2/3 参数化消融测试已加入；`execute_adaptive()` 支持 measurement 后中断/重排，并有 executor 测试。正式大样本性能曲线仍属于 P3 评测项。 |
| [~] | P1-012/P1-013/P1-014/P1-015 | 已有 `InformationRidge`、spatial stop/joint route，并新增可独立运行的 `coverage_greedy()` baseline 与测试；正式跨场景性能对比仍缺失。 |
| [~] | P1-016/P1-017/P1-018/P1-019 | 圆盘型 `K_i` 路由与有限 polygon 顶点 MEC `K_i` 证书均已实现；routing 回归 12 passed。仍缺多极点/真实 multipolygon 边界处理、大规模 TSPN 性能对比及实测统计。 |
| [~] | P1-020/P1-021/P1-022/P1-023 | `FutureCost` 保留任务分解，route cache 按 target set 变化重算，并有按频道 uncertainty 与 crossing proxy 测试；仍缺 MEC/circle/K_i 正式消融、大规模统计及独立线上指标。 |
| [~] | P1-024/P1-025/P1-026/P1-027 | 已有 deterministic expert release；新增 `Decision.to_record()`/`expert_dataset()` 导出全部候选 features、Q_math、chosen 与 temperature，并有测试。仍缺跨场景专家动作全集与 counterfactual rollout 统计。 |
| [~] | P1-028–P1-036 | residual planner 已保持 `Score_math + Δθ` 且 learner 不输出坐标；新增 `rl/objectives.py` 明确验证 `r=-Δt`、full-clear terminal cliff、`V=-E[T_remaining]`，并提供 analytic-only / learned-only / analytic+residual 三组 value ablation 契约，3 个新测试通过。仍缺真实训练曲线、独立 critic 训练和跨 seed full-clear 对比，故保持 partial。 |

## P2 — learning/theory extensions

| 状态 | ID | 证据与结论 |
|---|---|---|
| [x] | P2-001–P2-005 | `rl/features.py` 的默认 `evaluation_features()` 已包含 effective area/diameter、kappa、coverage debt、initialization readiness；feature 维度与回归测试同步锁定。 |
| [~] | P2-006–P2-012 | 新增 `planner/lower_bounds.py`：open-MST search/localization/route lower bounds、服务时间下界、以 `max(travel alternatives)+service` 避免 double counting 的 mission LB，以及严格命名的 `certified_gap`；3 个测试通过。仍缺 coverage assignment/orienteering/TSPN 的 LP relaxation、全任务状态下的证明和 convergence/sample-efficiency 论文级实验，故保持 partial。 |

## P3 — engineering, tests, evaluation and ablations

| 状态 | ID | 证据与结论 |
|---|---|---|
| [x] | P3-001–P3-012 | 使用 `D:\Anaconda3\envs\gesture_env\python.exe` 执行完整 Way4 测试，205 passed、12 skipped；有 smoke、pipeline、timing、routing、fallback、diagnostic scripts。仍没有完整 artifact index。 |
| [x] | P3-013 | positive bearing soundness 与 `test_effective_soundness.py` 的 300 个随机合法 NO_SIGNAL 场景均验证真实 source 保留在 effective set。 |
| [~] | P3-014 | exclusion/no-false-presence 与随机 effective soundness 已有；新增 P4 directional blind-arc 测试，证明 NO_SIGNAL 不触发 omni absence certificate，certificate 回归 13 passed。仍缺多 seed/多场景统计。 |
| [~] | P3-015/P3-016 | 新增并运行 `scripts/negative_soundness_audit.py`，固定 seeds 11/23/47/71/99、500 trials 全部通过：NO_SIGNAL soundness 500/500、clear safety checks 500/500，结果写入 `results/negative_soundness_v1.json` 并有 artifact test。仍缺真实 certificate verifier 的全 seed no-false-absent 端到端表，故保持 partial。 |
| [x] | P3-017 | 新增随机 10～16 present cardinality interval/property tests，覆盖 `p,z,u,q_min,q_max` soundness 与不误标 unknown。 |
| [~] | P3-018/P3-019/P3-020 | 完整 Way4 回归为 205 passed、12 skipped；scheduler、NBV、TSP/TSPN 测试覆盖 timing/mask/hypothesis/brute force；新增固定 seed 端到端审计覆盖 5 个 problem-3 场景，均 full-clear、20/20 resolved、无 error。one-step objectives 的大样本效果仍缺失。 |
| [~] | P3-021/P3-022 | 新增并运行 `scripts/way4_seed_audit.py`，生成 `results/way4_seed_audit_1000_1004.json`：5/5 invariants passed、均正常退出；仍不是 stress 全 seed/多 field-kind 汇总，故保持 partial。 |
| [~] | P3-023/P3-024/P3-025/P3-026 | `way4_p4_gate_2000_2009.json` 固定 10 seeds 全部 full-clear；已有 Math vs Candidate-PPO paired smoke `runs/paired_p4_smoke.json`（2/2 两侧 full-clear，PPO win-rate 0.5，1 regression，worst regression -4238.41s），说明 evaluator 能发现性能回退。尝试 test split 2050–2059 时长时间无结果并已中止，故仍缺 V4 paired fair comparison/大样本稳定性，保持 partial。 |
| [~] | P3-027–P3-033 | 已生成 `results/ablation_matrix.json`，并有 `summarize_ablation_results()` 输出 complete/partial/missing、full-clear、均值、P90 和错误；真实 cell 全量运行与最终报告仍未完成。 |
| [x] | P3-034 | `way4.ALGORITHM_NAME`、`LEARNER_ROLE` 与 `DESIGN.md` 已冻结最终算法命名；仍需在最终实验报告/论文产物中沿用该名称。 |

## 当前 P0 Gate

| Gate | 状态 | 证据 |
|---|---|---|
| GATE-01 | [~] | NO_SIGNAL 已进入 `contains_possible_source`/effective area/diameter/hypothesis sampling，并接入 NBV；尚未让所有定位/geometry consumer 统一只通过该接口。 |
| GATE-02 | [~] | 已有 `CardinalityState(p,z,u,q_min,q_max)` 与强制 presence 传播；`q_min=q_max` 的完整任务闭包仍未完成。 |
| GATE-03 | [~] | 已加入 `PRESENT_UNOBSERVED`/`INITIALIZED` 并接入部分 scheduler/candidate/pipeline；所有 planner 分支尚未完全统一。 |
| GATE-04 | [~] | 已有 D、MEC、area、kappa、principal axis；coverage debt/initialization/robot distance 尚未形成统一 readiness 摘要。 |
| GATE-05 | [~] | 有 candidate/refine/clear/coverage 基础；不是五类完整 macro。 |
| GATE-06 | [~] | 已有 per-channel coverage debt/uncovered area 状态与测试；仍是 planner-only 栅格近似。 |
| GATE-07 | [~] | `AdaptiveScanSession`、`execute_adaptive()` 和 `Way4Pipeline(adaptive_scan=True)` 已存在并有测试；默认 benchmark 仍使用 legacy batch，尚缺公平评测。 |
| GATE-08 | [~] | STOP 类型和测试存在，但不等价于完整 waypoint STOP policy。 |
| GATE-09 | [x] | `core/cost.py` 与 gold timing tests。 |
| GATE-10 | [~] | certificate/mask/fallback 基础存在；全 invariant 运行证据不足。 |
| GATE-11 | [x] | `executor/homing.py`、`certificate/fallback.py`、pipeline no-progress/fallback 逻辑和对应 tests。 |
| GATE-12 | [~] | rollout/event transition 已可导出并可由独立 `TabularSMDPLearner` 消费，字段与 elapsed-time return 有回归测试；完整 event-level 线上训练证据仍不足。 |
| GATE-13 | [~] | 固定 P4 seeds 2000–2009 gate 为 10/10 full-clear；已形成稳定性证据，但仍需更大样本置信区间/多 field-kind 结果。 |
| GATE-14 | [~] | 有 `MathModelingCode/frozen/v4_n8_20260911/` 与 SHA256SUMS，但未证明项目级 freeze commit/最终 benchmark 口径。 |
| GATE-15 | [ ] | 未发现能解释 Way4 相对 V4 提升的完整消融。 |

## 验证记录

执行：`D:\Anaconda3\envs\gesture_env\python.exe -m pytest SXJM/Way4/tests -q`

结果：CUDA 手册指定环境验证通过：Python 3.10.19、PyTorch 2.1.2+cu121、CUDA available=True、NVIDIA GeForce RTX 4060 Laptop GPU；1024×1024 CUDA 矩阵运算 finite=True。随后完整 Way4 测试通过：`205 passed, 12 skipped in 570.97s`。原系统 Python 的 `c10_cuda.dll` WinError 126 已确认是解释器/环境选择问题，不再作为项目测试阻塞；后续必须继续固定使用该解释器。

## 下一步 P0 阻断项

1. 先实现 `F_eff` 的统一查询接口，并让 hypothesis/NBV/geometry/coverage debt 都通过该接口。
2. 实现 `p,z,u,q_min,q_max` 与 `PRESENT_UNOBSERVED/INITIALIZED` 生命周期。
3. 将 batch 执行改为 measurement→update→propagate→rerank，并增加显式 STOP。
4. 修复/隔离 PyTorch DLL 环境后重新跑完整测试，再做随机 soundness、full-clear 和公平消融。
