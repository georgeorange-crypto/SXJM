# Way4 P0 验收审计 — Space-Centric Multi-Service Routing

**日期**：2026-09-12 · **审计对象**：`way4` @ `ad58e25`（主树 `D:\George\SX\Way4\` dev 副本，与已提交内容一致）
**基线状态**：`python -m pytest -q` → **119 passed in 9.54s**（正确性底座绿）
**结论一句话**：正确性底座（belief / hard certificate / clear-guard / EXIT-guard / coverage fallback）**扎实**；但**「空间停靠点 → 多服务打包 → 统一路线 → 机会式证书 → 路线感知未来成本」这条链尚未存在**。当前 planner 是 **action-centric**（每 tick 选一个宏动作执行再重规划），未来成本是**可加的**（`J_route+J_localization+J_exploration+J_certificate` 各自从当前位置起路由再相加）。这正是本轮 P0 清单要攻的结构性瓶颈。

> 本文是验收报告 + 实施计划草案，**不修改冻结的 `DESIGN.md`**。SpatialStop 架构是 DESIGN.md 之外的新方向，落地需遵守两条硬约束：**P0-1 冻结正确性底座** 与 **P0-10 消融要求（只换 planner，其余冻结）**。

---

## 逐项验收表

| # | P0 项 | 状态 | 证据（file:line） |
|---|---|---|---|
| 1 | 冻结正确性底座 | ✅ **成立** | clear 仅来自 `clearable_channels()`（candidates.py:91）；EXIT 需 `resolved==n` 且 `_exit_allowed` force 跑三来源（pipeline.py:137,242-250）；三来源析取 + Invariant A/B/C/D 写点纯净（manager.py:128-249）；Way3 legacy backbone + COMPLETION 兜底（candidates.py:209-266）；119 tests 绿 |
| 2 | SpatialStop 为基本对象 | ❌ **未做** | planner 基本对象是 `MacroCandidate = action_type + 单 target + scan_channels + clear_channel`（actions.py:53-72），彻底 action-centric；代码中不存在 location-first 的「停靠点 + 可完成服务集」对象 |
| 3 | Multi-Service Bundling | 🟡 **部分** | **已有**：同类频道在一个 waypoint 批量扫（scheduler.py:138-202；REFINE 用 `must_include=[c]` 带 free-rider，candidates.py:125-132；EXPLORE 批多个 UNKNOWN）。**未做**：(i) CLEAR 无法进入扫描 bundle —— `primitives()` 要么全 MEASURE 要么单 CLEAR（actions.py:74-84）；(ii) bundle 位置由**单一 focus 动作**决定（频道 c 的 NBV 点 / 环点 / anchor），不是「选一个能完成最多服务的地点」；(iii) 没有「一地点打包 REFINE ch4 + REFINE ch6 + VERIFY ch1 + SEARCH ch17 + CLEAR ch9」的概念 |
| 4 | 统一剩余任务路线 | ❌ **未做** | `estimate()` 把 j_route（clearables Held-Karp）、j_localization（detected TSPN）、j_certificate（holes greedy-cover）**各自从 `view.pos` 起路由再相加**（future_cost.py:174-221），腿段不共享；且 planner 每 tick 只选**一个**宏（pipeline.py:239-240），路线只隐式来自贪心一步 Ĵ 最小化 |
| 5 | CLEAR 入池 + 插入增量 ΔL | ❌ **未做** | CLEAR 是独立候选按 `Q=C+Ĵ` 竞争；无 `ΔL=d(a,c)+d(c,b)−d(a,b)` 顺路插入。「尽早生成」半边有（clearable 即生成，candidates.py:89-102），「路线决定何时执行」半边无（没有可插入的路线对象） |
| 6 | Certificate opportunistic-first | 🟢 **累积已做** / 🟡 **路由未做** | **已做**：任何真实 NO_SIGNAL（不论扫描目的）都累计覆盖（manager.py:128-164），候选把 coverage_gain 当 free-rider，dedicated VERIFY 仅在 VERIFICATION/COMPLETION 出现 —— 「机会优先、末尾才补洞」基本就是现设计意图。**缺口**：dedicated 补洞成本作为**独立可加项** J_certificate（与 #4/#7 重叠），未并入统一路线 |
| 7 | 重写 FutureCost（次可加联合路线） | ❌ **未做** | 即上面的可加式 `total = w_route·j_route + w_loc·j_loc + w_expl·j_expl + w_cert·j_cert`（future_cost.py:179-185）；无「剩余停靠点联合路线长」估计，无法体现 `C(A∪B)<C(A)+C(B)` |
| 8 | batch 内动态 rerank + STOP | ❌ **未做** | `MacroExecutor.execute` 按固定顺序跑 `macro.primitives()`（macro_executor.py:95-119），每 primitive 折进 belief 但**不重排剩余频道、无 VOI STOP**；唯一的 STOP 是时间预算，唯一的 mid-batch 反应是 `near` 机会清除。（= 记忆里的 P0-D） |
| 9 | 路线诊断指标 | ❌ **未做** | 代码中不存在 `pure_refine_travel / certificate_only_travel / revisit_distance / shared_stop_ratio / services_per_stop / clear_insertion_delta`（grep 仅 DESIGN.md 命中其它词）。现有仅 move/measure/switch/clear 计数（pipeline.py:47-66）与 `certificate.metrics()` 的 active_scan fraction |
| 10 | 干净关键消融（旧 vs 新 planner，余冻结） | ❌ **未做** | 新 planner 不存在，故消融无从谈起；`compare_way3_way4.py` 比的是 Way3 stack vs Way4 stack，不是 Way4-旧planner vs Way4-新planner；DESIGN §15 列的消融开关（`future_route_cost / multipurpose_waypoints / …`）**未接线** |

**额外发现（重要）**：`MacroCandidate` 的分解增益向量在当前 planner 里**基本是死重** —— `receding_horizon.plan()` 的 `Q=expected_time+Ĵ` 只读 `expected_time`，增益里**仅 `refinement_gain` 被用到**（receding_horizon.py:138，且只用于 outcome 预测的 region shrink）；`exploration_gain / certificate_gain / route_gain` 从不进入打分。选择完全由 OutcomePredictor 对 CostView 的 Ĵ-delta 驱动，而非 DESIGN §6.6/§8 写的 `Utility = w_E·explore + w_L·loc + …`。→ 若新架构要「每停靠点 Utility = Σ services」，这套增益组合必须**真正驱动选择**。

---

## 结构性瓶颈（三合一）

1. **Action-centric 候选** → planner 每 tick 只能选**一个**动作，永远无法「组合」成一个多服务停靠点。
2. **可加 Ĵ** → 未来成本看不到腿段共享（顺路清/顺路补洞的红利没进信号）。
3. **无显式停靠点/统一路线对象** → 路线只是贪心一步决策的副产品，`ΔL` 顺路插入无处安放。

三者耦合：即使把 Ĵ 改成联合路线，只要候选仍是 action-centric、每 tick 选一个，路线红利也落不了地。所以 **#2（SpatialStop）是链条的地基**。

---

## 实施计划草案（尊重冻结 + 消融要求）

> 原则：**correctness > robustness > time**；每步**behind an ablation switch**（`planner=legacy|spatial`）；**full-clear 比例绝不下降**（禁止10）；**belief / NBV / certificate / guards 全程冻结**（P0-1 + 消融10）。

- **Phase A｜先上指标（#9）** — 给 `EpisodeResult`/pipeline 加路线分解指标（services_per_stop、pure_refine_travel、certificate_only_travel、revisit_distance、shared_stop_ratio、clear_insertion_delta）。**低风险、纯增量**，且为 #10 提供「旧 planner 现状数字」，先量化浪费再改。
- **Phase B｜batch rerank + STOP（#8）** — 在 executor/薄控制器里做 batch 内 VOI 重排 + STOP（测一个 → 更新 belief → 重排/决定 next|STOP）。**收敛的已知缺口**，只丢多余扫描、绝不跳过必需清除、guard 不动。
- **Phase C｜SpatialStop + 服务打包（#2/#3）** — 引入 `SpatialStop{location|neighborhood, services:set[(kind,channel)]}`；generator 产候选停靠点，每点算出可完成的全部服务（含 CLEAR）。behind `planner=spatial` 开关，legacy 并存供消融。
- **Phase D｜统一路线 + ΔL 插入 + 路线感知 Ĵ（#4/#5/#7）** — 对剩余停靠点跑统一 route heuristic（NN + insertion + 2-opt / 复用 TSPN）；CLEAR 用 `ΔL` 顺路插入；Ĵ 改为估「剩余停靠点联合路线长」（次可加）。
- **Phase E｜证书并入路线（#6）** — opportunistic-first；dedicated 补洞点只在末尾用同一 route optimizer 生成。
- **Phase F｜关键消融（#10）** — 冻结 belief/NBV/cert/guards，仅换 planner，在 offline_sim / Way3 engine 固定 seed 集跑；报新指标 + full-clear（不得降）+ 时间。

**建议顺序理由**：A→B 先把「可测量 + 已知缺口」清掉（低风险、立即给消融打表），再进 C→D→E 的架构主体，F 收口。

---

## 待用户拍板（scope check-in）

1. **自主度**：按「一口气执行所有里程碑」授权把整链自主实现，还是**逐 Phase 汇报/确认**？（本方向不在冻结 DESIGN.md 内，倾向逐 Phase 确认）
2. **是否先做 Phase A（指标）** 量化当前 planner 的绕路浪费，再动 planner？（**建议是** —— 让 #10 有意义、并 de-risk）
3. **新旧 planner 并存**（`planner=legacy|spatial` 开关）确认？消融 #10 需要两者共存。
4. **提交节奏**：每 Phase 经 `D:\George\way4-wt` worktree 提交到 `way4` 分支（禁止在共享主树 checkout）。
