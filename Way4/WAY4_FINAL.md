# Way4-Final —— Global-Route Hybrid Planner + Safe Candidate PPO

> **状态：主线冻结（FROZEN MAINLINE）· 2026-09-12**
> 本文件是 Way4 从今天起的**唯一工程主线定义**。任何新想法，除非能在**相同 simulator、相同 seeds、100% full-clear、paired benchmark** 上打赢本方案，否则只进 `experiment/`，**不得改动主线**。
>
> **与既有文档的关系**（三者并存，不互相覆盖）：
> - `DESIGN.md` = **Layer 1 正确性契约**，冻结，不改。本文件不修改它。
> - `AUDIT_P0_ROUTING.md` = Layers 2–3 的 **A→F 落地计划**，已折叠进本文件的「确定性路由轨」。
> - `WAY4_FINAL.md`（本文件）= **架构 + 版本 + 晋级门槛 + 反模式**的总冻结。

一句话定义：

```
Way4-Final = 硬数学信念与安全约束
           + 统一全局任务路由
           + 强 Candidate-PPO 调度
```

**不是**纯数学；**不是** tiny REINFORCE；**不是** Way2 端到端 PPO；**不是**现在「NBV 先挑一个点、再让 planner 排序」的 Way4。

---

## 0. 当前实现状态（诚实对账，2026-09-12）

主线的**确定性路由半程已建到对象层，且 Phase C 接线现已解阻**；RL 半程未动；P4 经 homing(`b58008d`) 声称已达 100%，须独立重测复核。

| 里程碑 | 映射 | 状态 | 证据 |
|---|---|---|---|
| Layer 1 正确性底座 | P0-1 | ✅ **扎实** | belief / hard-certificate / clear-guard / EXIT-guard / coverage-fallback；119+ tests 绿（AUDIT） |
| 路线分解指标 | Phase A / P0-#9 | ✅ **committed** | `653869e`；`metrics.compute_route_metrics` + 9 tests |
| batch 内 STOP | Phase B / P0-#8 | ✅ **committed（默认 OFF）** | `804fb49`；`batch_stop`，禁止10 gate PASS（24/24==24/24） |
| SpatialStop 对象 + 服务打包生成器 | Phase C 对象层 / P0-#2/#3 | ✅ **committed，但未接线** | `a803f9b`；`core/actions.py:SpatialStop` + `planner/spatial.py`（HYBRID 非破坏），8 tests |
| P4 软信念假设层 `H_c` | M8 | ✅ **committed** | `885408a`；9 property tests 绿；`belief/__init__` 导出待协调跟进 |
| Way3-homing P4 fallback + 方向证书门 | M9 / sec.11 | ✅ **committed** | `b58008d`（commit 声称 P4→100%，待独立重测复核） |
| Phase C 管线接线 | P0-#2/#3 wiring | 🟢 **UNBLOCKED（可落）** | homing 已提交、`pipeline.py` 释放、worktree clean；~6 行 `planner_mode` 待落于 `b58008d` 之上 + 集成测试 |
| 统一全局路由 / ΔL 插入 / 路线感知 FutureCost | Phase D / P0-#4/#5/#7 | ⬜ **未做** | 现 `future_cost.py` 仍是可加式，腿段不共享 |
| 证书并入路线 | Phase E / P0-#6 | ⬜ **未做** | opportunistic 累积已做，dedicated 补洞未并入统一路线 |
| 关键消融（换 planner，余冻结） | Phase F / P0-#10 | ⬜ **未做** | 依赖 Phase C 接线先跑通 legacy-vs-spatial full-clear gate |
| OutcomePredictor 重写（采样，去掉 0.85/0.10/0.05） | 本文件 §9 | ⬜ **未做** | |
| R2/R3/R5 长尾护栏（WAIT gate / detour guard / hysteresis） | 本文件 §12 | ⬜ **未做** | |
| Layer 4：Candidate-PPO（Set-Transformer + masked actor-critic） | M10 / 本文件 §15–24 | ⬜ **未做** | Way2 组件待 cherry-pick |
| residual REINFORCE | —— | ✅ 存在 → **永久降级为消融 baseline** | `way4/rl/`（scorer/train/rollout） |

> ⚠️ **P4 正确性状态在变动**：SCC 报告曾记 Way4 P4 = 3/8（seeds `2000/2003/2004/2005/2007`，`2004` no_progress_stall）——那是 **homing 之前**的数字。此后 M8 假设层(`885408a`) + Way3-homing fallback(`b58008d`) **均已提交**；`b58008d` commit 声称 **P4→100%**（并发会话实测 seed2000–2007：62.5%→100%）。→ 仍须以固定 seed **独立重测** full-clear 复核该声称后再定稿；在重测前不以 3/8、也不以 100% 为最终结论。

**协作硬约束（这是执行的绑定条件）**：
- 所有 `way4` 提交**串行经过** `D:\George\way4-wt`（禁止在主树 checkout way4）；一次一个 lane 持有，只 stage **自己的显式路径**，**永不 `git add -A`**，**绝不在 peer 文件 staged 时提交**。见 `[[git-shared-repo-race]]`。
- worktree 现 **clean**（homing lane 已提交 `b58008d`，`pipeline.py` 释放）→ Phase C 接线现可安全落于 `b58008d` 之上；落前须重读当前 `pipeline.py`（已含 homing 改动）并跑测。
- **5 个提交未推送**（`653869e 804fb49 a803f9b 885408a b58008d`，origin/way4 仍 `ad58e25`）→ **用户需交互推送**：`git -C D:\George\way4-wt push origin way4:way4`（非交互 shell 有 wincredman 失败）。
- P0-A/B（belief/certificate 的 P0/P1/P2 确定性轨）在**独立分支 `way4-p012`**（sx-d3），未来合并；本主线的路由**只读**冻结的 belief/certificate。

---

## 1. 最终架构（四层，钉死）

```
Simulator observation
        ↓
[Layer 1] Set-membership belief / directional hypotheses
          Certificate + channel states
        ↓
[Layer 2] Generate ALL useful task candidates
          统一任务池：SEARCH / REFINE / CLEAR / VERIFY
        ↓
[Layer 3] Global route-aware deterministic scoring
        ↓
[Layer 4] Candidate PPO re-ranking（masked，仅在 safe 集内）
        ↓
      SafetyShield
        ↓
Execute ONE macro action
        ↓
observe → rebuild everything → replan
```

**职责分离，永不改变**：数学层负责 **Correctness**；学习层负责 **Efficiency**。

---

## 2. Layer 1 —— 数学核心全部保留（不重写）

`belief/` · set-membership `F_c` · ±1° wedge intersection · MEC/clearability · `certificate/` · hard disc-cover · cardinality invariant · P4 directional hypothesis(`H_c`) · CLEAR guard · EXIT guard · Way3 homing · fallback · executor —— **全部保留**。

职责：**告诉上层「世界可能是什么样 / 什么动作合法 / 何时一定可清 / 何时一定可退」**。
**RL 永远不能修改这些结论。** correctness > robustness > time；full-clear 比例**绝不下降**（禁止10）。

---

## 3. Layer 2 —— 统一任务池 + 多候选 REFINE + 共享 waypoint

### 3.1 推翻「单频道先选唯一 NBV」

现状 `res = nbv.choose(channel)`：把 100+ 补测点在全局路线优化前就砍成 1 个。**错误**。

改为 `NBV.generate_candidates()`，一个频道保留多个候选点：

```
Q_c = { q_MEC, q_centroid, q_ring(400/700/1000), q_⊥, q_future-search, q_route-near, q_other-clear, q_certificate }
```

NBV 的 minimax 信息质量仍有用，但降级为 **feature**：`U(q)=worst-case diameter after measurement`，**不再** `q*=argmin U(q)` 直接替全局 planner 决策。

### 3.2 统一任务图（本轮最重要结论）

机器人只有一条腿，只走一条路线。放弃 `Route(CLEAR)+Route(LOC)+Route(CERT)` 的分项相加，统一为：

```
Route( SEARCH ∪ REFINE ∪ CLEAR ∪ VERIFY )
Task = (type, point, channels, value, constraints)
```

一个 waypoint 可**同时完成多个服务**：到 `q` 可以 `REFINE ch4 + REFINE ch7 + VERIFY ch12 + SEARCH ch3,8,11`，生成**一个** `Task(q,{4,7,12,3,8,11})`，而不是 6 个独立移动任务。
> 落地对象：`SpatialStop{location, services:set[(kind,channel)], clear_channels, clear_targets}`（已建于 `a803f9b`）。

### 3.3 共享 waypoint —— 先问「这点能服务多少任务」

```
Service(q) = { c : q 对频道 c 有价值 }
V(q) = Σ_c V_c(q)     价值来自 IG_c / CertificateGain_c / DiscoveryGain_c / ClearOpportunity_c
C_move(q) = ‖q − x_t‖ / 5     移动只付一次
```

这才真正利用「20 频道共享空间路线」的结构。

---

## 4. Layer 3 —— 全局路由是 P3 的主优化器

P3 第一目标不是 `max IG`、不是 `max P(clear)`、不是 `min MEC`，而是：

```
min T_finish ,  T = T_move + T_measure + T_switch + T_clear
```

实验已证 `T_move` 占绝对大头 → **路线是主目标**。

### 4.1 吸收 V6 最有价值思想：候选的路线边际

对某候选 `q`，不只算 `IG(q)`，而是相对当前任务集 `T` 的插入增量；若替换频道 `c` 原 REFINE 点，算替换增量而非简单插入：

```
ΔC_route(q)   = C(T ∪ {q})     − C(T)
ΔC_route(q,c) = C(T_{-c} ∪ {q}) − C(T_{-c})
```

### 4.2 最终数学 candidate score（先做强确定性基线）

```
J_math(a) = ΔC_route(a) + C_service(a) + E[C_downstream | a] + P_reorder + P_risk
```
1. **ΔC_route** —— 最重要项（路线边际）。
2. **C_service** —— `5s × N_measure` + 频道切换时间。
3. **E[C_downstream]** —— 测完后剩余定位+清除时间。
4. **P_reorder** —— 防止「当前点便宜却把后续十个任务顺序搅乱」（V6 病）。
5. **P_risk** —— 尾部惩罚，防 +500s 长尾。

### 4.3 Route solver

- 任务数 `n ≤ 12`：**Held-Karp 精确开放路径**（open-path TSP）。
- 任务多：`nearest neighbor / cheapest insertion / 2-opt / Or-opt`，**多初始解**取最好（不再只从一个 heuristic 出一个解）。
- route cost 含频道顺序：`C_route = L/5 + 5·N_measure + N_switch + C_clear`。

---

## 5. OutcomePredictor 重写（§9）——去掉写死概率

从核心 planner 拿掉 `P(detect)=0.85 / P(near)=0.10 / P(no_signal)=0.05`。

**P3**：从真实 feasible polygon `F_c` 采样 `p_1..p_N`（`N=12~32`）；对候选 `q` 算真 bearing `θ(q,p_j)`，枚举 `ε∈{−1°,0,+1°}`，更新 `F'_c = F_c ∩ W(q,θ+ε,±1°)`，重算 `MEC(F'_c)` → 得 `P_clear(q)`、`E[r'_MEC|q]`、`E[C_remaining|q]`。

**P4**：NO_SIGNAL 不能简单 `F_c \ B(q,1000)`；保持 `Hard positive set-membership + Soft directional hypothesis(H_c)` 隔离。route scoring 可用 `H_c=(position,heading,type)` 预测 `P(signal|q,H_c)` 与 visibility gain。
> **Soft hypothesis 永远不能用于 CLEAR / ABSENT，只用于规划评分。**（保 Layer 1 正确性）

---

## 6. 长尾护栏（吸收 V6 已暴露的三坑，§12）

- **R2 过度等待未来 SEARCH**：设 `WAIT_SEARCH` gate。仅当 `ΔC_route(q_s) < ΔC_route(q_d)+τ` **且** `IG(q_s) ≥ ρ·IG(q_d)` 才 WAIT；否则**不等，直接 REFINE**。初值 `τ=30~60s`，`ρ=0.6~0.8`，最终验证集选。
- **R5 任务排序震荡**：route hysteresis。新 route 仅优于旧 route `< δ`（`20~40s`）时，保持当前前 1–2 个任务；明显改善才重排。
- **R3 过度 detour**：baseline regret guard。若 `J(q) − J(q_baseline) > τ_detour`（`100~150s`）直接 fallback。杜绝 500s 级疯狂回退。

`future_cost.py`：**不删**，从核心决策器**降级为 feature + auxiliary estimator**（继续提供 route/loc/cert 估计，但不再单独主导 ranking）。

---

## 7. Layer 4 —— 最终 RL：Masked Candidate Actor-Critic PPO

**不是 tiny MLP，不是端到端。** 数学层产生 `A_safe(B)`；RL 只能选 `a ∈ A_safe(B)`，但在集合内**不给 RL 人为降智**。

### 7.1 网络（钉死；组件源自 Way2，但不搬旧 action logic）

- **Channel encoder**：20 频道当无序 set，每频道特征 `x_c=[status, mec_radius, diameter, center-relative, bearing_count, no_signal_count, coverage, certificate, clearable, age, ...]`；P4 增 `[directional entropy, visibility ratio, heading uncertainty, omni/dir prob]`。`h_1..h_20 = Transformer(x_1..x_20)`，`d_model=128, heads=4, layers=2~3`（**不再限 (64,64)**）。
- **Candidate encoder**：每候选 `[action type, target rel x/y, distance, service channels, route marginal, insertion pos, math score, immediate time, P_clear, expected MEC, IG, cert gain, discovery gain, future cost, detour regret, wait flag]` → `g_i`；`z_i = Attention(g_i, H)` → `π_θ(a_i|B)`。
- **来源**：cherry-pick Way2 的 `channel_transformer` / `PPO trainer` / `pluggable encoder` 组件；**丢弃** Way2 旧 action logic。

### 7.2 数学是先验不是牢笼（Math prior, not Math prison）

```
l_i^(0) = −norm(J_math(a_i))          # logits 初始化
l_i     = −α·J_math(a_i) + f_θ(B,a_i) # f_θ 不是 tiny residual，可大幅改排序
```

### 7.3 Reward（钉死）

```
r_t = −Δt                             # 题目最终就是时间
```
IG / clear prob / MEC shrink **不当主 reward**，最多做 potential shaping：
```
r'_t = −Δt + η[Φ(s_{t+1}) − Φ(s_t)] ,  Φ(s) = −J_math(s)   # 仅训练加速，不带偏总目标
```
**字典序目标**：① `P(full_clear)=1` ② `min T`。失败 `R=−M`（`M ≫ 360000` 或足够大 normalized penalty），成功 `R=−T`。**绝不能让 15/16 与 16/16 只差一点 reward。**

### 7.4 训练

- `actor-critic + GAE + advantage norm + PPO clip + entropy bonus + grad clip + value loss`；不再 REINFORCE-only。
- 起始：`clip=0.2, GAE λ=0.95, entropy≈0.01(衰减), value_coef≈0.5, grad_clip≈0.5`；有限时域真实目标是 undiscounted completion time → `γ≈1`。
- **Warm start（不是限制）**：初期 `π_θ≈π_math`（logits 初始化 / 短暂 BC / 前期 KL `D_KL(π_θ‖π_math)` 逐渐衰减），训练后**允许真正超过 Math**。
- **P3/P4 分模型/分 head**：共享 backbone 可，但分别 fine-tune（P4 多 directional visibility / heading hypothesis / ambiguous no-signal，状态结构不同）；可 `P3→P4 fine-tune`，但不为「统一」牺牲性能。

---

## 8. P4 正确性（旧 3/8 → homing 后声称 100%，待复核；不允许扔给 RL）

必须逐个 trace 失败 seed `2000/2003/2004/2005/2007`（尤其 `2004 no_progress_stall`），把 **P4-Math 先做到 8/8**，再扩 `50→200→stress` 保持 100% full-clear，才能把 RL 模型列入最终候选。**但这不阻塞 P3 route 改造与 P3 PPO —— 两条线并行。**
> 注：M8(`885408a`)+homing(`b58008d`) 均已落定，`b58008d` 声称已把这些 seed 清到 100%；**先独立重测**复核，再决定 P4-Math 是否还有确定性缺口要补。

---

## 9. 版本体系（钉死）

| 名称 | 用途 |
|---|---|
| Way3 | 老安全基线 |
| Way4-Legacy | 当前数学版本 |
| Way4-Route-Math | 新统一全局路由 deterministic |
| Way4-REINFORCE | 现有小 residual，**消融** |
| **Way4-Final-PPO** | **最终强模型** |

不再有「可能 residual / 可能 PPO / 可能 Way2 / 可能又改」。主线只有 **Way4-Final-PPO**。

---

## 10. 统一晋级门槛（钉死）

**第一条永远是 `FullClear = 100%`**；不满足 → 平均时间全部作废。之后才比 `Mean / Median / P90 / P95 / Max / Movement / PairedWinRate / RegressionTail`。特别记录 `>100s regressions`、`>300s regressions`、`worst paired regression`。
**杜绝**「平均快 2%，但十几个 case 慢 500s，还说升级」。

保留 evaluate 分支的 `LB / Oracle / Math / Hybrid`；核心指标之一：
```
R_LB = T_algorithm / T_lowerbound     # 否则 3500s 到底好不好没有尺度
```

---

## 11. 从今天起彻底停止（反模式）

- ❌ 单频道先选唯一 NBV。
- ❌ `J_route + J_loc + J_cert` 分路线相加当主模型。
- ❌ 固定 `0.85/0.10/0.05` outcome probability。
- ❌ tiny REINFORCE 当最终 RL。
- ❌ Way2 full end-to-end PPO 当最终主算法。
- ❌ 为「稳」而禁止 Transformer/PPO。
- ❌ RL 任意输出坐标。
- ❌ 为平均值降低 full-clear。
- ❌ P4 用 soft directional hypothesis 做 hard absent。

---

## 12. 冻结开发清单（可直接执行；标注真实状态）

**确定性路由轨（先做，RL 之前必须先赢）**
- [x] 冻结 baseline/seeds（Phase A 记录 legacy 12-seed 基线；`BASELINE.md`）
- [x] `SpatialStop` 对象 + 服务打包生成器（`a803f9b`，未接线）
- [x] batch 内 STOP（`804fb49`，默认 OFF）
- [ ] 🟢 **Phase C 接线**：`planner_mode:str="legacy"` + generator 选择（spatial→`SpatialStopGenerator(base)`）+ 集成测试 —— **已解阻**（homing `b58008d` 已提交、worktree clean）；落于 `b58008d` 之上
- [ ] `UnifiedTask/TaskPool` 正式统一四类任务
- [ ] `NBV.generate_candidates()` 一频道多 REFINE 点 + 未来 SEARCH/其他 clear/certificate/route-near 共享 waypoint
- [ ] `GlobalRoutePlanner`（统一优化，替代分项 future cost）
- [ ] candidate replacement route marginal `C(T_{-c}∪q) − C(T_{-c})`
- [ ] Held-Karp（`n≤12`）+ multi-start insertion + 2-opt/Or-opt（大集）
- [ ] 重写 OutcomePredictor：P3 feasible-set 采样 + ±1° 枚举（去掉写死概率）
- [ ] `future_cost.py` 降级为 auxiliary feature
- [ ] `WAIT_SEARCH` gate / route hysteresis / detour regret guard
- [ ] 至少 200 个同 seed P3 benchmark（不明显超过当前 Way4 就**不碰 RL**）
- [ ] P4 五个失败 seed 逐个修 deterministic gap → 8/8 → 扩量保持 100%

**学习轨（deterministic 赢了之后）**
- [x] 保留 residual REINFORCE，标记 `ablation-baseline`
- [ ] 从 Way2 cherry-pick Transformer encoder + PPO trainer 组件（不搬旧 action logic）
- [ ] `CandidateActorCritic`：Set-Transformer + candidate encoder + masked actor + critic
- [ ] PPO action 永远只选 `safe candidate mask=True`
- [ ] reward 主体 `−Δt` + 失败强 terminal penalty
- [ ] math score 作 prior/input，不作硬排序
- [ ] P3/P4 分模型或分 head 训练

**评测轨（贯穿）**
- [ ] 统一评测脚本：一次输出 success/time/move/measure/switch/P95/max/paired regression/LB ratio
- [ ] 训练集 seeds 禁止出现在最终 test
- [ ] 每版本 paired same-seed 对比（不再只看平均）

---

## 13. 工作顺序（钉死）

```
确定性路由轨：
 1. Phase C 接线（**pipeline.py 已释放，可落**）  →  2. UnifiedTask  →  3. REFINE multi-candidate
 →  4. GlobalRoutePlanner  →  5. sample-based outcome  →  6. 200-seed deterministic benchmark
     ↳ 若不能明显超过当前 Way4：不碰 RL，先把 route planner 修对。

并行线：P4 —— homing(`b58008d`) 已声称 100%，先独立重测复核；若仍有确定性缺口再逐 seed 修。

学习轨（deterministic 赢了才进）：
 7. Way4-Final-PPO（复用 Way2 Transformer/PPO 组件）  →  8. 大规模训练
 →  9. 独立 test set  → 10. Math vs REINFORCE vs PPO 消融。
```

---

## 14. 一句话总纲

最初的错误：**先优化「每个任务怎么做好」，再拼路线。**
现在改成：**从一开始就优化「整局走哪条路线」，定位/搜索/证书/清除都只是这条路线上的服务任务。**

```
Way4-Final = [Hard Belief + Certificate + Safety]  (数学保证)
           + [Unified Global Routing]              (主要性能来源)
           + [Transformer Candidate-PPO]           (长期调度与非线性修正)
```
