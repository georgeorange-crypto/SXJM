# Way4 冻结设计基线 (Frozen Design Baseline)

## 最终算法命名

论文与评测统一使用 **Certificate-Guided Active Belief Planning**（基于集合
信念与完备性证书的主动滚动规划）。RL 组件统一称为 **Learning-Augmented
Planner**，不将最终算法命名为 PPO Method。

**方法名**：Set-Membership Belief-Based Active Multi-Target Search with Certificate-Constrained Adaptive Routing
**目标分支**：`way4`（从 `origin/way3` 派生，见 §2）
**本文状态**：设计冻结基线，**已审阅通过并 Greenlight（2026-09-11）**。所有物理/协议数字引自 `附件2.txt` + `B题(1).pdf`（行号标注），已与仓库现状对账（§2）。**DECISION-1=采纳**（P3 任意扫描点 hard 证书，§6）；**DECISION-2=Greenlight**（授权按 §6 尾实现顺序开工，§17）。审阅补入的三项修订（Conservatism note A/B、Resolution+rescue、Cardinality Invariant D）已写入 §6。

> 本文是 Way4 的单一事实来源，用于取代散落在 `建议.docx` / 聊天记录里的设计。它不重抄公式推导，只固化「可实现的规格 + 与代码现状的对账 + 里程碑」。

---

## 1. 权威模型与协议事实（来自 `附件2.txt`，勿再猜）

| 项 | 值 | 出处 |
|---|---|---|
| 目标区域 | 半径 **1800 m** 圆盘，圆心 (0,0)，x 东 y 北 | line 6,8,9 |
| 源位置域 | **所有源在圈内** `‖p‖≤1800` | line 7 |
| 机器人动作域 | **可出圈**；仅要求每分量有限且 `|coord|≤2,000,000`（越界/NaN/Inf → HTTP 400 不执行、不推进时间） | line 12,13 |
| 频道 | 整数 `{1..20}`，每频道 ≤1 源 | line 24,25,26 |
| 源总数 | **10–16 个**，接口不返回，需自推 | line 27,28 |
| 有效接收半径 R_eff | **[1000,1500] m，逐源不同、不返回** | line 47,49,50 |
| 全向源 | 距离 ≤ R_eff 即可检测 | line 53 |
| 定向源 | 距离 ≤ R_eff **且**落在 **180°** 覆盖弧内（含边界），**方向未知** | line 54 |
| no-signal 三成因 | ①该频道无未清源 ②距离>R_eff ③定向源但不在覆盖弧 | line 55–58 |
| 示向度误差 | svd_deg ∈ 真方位 ±**1°**，保留两位小数；不是精确真方位 | line 60,62,63 |
| 近距离 near | 距离 ≤**5 m 且在覆盖弧内** → 信号过强，返回 `near`，**不返回 svd_deg** | line 68 |
| 清除半径 | **20 m**；`/clear` 成功仅取决于清除点到源距离 ≤20，**与定向朝向无关** | line 69 |
| 移动 | `t=dist/5`（5 m/s） | §4.2 |
| 检测 | 5 s；换频道 +1 s | 建议.docx / Way2 constants |
| 清除耗时 | 未命中（仅光学定位）**3 s**；成功（定位+清除）**5 s**；`/clear` **不产生切频耗时、不改当前频道** | line 124,125,126 |
| 时限 | 真实运行 ≤**20 min**（用返回的 `remaining_real_duration_s∈[0,1200]`，勿假定 1200）；虚拟上限 360000 s | line 129,130 |

**⟨已闭合-A｜B题(1).pdf 附录2 原文⟩**：「在一段时间内，**同一地点的电磁环境干扰是固定的，所以重复检测不会改变检测误差**。只有在不同地点、不同电磁环境下，误差才会呈现统计规律。」→ 误差**逐点确定**（不是随机噪声）：同点重测得到**完全相同**的示向度、零新信息 → `禁止4`（同点重复测量求平均）**成立且已坐实**。**设计含义**：独立的第二条 bearing 必须**换到不同地点**取得；±1° 楔宽是每个观测点的**固有量**，无法靠平均收窄 → 直接支撑 set-membership 楔形与 minimax NBV（每条新 bearing 只能来自新位置，∴ NBV 选点即选信息）。

**其它已从题面正文/附件1坐实的协议细节**（补入冻结基线）：
- **清除确定性**（附录2(8)/附件1 line58）：20 m 内**必定**成功定位并清除，**无概率**；同一源只能清一次，重复 `/clear` 返回 `no_target_in_range`。
- **示向度定义**（附录2(1)）：`svd_deg` = 从 +x 轴逆时针到「检测点→干扰源」向量的角度 ∈ [0°,360°)，即 `wedge_halfplanes` 用的「观测点→源」方位。
- **定向覆盖**（附件1(3)）：以**定向发射方向**为中心、两侧各 90°（含边界）= 180° 半平面；覆盖内信号仅随距离衰减、与角度无关。
- **/enter 返回本次实际剩余程序运行时间**（正文 page4）：早调用最多 20 min，晚调用受 25 min 窗口挤压 → planner 必须读取该返回值，勿硬编码 1200 s。
- 初始状态 (0,0)、初始测向频道 1（附件1 line4）。

### 1.1 计时黄金样例（附件1 表2 — M4 验收 oracle）
从 (0,0) 出发、初始频道 1：

| 步 | 指令 | 位置 | 频道 | 移动 m | 移动 s | 切频 s | 检测/清除 s | 步耗 s | 虚拟时刻 s |
|---|---|---|---|---|---|---|---|---|---|
| 1 | /enter | - | - | - | 0 | 0 | 0 | 0 | 0 |
| 2 | /measure | (300,400) | 1 | 500 | 100 | 0 | 5 | 105 | 105 |
| 3 | /measure | (300,400) | 2 | 0 | 0 | 1 | 5 | 6 | 111 |
| 4 | /clear | (300,0) | 3 | 400 | 80 | 0 | 3（未发现） | 83 | 194 |
| 5 | /measure | (300,0) | 2 | 0 | 0 | **0** | 5 | 5 | 199 |
| 6 | /exit | - | - | - | 0 | 0 | 0 | 0 | 199 |

关键点（§7 成本模型必须复现）：步5 切频 = **0** —— 步4 `/clear ch3` 只指定「要清除的源频道」，**不切换、也不改变测向机当前频道**（仍是步3 的 ch2），故步5 测 ch2 无需切频；初始频道=1 使步2 测 ch1 也无切频。

---

## 2. 与仓库现状的对账（已核实）

**分支谱系（远端权威 heads）**
- `origin/way3 = 6b8072f`，树内已含 `estimator + Way2 + Way3 + offline_sim + B题(1).pdf + 附件1/2 + 思路.md` → **way4 从 origin/way3 派生**（三条线的汇合点）。
- `origin/way3` 的 `Way2/` 停在 Milestone-1（`9b32d84`）；本地 `way2` 已到 **M3 `67faa52`**（M2 `be15090` PPO、M3 pluggable stack）。→ way3 的 Way2 落后 M2+M3，**RL 阶段（M10）再 cherry-pick**，Way4-Math 不受影响。
- **绝不能在共享工作树 `git checkout way3/way4`**（会删磁盘上 Way2、抽走其它 session 的 HEAD）。way4 用 `git worktree add` 建（同 way3 当初做法）。属实施动作，设计冻结后再执行。

**estimator 源码位置**：当前 way2 磁盘上 `estimator/` 只剩 `__pycache__/*.pyc`；但源码 committed 在 `origin/way3` 与 `origin/p1p2`（`geometry.py / problem1.py / problem2.py / figures.py / __init__.py / test_estimator.py`）。→ 从 way3 建 way4 即带回源码。API 已从 bytecode 核实无误。

**几何库现状 = 4 套并存**（`禁止2` 比设计写的更重）：
1. `estimator/geometry.py`：`norm_deg / wedge_halfplanes / Halfplane / _clip_polygon(S–H) / halfplane_intersection / convex_hull(Andrew) / polygon_diameter / diameter_circle_covers(Thales) / min_enclosing_circle(Welzl)`
2. `Way2/src/radio_rl/geometry/region.py`（自带全套，`belief.py` 走 `from .region import Region`，**不** import estimator）+ `information_gain.py`
3. `Way3/jammerhunt/geometry.py`
4. `Way1/radio_locator/src/geometry/`（`convex_hull/polygon_clip/rotating_calipers/mec`）
→ M1 目标：抽出唯一 `sxjm_core/geometry/`，四处改为 `from sxjm_core.geometry import *` 兼容 wrapper。

**Way2 框架断言全部 MATCH**（可直接作为 Way4 母体）
- `Pipeline`：`env.reset → belief.update → generator.generate → feature.build → agent.select → shield.apply → env.execute`
- `ChannelStatus = {UNKNOWN, DETECTED, LOCALIZED, CLEARED}`；`ChannelBelief/BeliefState`
- MEC clear：`region.min_enclosing_circle()`，`localize_mec_radius=18`（即当前清除阈值取 18，比 20 留 2 m 余量）
- 正区域 `Region` + 独立 `ExclusionDisc` 列表（正/负分离，符合设计）
- `CandidateGenerator.generate(belief)→CandidateSet`（`k_max=32`、index-only、带 mask）；item = `CandidateAction`
- `AnalyticalWorldModel`（`predict_time` move=d/5 +switch1 +detect5 +clear hit5/miss3；`predict_clear_probability`；`predict_region_reduction`）
- `core/`：`TaskConstants CONSTANTS`（region 1800, receive 1000/1500, bearing 1.0, source_count **10–16**, channel_max 20, clear 20, speed 5, detect 5, switch 1, too_strong 5, clear hit5/miss3, **dir_half 90**）+ `datatypes.py` + `config.py`(OmegaConf) + `registry.py`

**已确认的 Way2 缺陷（Way4 必修）**：candidate generator 把坐标 clip 回 1800 盘 → 与 `附件2` line 12 冲突，**丢失 Q4 外圈扫描点**。Way4 分离 source/robot 域（§7）。

---

## 3. 三个空白的正式闭合（本次「钉死」的核心）

### 3.1 Gap-1 — Q4 隐藏状态 H_c 的数据结构
**双层信念，职责严格分离：**

- **精确正层（HARD，驱动 clear 与证书）**：与 P3 相同，`F_c ⊂ ℝ²` 凸区域
  ```
  F_c = D_1800 ∩ ⋂_i W(S_i,θ_i,±(1°+ε_num)) ∩ ⋂_i B(S_i,1500)
  negative_discs: list[ExclusionDisc(S, 1000)]   # no-signal 排除，非凸，单独存
  ```
- **粗假设层 H_c（SOFT，仅供 Q4 规划打分，永不用于 clear/absent）**：网格 mask
  ```
  spatial cells:  60 m 网格，仅保留中心 ∈ D_1800 的格 (~2800 格)
  heading bins:   36 个 × 10°（仅 τ=directional 有意义）
  type:           {omni, directional}
  hypothesis h = (cell, heading_bin, type)，状态 alive/eliminated
  初始：每 cell 作为 omni alive；每 (cell,heading) 作为 directional alive
  ```
  用途：Q4 no-signal 信息价值、visibility 规划、朝向推断、planner 打分。**不得据此判 absent。**

### 3.2 Gap-2 — 定向可见性 V(·) 与负信息（题面 line 54 直接定义）
```
would_definitely_detect(h=(p,φ,τ), S):   # 用保证下界 R_eff≥1000
    if τ == omni:         return ‖S−p‖ ≤ 1000
    if τ == directional:  return ‖S−p‖ ≤ 1000 and bearing(p→S) ∈ [φ−90°, φ+90°]  # 含边界
```
- **NO_SIGNAL@S**：删除所有 `alive h` 满足 `would_definitely_detect(h,S)`。**必须用 1000，不用 1500**（R_eff 未知，仅 ≤1000 保证被检测；line 47–50）。
- **正检测（bearing θ）@S**：精确层按 P3 更新 F_c；假设层保留满足 `p∈W(S,θ,±1°)∩B(S,1500)` 的 cell，且若 τ=directional 还须 `S` 落在以 p 为顶点、朝 φ 的 180° 弧内（该源确实可能朝 S 发射）。
- **near@S**（≤5 m 且在弧内，line 68）：视为最强正信息，触发 §11 opportunistic clear。

### 3.3 Gap-3 — 定向覆盖证书 与 clear 证书
- **CLEAR 证书（与朝向无关，line 69）**：`clearable(c) ⇔ MEC(F_c).r ≤ clear_threshold`。达标即可在 MEC 圆心盲清，**全向/定向同一规则**（清除只看距离）。F_c 由 bearing 定位，与类型无关。`clear_threshold` 取 Way2 现值 **18**（20 留 2 m 余量）或配置项。**清除是确定性的**（附录2(8)/附件1 line58：20 m 内必定成功、无概率）→ `MEC.r≤18<20` 时在圆心盲清**保证命中**（真源距圆心 ≤ MEC.r ≤ 18 < 20），非概率事件。
- **ABSENT 证书（分 P3/P4）**：见 §6。定向 absent 用「三角形/角隙」几何：对任意潜在 G∈D_1800，在 `‖S_i−G‖≤1000` 的已完成扫描点中，**从 G 看的最大角隙 < 180°**（`建议.docx` line 2520；Way3 `verify_three_cover`）。因近边界的定向源可能只能从**圈外**探到 → 需外圈扫描点（∴ 机器人必须允许出圈，line 12 已证实）。

---

## 4. 统一信念层
- P3：每频道 `ChannelBelief = (status, F_c, negative_discs, bearings, near_points, mec, diameter, area, scans, last_scan_time)`，`status∈{UNKNOWN,DETECTED,LOCALIZED,CLEARED, ABSENT_CERTIFIED}`（在 Way2 四态基础上加 `ABSENT_CERTIFIED`；**不引入概率型 absent**）。
- Q4：追加 §3.1 的 H_c 粗假设层。
- 底层几何统一走 `sxjm_core.geometry`（M1 后）。

## 5. 观测更新
- **P3 positive**：`F_c ← F_c ∩ W(S,θ,±(1°+ε_num)) ∩ B(S,1500)`；重算 `mec/diameter`；`mec.r≤thr` 置 `clearable`（不自动 CLEARED）。
- **P3 no-signal**：`negative_discs.append(ExclusionDisc(S,1000))`（正区域凸性不动，规划时判候选是否落入排除盘）。
- **Q4**：§3.2。精确层同 P3；假设层做正/负更新。

## 6. 覆盖证书（Certificate Manager，per-channel）— P3 任意扫描点 Hard Disc-Cover【DECISION-1 已采纳并细化】

**核心原则（语义翻转）**：hard certificate 定义为**「连续竞技区域 D_1800 已被该频道**所有真实 NO_SIGNAL 检测圆**的并集覆盖」**，**不是**「访问过 Way3 固定 7 点」。Way3 `omni_scan_points()` 从「证书本体」降级为 **`fallback_anchors`（completion template）**：一个已知能补完覆盖的兜底可行解。∴ 任何为定位/搜索/路线而做的主动 NO_SIGNAL 扫描都天然积累证书 → 消除「主动路线 + 最后再走固定覆盖路线」的双重路程。**一句话定位（论文可用）**：Way3 覆盖 backbone 自此是 **Way4 的安全下界（guaranteed completion），不再是 Way4 的运动轨迹上界**——正常局面由主动扫描形成的自然证书取代，仅在尾声兜底补洞。

### 6.1 数学（P3 全向，R_eff≥1000）
频道 c 记录所有真实 `scan(c)→NO_SIGNAL` 的位置 `S_c={s_1..s_m}`。全向检测 ⇔ 距离≤R_eff，故 NO_SIGNAL@s ⟹ 若存在源 p_c 则 `‖p_c−s‖>R_eff≥1000` ⟹ `p_c∉B(s,1000)`。于是 `D_1800 ⊆ ⋃_i B(s_i,1000)` ⟹ D_1800 内无合法源 ⟹ `ABSENT_CERTIFIED`（sound，用 1000 下界）。

### 6.2 两层严格分离（禁止混用）
| 层 | 作用 | 精度 | 可触发 ABSENT? |
|---|---|---|---|
| `CoverageGainMap` | planner 打分（候选覆盖价值） | 允许近似（40m grid 点 mask） | **否（Invariant B）** |
| `HardDiscCoverVerifier` | 唯一有资格 `is_complete=True` 的数学证明 | conservative：允许 false negative，**绝不允许 false positive** | 是 |

### 6.3 HardDiscCoverVerifier = 保守 Adaptive Quadtree
包围盒 `[-1800,1800]²` 递归四分。每 cell（axis-aligned square K）：
- **OUTSIDE_ARENA**：`min_dist(0,AABB) > 1800 + eps` → 剪枝（`+eps` 使剪枝更严=安全方向；勿把相切 cell 当 outside）。
- **CERTIFIED**：∃ 单个 `s∈S_c` 使 `max_{v∈corners(K)}‖v−s‖ ≤ 1000 − eps`（方块到定点最远必在角点）→ 整 cell ⊆ B(s,1000)（`−eps` 使认证更严=安全方向）。
- 否则四分至 `size ≤ min_cell_size`；仍未 CERTIFIED → `UNCERTIFIED`。任一 in-arena cell 到最小尺寸仍 UNCERTIFIED → verifier 返 **False**（≠「存在 hole」，仅「保守证不出」）。

**Soundness**：返 True ⟺ 每个 in-arena cell 被某单圆整体覆盖 ⟹ 必然 `D_1800⊆⋃B`。**已知保守性（false negative only；Way3 fallback 兜底、非 bug，须记录）**：
- **Conservatism note A（单圆判据充分不必要）**：单圆 cell 判据对并集覆盖**充分而非必要**。点若在圆盘并集内部，细分通常终能找到完全属于某单圆的足够小邻域；但**仅由多个闭圆盘边界共同覆盖、无任何单圆提供正覆盖裕量（positive coverage margin）** 的退化点，即使无限细分也认证不出（`1000−eps` 的刻意效果：恰好 `d=1000` 的边界覆盖本就不被 arbitrary verifier 轻易认证）。仅致 false negative，soundness 不受影响，legacy backbone 兜底。*The single-disc cell test is sufficient but not necessary for union coverage; points covered only through zero-margin intersections of multiple disc boundaries may remain uncertifiable under arbitrary subdivision — false negatives only.*
- **Conservatism note B（越界薄壳 & planner 隔离）**：V1 判「整 cell（含越界角点）被覆盖」而非 `K∩D_1800`，故竞技圆边界额外要求覆盖一圈越界薄壳（V2 可改测 `K∩D_1800`）。**hard verifier 的 unresolved cells 绝不直接作 planner 目标**：规划用竞技圆内独立近似图（§6.6）算 gain/holes，conservative quadtree 仅作 hard 布尔 gate，避免边界保守 cell 诱导机器人为覆盖圈外不存在区域绕路。*Hard-verifier unresolved cells are not used directly as planning targets; planning uses an independent in-arena approximate coverage map, the conservative quadtree is only the hard Boolean gate.*

### 6.4 Per-channel 状态与接口
`OmniChannelCertificate{channel, negative_scan_points:list[Point], heuristic_coverage_ratio, hard_complete, completion_source:CertificateSource|None, active_scan_count, fallback_anchor_count}`；`CertificateSource={ARBITRARY_DISC_COVER, LEGACY_BACKBONE, CARDINALITY}`。
`CertificateManager`：`record_observation(c,point,obs)` / `is_absent_certified(c)` / `hard_coverage_complete(c)` / `heuristic_coverage_ratio(c)` / `coverage_gain(c,q)` / `remaining_holes(c)` / `suggest_completion_points(c,robot_pos)`。
`record_observation`：positive → 标 PRESENT 并**停止该频道证书计算**（§6 尾；历史 negative disc 仍留作空间排除）；`NO_SIGNAL & P3` → 追加 `negative_scan_points` + 更新 CoverageGainMap +（时机允许才）跑 hard verifier，complete 则置 ABSENT_CERTIFIED。

### 6.5 三来源析取
`is_absent_certified(c) = arbitrary_disc_cover_complete(c) ∨ legacy_backbone_complete(c) ∨ absent_by_cardinality(c)`。
- **Cardinality**（line27 上限16）：`|{c: state(c)∈{PRESENT,CLEARED}}| = 16`（数**不同频道**；CLEARED 是 PRESENT 的后继态、不重复计） → 其余 UNKNOWN 直接 ABSENT_CERTIFIED（source=CARDINALITY，无需覆盖）；但这 16 个仍须**全部 CLEARED** 才能 exit（existence certificate 答「有没有」、clear certificate 答「清完没」，不可混）。可计入的 PRESENT 之合法性见 **Invariant D**（§6.10）。
- **Legacy backbone**：走完 Way3 `omni_scan_points` 且 `verify_one_cover` 通过 → hard（保证可结束，Invariant C）。

### 6.6 CoverageGainMap（planner 用，40m）
竞技圆内预生成 ~数千 `grid_centers[N,2]`；每频道 `covered_mask[c]:bool[N]`。NO_SIGNAL@s：`covered_mask[c] |= ‖grid_centers−s‖²≤1000²`。`coverage_gain(q,c)=|(‖grid−q‖≤1000) & ~covered_mask[c]| / N`；batch `G`：`Σ_{c∈G} w_c·gain`（UNKNOWN 从未出现→权重大；PRESENT/CLEARED/ABSENT_CERTIFIED=0）。并入候选效用（真正 multi-purpose waypoint）：`Utility(q,G)=w_E·explore + w_L·localization + w_C·certificate + w_R·route − w_T·T`。

### 6.7 Verification mode（后期只剩 UNKNOWN）
不机械走满 7 点。查 `remaining_holes` + coverage map 生成 `CertificateCompletionCandidate`（来源：Way3 fallback anchors + hole 附近优化点；V1 仅从 anchors 选、但**顺序动态**）。未访问 anchor a_j：`Score=Gain(a_j)/Cost(a_j)`，`Gain=Σ_{c∈UNKNOWN}coverage_gain(a_j,c)`，`Cost=‖x−a_j‖/5 + 频道扫描耗时`；贪心取最高 Score，每完成一个**重跑 verifier，complete 立即停**（不补剩余 anchor）。completion waypoint **只 batch 扫仍 UNKNOWN 且未 certified 的频道**，与 §7 联动。

### 6.8 性能（勿拖慢 planner）
Hard verifier 只在**真实 NO_SIGNAL 后**与**准备 EXIT / 进 verification mode 时**调用；候选模拟（32候选×horizon）**只用 CoverageGainMap，绝不跑 quadtree**。quadtree 优化（不牺牲 soundness）：scan 点存 numpy；cell 先 center 便宜接受 `‖center−s‖+ρ ≤ 1000−eps`（`ρ=√2/2·size` 半对角，三角不等式上界）；缓存 `scan_set_hash→result`；新增 scan 后只重查未决 cells。

### 6.9 P4（定向）本阶段不升级
P4 continue：Way3 `directional_scan_points`（~31 点，`verify_three_cover` 角隙<180°）= hard certificate；主动任意扫描点**仅算 directional heuristic gain**。原因：P4 需对潜在 p 保证 `Δα_max(p)≤180°`，`‖S_i−p‖≤1000` 扫描点集随 p 连续变化，普通网格/quadtree 不能直接成 hard certificate。不为架构对称强行提前。

### 6.10 四条不可违反 invariant
- **A**：只有**真实 NO_SIGNAL observation** 才能产生 P3 hard exclusion disc（机器人经过 / 在该点扫别的频道 / 计划要去但没去 / world-model 或 RL 预测「大概没信号」——**都不算**）。
- **B**：heuristic coverage map **永不**直接触发 ABSENT_CERTIFIED（只有 §6.5 三来源）。
- **C**：arbitrary verifier 证不出时，Way3 fallback 仍必保证任务可结束。
- **D（cardinality 纯净性）**：只有**至少一次合法 positive sensor observation（或接口等价确定性事件）确认的 PRESENT** 才可计入 cardinality 的 16；high-probability / coarse hypothesis / planner prediction / RL 输出 /「应该有」/ 可疑信号**一律不算**。否则错认一个 PRESENT 会经 cardinality shortcut 把某个真实频道直接误判 absent（危险放大器）。
- **Resolution note（`min_cell_size` 是求证参数、非误差精度）**：它只控制 conservative verifier 的**证明能力与计算开销**，不影响 soundness——终止尺度偏大只会让「已真实覆盖但局部关系解析不出」的区域返回 incomplete（false negative），**绝不会让存在未覆盖区域者错误返回 complete**。故勿表述成「10m 以下的洞认不出」（会被误读为可靠性精度）。*`min_cell_size` controls proof completeness and computation, not certificate soundness.*
- **配置**：`certificate.p3{hard_verifier:adaptive_quadtree, arena_radius_m:1800, guaranteed_detection_radius_m:1000, nominal_min_cell_size_m:10, rescue_cell_sizes_m:[5,2.5], rescue_only_in_verification:true, max_depth:11, numeric_eps_m:1e-7}`（`max_depth` 从 10 提到 **11**：3600m 根到 depth10 才 3.5m，够不到 rescue 的 2.5m）。**Rescue（性能优化、不影响 correctness）**：仅在 verification 尾声（heuristic coverage≈100% 且只剩极少 unresolved cells）对**未决叶子局部**续分 10→5→2.5m（不重展整树），仍证不出才调 Way3 fallback——避免「实际已覆盖、只差一条 ~8m seam 却为此多走数百米去 anchor」。慢→调大 nominal、false-neg 多→调小；**绝不减 safety margin（1000−eps）换效率**。

**实现顺序（规格 §42，DECISION-1 部分，M3 展开）**：①CertificateManager per-channel 状态 ②CoverageGainMap ③HardDiscCoverVerifier + false-positive/property tests ④接 P3 NO_SIGNAL ⑤EXIT guard 改三来源 ⑥Candidate 加 certificate_gain ⑦verification planner 动态选 anchor ⑧ChannelScheduler 只 batch 未 certified 频道 ⑨FutureCost `J_certificate` 依赖 holes ⑩Way3-vs-Way4 metrics ⑪P3 稳定 100% full-clear 后再碰 Q4。**全程不同时改 planner 与 RL。**

## 7. 候选宏动作 + Batch Scan + Channel Scheduler
- `MacroActionType = {EXPLORE, INITIALIZE, REFINE, PURSUE, CLEAR, VERIFY, EXIT}`
- `MacroCandidate` = Way2 `CandidateAction` + `scan_channels: tuple[int,...]` + `exploration_gain/refinement_gain/certificate_gain/route_gain`。RL 只对 `MacroCandidate[]` 排序。
- **Batch scan**：一次移动到 q，按 scheduler 顺序测多频道；每 primitive obs 立即更新 belief；出现 near/too-strong 允许 opportunistic clear；超时即停。
- **源/机器人域分离**（修 Way2 bug）：源候选 `‖p‖≤1800`；**机器人扫描点不 clamp 回 1800**（外圈点对边界定向源必要，line 12）。
- **Channel Scheduler**：对点 q 选频道集 `G_q` 最大化 `Σ_{c∈G} V_c(q) / (5|G| + N_switch(G))`，`V_c=V_discover+V_refine+V_certificate`；早期允许全 20 频道 batch，中后期收缩；CLEARED/ABSENT 永不再扫；PRESENT+clearable 优先 CLEAR。注意 `/clear` 不计切频（line 126）。**Verification 阶段**：completion waypoint 只 batch 扫仍 UNKNOWN 且未 certified 的频道（§6.7），不重扫全 20，省 measure+switch。

## 8. 主动感知
- **Minimax NBV**（Tokekar 迁移，非固定 90°）：对 detected c，代表假设 `P={顶点,重心,MEC 边界,长轴端点}`（限 ≤8 个）；候选 q 打分 `U(q)=max_{p_j∈P} diam(F_c ∩ W(q,θ_j,±1°) ∩ B(q,1500))`，`NBV(q)=U(q)+λ_t T(q)`，取 argmin。候选来源：Tokekar minimax 点、Bishop 90° 交会点、Way3 垂直 baseline、近距 MEC 侧点。
- **Multi-purpose waypoint**：`I_total(q)=Σ_c I_c(q)+λ_C C(q)`，一个点同时服务多频道/覆盖/路线。
- **Event-triggered VOI**（Cautious-Greedy）：`VOI(a)=ΔU_pred/(t_move+t_detect+t_switch)`，低于阈值不单独停测，但叠加 coverage/route gain 后可能仍高。

## 9. 路由
- **Clearable 目标**（`MEC.r≤thr`，≤16 个，视作确定点）：**Held–Karp 精确 open TSP**。**⟨工程约束⟩** Held–Karp 是 `O(2^n·n²)`——n=16 单次 ~16.7M ops 可接受，但**严禁在 receding-horizon 每次 replan / 每个候选的 future-cost 里重算**；必须 memoize，仅当 clearable 集合变化时重解。
- **Uncertain 目标**（neighborhood `F_c`）：TSPN 近似，接近距离 `d_N(x,F_c)=max(0,‖x−m_c‖−r_c)`，再加预计定位代价 `L_c`（可用 offline_sim 拟合 `L̂=a0+a1·r_MEC+a2·diam+a3/max(sinγ,ε)`，仅估成本、不改硬信念）。

## 10. 未来成本 + Receding-Horizon
- `Ĵ(B)=J_route+J_localization+J_exploration+J_certificate`（全部**廉价解析代理**，禁用整局精确树）。**`J_certificate` 重定义（DECISION-1 后）**：不再是「剩几个固定 anchor」，而是「按**当前实际 coverage holes** 完成 hard 覆盖的预计最小成本」——对 UNKNOWN 频道联合未覆盖 mask 跑 greedy set-cover(剩余 fallback anchors)+travel 近似。∴「现在为定位 ch3 去某点、顺便消掉大片 hole」会真正压低 `Ĵ(B')`，future-cost planner 才理解长期证书节省。**（Conservatism note B）** `J_certificate` 的空间权重只用竞技圆内近似 coverage map（§6.6）的 holes，**不**用 hard verifier 的 unresolved cells（含越界薄壳保守 cell，直接当规划目标会诱导机器人为覆盖圈外不存在区域绕路）。
- 决策：`Q(a)=C(a)+ f_o[Ĵ(B'|a,o)]`，默认 **robust/minimax**：`a*=argmin_a[C(a)+max_{o∈O(B,a)} Ĵ(B')]`；`outcome_mode: robust|expected` 可配。
- **Rolling**：`Observe→Update→Plan→Execute one`。V1 `horizon=1`（route-aware 一步）；V2 `horizon=2, beam_width=8`。**⟨实时预算⟩** 全程真实 compute ≤20 min（line 129），每步规划须快 → outcome 集 O 限 ≤8 代表观测，Ĵ 用解析代理，路由 memoize。

## 11. 安全层 + Way3 Homing + Fallback
- **SafetyShield 可否决 planner**：CLEAR guard（无 `MEC.r≤thr` 不许盲清）、EXIT guard（`∀c: CLEARED(c)∨ABSENT_CERTIFIED(c)` 才许退出）、cardinality shortcut、Q4 visibility guard、timeout guard。
- **Way3 homing 原样移植**（已核实函数存在，内部安全逻辑于 M9 移植时逐一比对）：`_establish_baseline`（单 bearing 时**垂直优先**，勿沿 bearing 直进以免越过定向源到背面）、`_orbit_triangulate`（精定位侧点朝**历史成功检测点相对估计点的方向圆均值**，非当前机器人方向）、`_try_clear`/`_home_and_clear`（clear guard）。**这些硬安全规则 RL 不得覆盖。**
- **Safe fallback**：planner 异常/无候选/数值退化/RL 不可用 → 退回 Way3 式保证完成策略（补完 backbone → 确认所有 present → Way3 homing → clear → 证书完成 → exit）。

## 12. RL（最后做，M10）
- 只在 Way4-Math 稳定后。不重搭 env。`Belief → MacroCandidate set → feature → 选择`。
- 推荐 **residual**：`Q(B,a)=Q_math(B,a)+ΔQ_θ(B,a)`。数学负责正确/安全/几何/覆盖/清除保证；RL 只学长期 trade-off。先 MLP scorer，行有余力上 Set Transformer（编码 20 频道无序集）。不上 Dreamer/Mamba/GNN/LSTM（belief 已压缩历史）。

## 13. 架构 / 目录 / Pipeline
```
SXJM/
  offline_sim/  estimator/  Way1/  Way2/  Way3/
  sxjm_core/geometry/            # M1 唯一几何核心
  Way4/
    DESIGN.md (本文)  configs/  tests/  scripts/{evaluate,compare_way3_way4,run_stress,ablation}.py
    src/way4/
      core/ belief/ certificate/{omni,coverage_gain,hard_disc_cover,manager,fallback} sensing/ channels/ routing/ planner/ executor/ safety/ rl/ evaluation/ pipeline.py
```
Pipeline：`Env → BeliefUpdater → CertificateManager → CandidateGenerator → ChannelScheduler → AnalyticalCostModel → RouteEstimator → FutureCostEstimator → RecedingHorizonPlanner (→ optional RL residual) → SafetyShield → MacroExecutor → Env`。

## 14. 里程碑
- **M0** 冻结 baseline：offline_sim/estimator/Way2/Way3 测试仍绿；记录 Way2/Way3 在固定 seeds 的 clear success / time / move / measure / switch 作 regression benchmark。
- **M1** Geometry consolidation：建 `sxjm_core.geometry`，四处 wrapper 化；**验收**：estimator/Way2/Way3/Way1 几何测试全过；**不同时改 planner**。
- **M2** Way4 basic belief（P3）：positive/near/negative/MEC/clearable；**property test**：随机合法源 + 任意 ±1° 误差，真源永不被 positive update 排除。
- **M3** Certificate Manager（DECISION-1 细化，§6）——**M3A** P3 任意扫描点 Hard Disc-Cover（adaptive quadtree，验收：任意真实 NO_SIGNAL 都入证书；verifier 无已知 false positive；planner 可读 coverage gain；主动 scan 能减少后续 fallback anchor）；**M3B** P3 Way3 legacy fallback（`omni_scan_points`+`verify_one_cover`）；**M3C** P4 暂仅 legacy directional backbone。Way3 覆盖测试全过。
- **M4** Macro Candidate + Batch Scan：一次移动扫多频道；**验收 = 逐秒复现 §1.1 黄金样例**（尤其步5：`/clear` 不改测向频道 ⇒ 切频=0）。
- **M5** NBV/refinement：minimax feasible-set shrink；随机单源下 NBV 后 region diameter 显著优于随机第二点。
- **M6** Routing：Held–Karp（小 n 对拍 brute-force 一致）+ TSPN + Way3 2-opt fallback。
- **M7** Future cost + rolling（H1→H2 beam）：**不得降低 full-clear 比例**；开始 Way3 vs Way4 对比。
- **M8** Q4 directional：粗假设层 + 正/负信息 + directional visibility gain；**property test**：随机合法定向源，sound negative rule 不消除真源所在保守 cell。
- **M9** Safety + fallback：Way3 homing 移植 + EXIT/CLEAR guard + 异常 fallback；全 stress 家族优先 100% 全清。
- **M10** RL（可选）。

## 15. 测试 / 消融 / 对比
- 五层测试：unit / property（**真源永不被 sound update 排除** 最重要）/ scenario（collinear, far pair, boundary, clustered, tiny region, Q4 outward, Q4 evasive）/ cross-impl（Way3 vs Way4 同 case，不得以成功率换速度）/ Monte Carlo（先 `P(clear all)`，再 time P50/P90/P95/max、move、scans、switches、failed clears）。
- **P3 certificate 专项**：`test_p3_disc_cover_verifier`（空集→False / 单圆→False / 人工明显 hole→False / 仅剩 20-10-5m 小洞→False / Way3 七点→`legacy_backbone_complete`；superset 单调性 `S1⊆S2 ∧ verify(S1)⟹verify(S2)`；置换不变——**重点堵 false positive**）；`test_way4_active_certificate`（**核心回归**：没走满 7 anchor 但主动扫描已成完整覆盖 ⇒ `is_absent_certified==True`；反之留一个 hole ⇒ False）；property：存在 arena 内且在所有 NO_SIGNAL 圆外的合法源 ⇒ verifier 不得 complete（大量随机/adversarial witness）；dense 1/2/5m grid 仅作独立 oracle（**本身非形式证明**）。指标（Way4 应 < Way3）：`active_scan_certificate_fraction / fallback_anchor_count / certificate_only_travel_distance / certificate_completion_time`。
- 对比矩阵：`Way2 Greedy | Way3 Fixed-Certificate | Way4-Math | Way4-Hybrid`，统一跑 `offline_sim` 的 smooth/iid/biased/piecewise/adversarial/stress。
- 消融开关：`early_global_exploration / minimax_nbv / multipurpose_waypoints / future_route_cost / adaptive_channel_schedule / certificate_gain / q4_negative_information / receding_horizon`。
- **验收优先级：correctness > robustness > time**。任何降低清除成功率的版本不作正式 Way4。

## 16. 禁止事项（已对题面核实）
1. 重写 simulator（用 offline_sim）2. 第四/第五套 geometry（先合并 sxjm_core）3. 把 ±1° 当 Gaussian 作主硬信念 4. 同点重复测量求平均（题面已坐实：同点误差固定、重测零增益）5. P4 no-signal 直接减 1000 圈（定向可能背对）6. planner 直接标 absent（只有 CertificateManager/cardinality 可）7. planner 绕过 clear guard 8. RL 输出任意坐标 9. 一上来 Dreamer/Mamba/GNN 10. 为提速降 full-clear 比例 11. 用「dense grid 采样点全被覆盖」当 hard 证书（只证采样点被覆盖、证不了点间无 hole；只能做 heuristic）——hard 证明只能来自保守 quadtree（§6.3）/ Way3 定理 / cardinality。

## 17. 待用户拍板的决策
- ✅ **DECISION-1 已定：采纳**——P3 任意扫描点 hard disc-cover（adaptive quadtree）进 V1，细化为 §6（M3A/B/C、§10 `J_certificate`-by-holes、§6.7 动态 anchor、Invariant A/B/C、§16 禁令11）。P4 仍 backbone 到 V2。
- ~~待核实-A~~ **已闭合**（§1）：`B题(1).pdf` 附录2 确认「同一地点误差固定、重复检测不改变误差」→ `禁止4` 坐实，无需再查。
- ✅ **DECISION-2 已定：Greenlight（2026-09-11 审阅通过）**——用户审校四项结论 #1/#2/#4 原样通过、#3 通过（`min_cell_size` 定为求证参数）；要求补入 Conservatism note A/B、Resolution+rescue、Cardinality Invariant D（均已写入 §6）。授权开工：M0（记录 Way2/Way3 baseline，无需分支）→ 首次提交 Way4 代码时用 `git worktree` 从 `origin/way3` 建 way4 → 按 §6 尾实现顺序 M1→M2→M3A… 落地，不同时改 planner/RL；是否 cherry-pick Way2 M2/M3 留到 RL 阶段（M10）再定。
