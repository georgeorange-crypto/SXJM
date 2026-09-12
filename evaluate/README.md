# evaluate —— 严格物理下界·Oracle·现实算法·消融 四层评价体系

针对 2026 CUMCM B 题第三/四问（机器狗无线电干扰源定位与清除），把评价从
“我们比某基线快 XX%”提升为**“距离物理极限还有多远，以及剩下的时间损失在哪里”**。

核心统一指标：`R = T / T_ABS-LB`（下界恒为 1）。并把每个方法的非理想损失干净地拆成
**额外移动 ΔT_move**（因不知真位置多走的冤枉路）与 **额外感知 ΔT_info**（相对理想探测器多花的信息获取时间）。

> 依赖 `Way3/jammerhunt`（真值 environment、OURS 策略 `Hunter`、几何工具）。`import eval_suite`
> 会自动把 `Way3/` 挂到 `sys.path`，无需手动配置。纯 Python，额外依赖 numpy/scipy/matplotlib。

---

## 一、四层（+梯队）结构

| 层级 | 方法名 | 知道什么 | 用途 |
|---|---|---|---|
| **LB** | `lb` | 一开始已知全部真值 | 不可突破的绝对物理时间下界（`R≡1`） |
| **O1** | `oracle` | 20 频道各搜一次即得精确坐标 | “检测完全理想化”后的工程极限 |
| **B1** | `reactive` | 发现一个、定位一个、清一个 | 最朴素现实基线 |
| **B2** | `greedy_scan` | 先全局侦察，但按最近目标贪心走 | 隔离“全局路径优化”的价值 |
| **OURS** | `ours` | 全局侦察 + 主动定位 + 全局规划/重规划 | 最终方法（= `Way3` 的 `Hunter`） |
| 消融 | `no_global_scan` / `no_global_route` / `no_active_sensing` / `no_replanning` | 各拔掉一个部件 | 归因每个部件的贡献 |

设计上 `no_global_scan` 行为等价于 `reactive`，`no_global_route` 等价于 `greedy_scan`
（互为交叉验证，实测时间逐位一致）。

## 二、下界与证书（`lowerbound.py`）

关键：机器狗**只需进入以真源为圆心、半径 20 m 的圆盘**即可清除，故全知移动问题是
**开放式带邻域 TSP（Open TSPN）**，而非点 TSP。

- **绝对物理下界 `L_LB`**：圆盘间最短距离矩阵（`w0_i=max(0,|p_i|-20)`，`w_ij=max(0,|p_i-p_j|-40)`）
  上的**精确** Held-Karp 开放式最短 Hamilton 路。因真实路径每段 ≥ 两圆盘最小间距，故
  `L_LB ≤ L_TSPN*` 可证。→ `T_ABS-LB = L_LB/5 + 5m`。
- **收紧可行上界 `L_UB`**：在 Held-Karp 给出的圆盘访问序上，对 `x_i∈D_i` 最小化路径长
  （凸问题，SLSQP）。任一可行路线都是上界，故 `L_LB ≤ L_TSPN* ≤ L_UB`。
- **简单可行上界 `L_center`**：走每个源中心的开放式 TSP（NN+2opt）。
- 报告相对间隙 `gap_rel=(L_UB-L_LB)/L_UB`，把全知最优移动夹在一个窄区间里。

`oracle` 的时间由定义直接给出：`119s`（20 频道各扫一次 = 100s 检测 + 19s 切频，初始频道 1）
`+ L_UB/5 + 5m`。

## 三、计时分解（`metrics.py`）

`InstrumentedWorld` 包住任意 `World`，按引擎计时规则逐动作累加，保证
`T = T_move + T_detect + T_switch + T_clear` 精确等于引擎虚拟钟（有单测校验）。由此：

```
ΔT_move = T_move - L_LB/5           额外移动代价
ΔT_info = T_detect + T_switch - 119 额外感知代价
C_move  = ΔT_move / (T - T_oracle)  非理想损失里“移动”占比
C_sense = ΔT_info / (T - T_oracle)  非理想损失里“感知”占比
```

### 逐动作诊断（`diagnostics.py`）

把上面的“移动 vs 感知”两分，进一步细化到**四个可操作动作**，各自对照物理地板，回答
**“时间到底浪费在哪个动作、还剩多少可优化空间”**：

| 动作 | 物理必要地板（每局） | 实际 | 超支 Δ 的含义 |
|---|---|---|---|
| 移动 move | `L_LB/5`（绝对移动下界） | Σ 每次动作移动/5 | 绕路 + 定位往返 |
| 检测 detect | `100s`（20 频道各扫一次） | `5s × 测量次数` | 交会定位的环绕补测 |
| 切频 switch | `19s`（频道 1→20） | `1s × 换频次数` | 清除阶段的频道穿插 |
| 清除 clear | `5·m`（每源必付） | `5s×命中 + 3s×漏命中` | 漏命中 = 定位精度余量 |

与既有归因严格对齐（有单测）：`Δ_move ≡ ΔT_move`，`Δ_detect+Δ_switch ≡ ΔT_info`，
`Σ超支 = T − T_floor`（`T_floor = L_LB/5 + 119 + 5m`，各分量独立成立的硬下界，因“边走边测”
不可同时取到，故不可达；可达联合理想即 Oracle）。移动超支再对照 Oracle 最优移动 `L_UB/5`
拆成**可约绕路**与**圆盘邻域松弛（不可约）**两部分。

CLI 会打印三张表（逐动作实际耗时 / 逐动作可优化余量Δ / 动作次数）加一段**自动诊断叙述**，
并另存**可优化余量图** `headroom_p{N}.png`（每方法一根堆叠柱 = 相对地板的超支，按动作着色）。

## 四、用法

```bash
# 全方法梯队 + 榜单 + 计时分解表 + 主图（问题 3）
python -m eval_suite --problem 3 -n 100

# 问题 4；只跑方法子集
python -m eval_suite --problem 4 -n 60 --methods ours,oracle,lb

# 问题 3↔4 定向性代价（同物理下界的配对案例）
python -m eval_suite --directional -n 80

python -m eval_suite --problem 3 -n 50 --no-fig   # 跳过画图
```

榜单按**字典序**排：先比全清成功率，再比成功局时间——漏源的方法没有资格用低耗时刷分
（正是这条挡住了“快而不全”的策略）。主图存到 `evaluate/out/figs/`。

## 五、实测结论（本包 environment，成功局均值）

**问题 3（全全向，n=60）** —— 梯队顺序如理论预期：

| 方法 | 全清率 | R_LB 中位 | R_oracle 中位 | ΔT_move | C_move |
|---|---|---|---|---|---|
| lb | 100% | 1.000 | 0.91 | 0 | — |
| oracle | 100% | 1.10 | 1.00 | 54 | — |
| **ours** | **100%** | **3.02** | **2.75** | 2576 | **0.80** |
| greedy_scan | 100% | 3.07 | 2.81 | 2650 | 0.81 |
| reactive | 100% | 12.50 | 11.44 | 19205 | 0.98 |
| no_replanning | **81.7%** | 2.80 | 2.54 | 2249 | 0.80 |

**问题 4（部分定向，n=40）**：`ours` R_LB≈6.22 全清 100%；`reactive` 掉到 95% 全清、R_LB≈37。

**关键实证结论**：OURS 的 `C_move≈0.80`（P4≈0.71）——**约 80% 的非理想损失来自额外移动，
而非检测本身**。这定量支撑了“用少量前期检测换全局空间认知”的核心思想。

**消融**：拿掉 replanning 会把全清率从 100% 打到 81.7%（P3）——**重规划是 100% 全清的保证**，
即便它单看更快。这正是字典序评价的意义：`no_replanning` 时间更短但成功率不足，排在 OURS 之后。

**定向性代价（配对案例，n=40）**：`P_dir = T_Q4/T_Q3 ≈ 1.96`，`R4-R3 ≈ 2.99`。同一物理下界下，
定向源的“不可见性”让任务耗时近乎翻倍——这是问题 3→4 的定量过渡。

**逐动作可优化余量（P3, n=30, OURS 成功局均值）**：相对逐动作物理地板总超支 ≈ 3314s，
其中**移动 2636s（80%）| 检测 571s（17%）| 切频 107s（3%）| 清除 0s（0%）**。进一步：移动实际
4267s 里，对照 Oracle 最优移动 1680s，**可约绕路 ≈ 2587s**（定位往返 + 清除排序），邻域松弛仅 ≈ 49s
（圆盘半径的不可约项）；检测多测 114 次源自交会定位的环绕补测；清除零漏命中（定位精度无余量）。
**结论：OURS 的优化空间几乎全在“移动”——具体是把定位往返与清除排序再压向 TSPN 最优**，而检测/
切频/清除均已接近各自地板。这为“下一步优化投哪里”给出了量化依据。

## 六、重要限制（务必在论文中说明）

正式/演练测试**不暴露真实坐标**（附件仅承诺演练后给出**数量**）。因此本 Oracle/下界体系
**只能用于离线、ground-truth 已知的自建仿真集做科学验证**；正式盲测仍按题目只报清除率、
平均定位清除时间与程序运行时间。分工：**离线实验负责科学验证，正式测试负责最终盲测。**

## 七、模块

```
eval_suite/
  lowerbound.py     L_LB (Held-Karp) / L_UB (凸放置) / L_center / 证书区间
  metrics.py        InstrumentedWorld 计时分解 + R_LB/R_oracle/ΔT/C 指标
  oracle.py         One-shot Oracle 策略（原点全扫 → 最优 TSPN 清除）
  baselines.py      reactive (B1) / greedy_scan (B2)
  ablations.py      AblatableHunter：四开关 = Full 与四个消融
  runner.py         方法×案例 → EpisodeMetrics；同批种子多方法；字典序汇总
  report.py         榜单表 / 计时分解表 / 论文主图（堆叠柱）
  diagnostics.py    逐动作诊断：move/detect/switch/clear 超支表 + 自动诊断 + 余量图
  directional.py    Q3↔Q4 配对案例的定向性代价
  interface_shim.py 复用 jammerhunt.interface 的 World 契约
tests/              下界正确性（Held-Karp vs 暴力）、计时一致性、方法契约、逐动作诊断恒等式
```
