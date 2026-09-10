# estimator —— 问题 1 / 2 的几何定位库

只依赖 `numpy` + `matplotlib`（出图）。核心几何为纯 Python、可被仿真与策略复用。

## 问题背景

- **问题 1**：对同一干扰源，在若干检测点各测得一次示向度（每次误差 ∈ `[-1°,+1°]`）。
  每个检测点给出一个 `±1°` 的"示向度楔形"，真源必落在所有楔形之交
  `D = ∩ W_i`（凸多边形）内。要求：求 `D`、求其**直径**、并判定
  **"以该直径为直径的圆"能否覆盖整个 `D`**。
- **问题 2**：先在 `S1` 测得 `svd1`，再选第二检测点 `S2`。两条 `±1°` 楔形之交是一个
  近平行四边形的小区域，其伸长量 `L` 度量定位精度。分析**交会角/距离**如何影响 `L`，
  并给出第二检测点的**候选区域**与**推荐点**。

## 模块

| 文件 | 职责 |
|---|---|
| `geometry.py` | 角度/向量、示向度楔形→半平面、半平面交（凸多边形）、凸包、多边形直径、直径圆覆盖判定（Thales）、最小包围圆（Welzl） |
| `problem1.py` | `locate_region` / `solve_problem1`：求 `D`、直径、直径圆覆盖 |
| `problem2.py` | `localization_length_L`、`crossing_angle`、`second_point_candidates`、`recommend_second_point`、`region_diameter_two_points` |
| `figures.py` | **论文插图**：12 张 PNG+PDF（见下） |
| `test_estimator.py` | 自测（单元 + 不变量 + 边界 + 一阶公式一致性），30/30 通过 |

## 关键结论（代码即证明）

- **楔形不变量**：误差 `∈[-1°,+1°]` ⇒ 真源必在 `±1°` 楔形内（`test_wedge_contains`、
  `test_p1_source_always_inside`：1500 随机局全部命中）。
- **直径圆不一定覆盖 `D`**：近等边三角形是反例——顶点落在直径圆外，最小包围圆半径达
  `D/√3 > D/2`（Jung 定理）。见 `p1_cover_fail`。故论文应回答"**不一定**，充要条件是
  每个顶点 `P` 满足 `(P-A)·(P-B) ≤ 0`；钝角/长条形状可覆盖，近等边三角不可"。
- **P2 精度公式**：`L ≈ (2δ/sinγ)·√(r1²+r2²+2·r1·r2·|cosγ|)`，
  在 `γ=90°` 取最小、随 `r2` 单调增（`test_p2_*`）。一阶 `L` 与两楔形真实交区直径
  数值吻合（误差 <15%，`test_p2_first_order_matches_geometry`）。
- **第二点是"候选带"而非单点**：`r1` 未知 ⇒ 沿 `svd1` 每个假设源位置对应一个
  垂直偏移 `r2` 的最优 `S2`，扫出测向线两侧两条平行月牙带（`p2_candidate_crescent`）。

## 出图

```powershell
python -m estimator.figures                 # → estimator/figures/*.png + *.pdf
python -m estimator.figures --outdir out --dpi 220
```

| 图 | 内容 |
|---|---|
| `p1_wedge_single` | 单检测点 `±1°` 示向度楔形 |
| `p1_two_wedges` | 两楔形之交（平行四边形定位区）+ 放大插图 |
| `p1_multi_region` | 四点之交（凸多边形 `D`）+ 直径 + 直径圆 |
| `p1_cover_success` | 直径圆覆盖成功（钝角/长条形 `D`） |
| `p1_cover_fail` | **反例**：近等边三角 `D`，顶点在直径圆外 + 最小包围圆对比 |
| `p1_shrink_vs_n` | 定位区直径随检测点数下降（含 10–90 分位带） |
| `p1_geometry_effect` | 两点张开角对区域直径的影响 |
| `p2_L_vs_gamma` | `L–交会角`曲线（90° 最优，多 `r2`） |
| `p2_L_heat_gamma_r2` | `L(γ, r2)` 热力图 + 等值线 |
| `p2_L_heat_r1_r2` | `L(r1, r2)` 热力图（γ=90°） |
| `p2_candidate_crescent` | 第二点候选月牙带（`r1` 未知）+ 推荐点 |
| `p2_good_vs_bad_second` | 好/坏第二点定位区对比（γ=90° vs 近共线） |

## 自测

```powershell
python -m estimator.test_estimator          # 30/30 通过
```
