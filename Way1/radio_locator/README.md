# radio_locator — Way1 方案实现

**2026 国赛 B 题**《无线电干扰源的快速自动定位与清除》纯数学 + 算法路线。

方法主线：**有界误差集合估计（Set-Membership）+ 计算几何 + 鲁棒主动感知（Minimax）+ 覆盖理论 + 在线组合优化**。
全程不使用机器学习 / 概率预测模型；一切推理建立在题目给出的硬约束之上。

## 最高优先级

```
Correctness > Completeness > Robustness > Efficiency
```

核心不变量（property-based test 强制）：

> 只要模拟数据满足题目硬约束，真实干扰源就永远不能被 feasible set 排除。

## 四条数学主线

1. **有界误差集合定位** —— 严格利用 `e∈[-1°,1°]`，不造概率分布。
2. **保证型覆盖搜索** —— Q3 圆盘覆盖；Q4 定向半平面三角剖分完备性。
3. **Minimax 主动感知** —— worst-case 不确定度收缩，回答"为什么选这个测点"。
4. **MEC 清除证书** —— `R_MEC ≤ 20m` ⇒ 移动到 MEC 圆心必进入光学范围，保证清除。

## 目录

```
radio_locator/
├── configs/            # common / q3 / q4 参数（全部配置化）
├── src/
│   ├── domain/         # 数据结构、观测、频道状态、世界状态
│   ├── geometry/       # 角度/向量/半平面/楔形/裁剪/凸包/卡壳/MEC/box距离/圆周区间
│   ├── set_estimation/ # quadtree / Q3 可行集 / Q4 全向+定向可行集 / 外包络
│   ├── localization/   # Q1 求解 / Q2 第二测点 / 自适应主动定位
│   ├── coverage/       # Q3 六边形覆盖 / Q4 三角网 / 覆盖验证
│   ├── optimization/   # Held-Karp / MST 下界 / 2-opt / 鲁棒前瞻 / 任务调度
│   ├── runtime/        # 顶层控制器 / 执行器 / 不变量 / 恢复
│   ├── simulator/      # 客户端 / 协议 / 本地 mock（含固定误差场）
│   └── analysis/       # 指标 / 绘图 / 导出
├── tests/              # 几何 / 可行集 / 覆盖 / 定位 / 集成
└── scripts/            # solve_q1 / demo_q2 / run_q3 / run_q4 / stress_test / benchmark
```

## 快速开始

```bash
cd radio_locator
pip install -e .            # 或 pip install -r requirements.txt
pytest -q                   # 全部单测
python scripts/solve_q1.py  # Q1 演示 + 图
python scripts/demo_q2.py   # Q2 第二测点候选区域
python scripts/run_q3.py    # Q3 mock 实测
python scripts/run_q4.py    # Q4 mock 实测
python scripts/stress_test.py --n 2000   # 不变量 property 压测
```

## 计时模型（对齐官方 附件2 §4）

```
T = Σ 移动(d/5) + Σ 切频(1·[Δchannel]) + 5·N_measure + Σ_clear(5 if success else 3)
```

- `/measure` = 移动 + 切频(变则1s) + 5s
- `/clear` = 移动 + (成功5s / 未发现3s)，不切频
- `/enter`、`/exit` 不推进虚拟钟
