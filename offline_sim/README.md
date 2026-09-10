# offline_sim —— 无线电干扰源模拟器（本地离线复现）

严格按《附件1 模拟器使用说明》《附件2 通信接口说明》复现官方模拟器，供问题 3/4 的机器狗策略**离线无限次**开发与调参。机器狗程序只需把 `BASE_URL` 从真机指向本模拟器地址，**无需改动任何代码**。

## 模块

| 文件 | 职责 |
|---|---|
| `case.py` | 案例/干扰源模型、随机案例生成、**压力/最坏情况**案例生成 |
| `fields.py` | **可插拔示向度误差场家族**（iid / smooth / biased / adversarial / piecewise） |
| `engine.py` | 纯逻辑引擎：物理判定 + 计时规则 + 状态推进（微秒累计，无 HTTP） |
| `server.py` | HTTP+JSON 层：方法/路径/头/体校验、状态码、`accepted` 语义、`request_id` 幂等 |
| `harness.py` | **蒙特卡洛试验场**：策略批量评估 + 统一指标（成功率 + 时间分位 + 动作计数） |
| `client.py` | 轻量客户端（自测用，可被机器狗复用） |
| `run.py` | 命令行启动器 |
| `test_sim.py` | 167 项自测（单元 + 性质/不变量 + 边界 + 误差场 + 压力/试验场），全部通过 |

## 定位

这不是"官方模拟器的严格复刻"（官方**未公开**随机案例分布与误差场概率分布），而是：

> **严格满足题面公开的物理、计时与交互约束的离线仿真环境，并提供多种随机及极端案例、多种误差场以做算法鲁棒性 / 最坏情况验证。**

### 误差场家族（`fields.py`）

题面只约束示向度误差：①有界 `∈[-1°,1°]`；②地点决定（同点重复不变）；③不同地点才呈统计规律——**未规定分布形状、是否零均值、是否空间独立**。故不假设唯一分布，提供 5 种都满足约束的场：

- `iid` — 不同地点独立 `U[-1,1]`（会奖励"原地微动多测取平均"，用来暴露依赖此假设的策略）
- `smooth` — 空间相关（相关长度 ℓ 可调；微动几乎测到同一误差，**默认**）
- `biased` — 整体偏置（均值≠0）→ 打击"误差均值为 0"的最小二乘
- `adversarial` — 误差贴近 ±1°（分块翻转 / 常偏）→ 最坏情况保证
- `piecewise` — 分区电磁环境，每块各有相关偏差

`generate_case(..., field_kind="biased", field_params={"bias":0.6})` 即可切换。

### 压力 / 最坏情况案例（`case.py: generate_stress_case`）

`edge_cluster / min_reff / tiny_cluster / collinear / far_pair / max_count / min_count / dir_outward / dir_boundary / dir_evasive`——专门生成"我们最怕"的布局（贴边、最小 R_eff、近共线致三角定位病态、定向源背向原点或朝向避开扫描点等）。仍严格满足题面硬约束。

### 蒙特卡洛试验场（`harness.py`）

```python
from offline_sim import evaluate, evaluate_stress
m = evaluate(my_policy, n_cases=200, problem=3, field_kind="smooth")
print(m.report())   # 成功率 / 时间 mean·P50·P90·P95·max / 动作计数 / clear 失败数
```

指标以 **成功率 P(clear all) 优先**，再看时间分位——一个平均更快但偶尔漏源的策略未必更好。

## 启动

```powershell
# 问题3（全为全向），随机案例
python -m offline_sim.run --robot-id <参赛队号> --problem 3

# 问题4（全向+定向混合），固定种子复现
python -m offline_sim.run --robot-id <参赛队号> --problem 4 --seed 2026

# 从固定案例文件复现
python -m offline_sim.run --robot-id <参赛队号> --case case.json
```

默认监听 `http://127.0.0.1:2026`（与官方一致，可用 `--port` 改）。`--mode practice` 退出时打印案例真值便于对策略打分；`--mode formal` 不揭示真值。

## 已严格落实的规则（对照文件）

**物理（附件2 §2 / 附件1 §2）**
- 有效接收半径 `R_eff∈[1000,1500]`，接口不返回；超距 → `no_signal`。（题面未规定各源互异，允许重复）
- 全向：距离 ≤ R_eff 即可；定向：还须检测点位于覆盖角（方向两侧各 90°，共 180°，**含边界**）内。
- 示向度误差 ∈ `[-1°,+1°]`、保留两位小数、`[0,360)` 归一化；**由地点决定 → 同一地点重复测量误差不变**。
- 近距 5 m 且在覆盖角内 → `near`（无 svd）；清除半径 20 m（与定向朝向无关）；同源只能清一次。

**计时（附件2 §4 / 附件1 表1）**
- 移动 = 距离/5；切换频道（仅 `/measure` 频道变）1 s；检测 5 s；清除未发现 3 s / 成功 5 s。
- `/enter`、`/exit` 不推进虚拟钟；`accepted=false` 时 `virtual_time_s=0`。
- 微秒累计，最多 6 位小数。已用附件2 §10 计时示例校验：`105 / 111 / 194 / 199` **精确一致**。

**通信（附件2 §5）**
- 方法/路径精确匹配；`Content-Type` 必须 `application/json`（仅允许 `charset=utf-8`）；`Content-Encoding` 省略或 `identity`。
- 体：无 BOM UTF-8 JSON 对象、拒绝重复键、深度 ≤16、≤65536 字节。
- `channel` 为 1..20 整数（`1.0` 接受、`1.5` → 400）；坐标有限且 `|·|≤2e6`。
- 状态码：`200/accepted`、`200/accepted=false`（未知字段/`arena_id`/`robot_id` 不符/重复 enter）、`400/404/405/409/413/415/500`。
- `request_id` 幂等：同 id 同内容返回首次结果、不重复推进；同 id 改内容 → 409；结构错误/未知字段不占用 id。
- 结束后再来动作 → **直接关闭连接、无 JSON 体**。

## 自测

```powershell
python -m offline_sim.test_sim   # 167/167 通过
```

## 与真机的差异（已知、无害）

- 现实时限用系统时钟近似；`remaining_real_duration_s` 默认 1200，可按需调。真实 25 min 窗口/5 s 倒计时不影响策略逻辑。
- 未复现登录、加密日志上传、界面等与算法无关的部分。
