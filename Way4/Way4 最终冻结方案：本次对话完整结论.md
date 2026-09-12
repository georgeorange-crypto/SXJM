# Way4 最终冻结方案  
## ——本次对话完整收敛结论

---

# 0. 先给最终结论

本次讨论最终确定的 Way4 主线是：

\[
\boxed{
\text{数学保证完成性和安全性}
+
\text{数学生成丰富的高价值空间候选}
+
\text{RL 学习候选位置的长期全局价值}
}
\]

完整表达为：

\[
\boxed{
\text{Way4-Final}
=
\text{Hard Mathematical Belief}
+
\text{Safe Spatial Candidate Generation}
+
\text{Unified Global Routing}
+
\text{Strong Candidate PPO}
+
\text{Deterministic Completion Fallback}
}
\]

其中最重要的责任划分是：

\[
\boxed{\text{数学负责“能不能做”}}
\]

\[
\boxed{\text{RL 负责“现在去哪里最划算”}}
\]

以及：

\[
\boxed{\text{数学保证不能提前退出、不能非法清除、最终可以完成}}
\]

\[
\boxed{\text{RL 专门优化完成时间、信息利用和长期路线效率}}
\]

这就是整个讨论最后应该钉死的核心。

---

# 1. 本次对话首先确认了一件事：当前 Way4 的数学骨架不是废物

当前 Way4 已经有很多正确且值得保留的东西：

- belief；
- set-membership 可行域；
- channel state；
- hard certificate；
- coverage gain；
- certificate manager；
- fallback；
- candidate generator；
- future cost；
- receding horizon；
- executor；
- homing；
- active sensing；
- routing；
- safety；
- residual RL 骨架。

这些模块的价值主要体现在：

\[
\boxed{\text{正确性、可解释性、安全性、可证明性}}
\]

所以 Way4 不应该推倒重来。

真正应该重构的是：

\[
\boxed{
\text{候选生成}
\rightarrow
\text{全局路线建模}
\rightarrow
\text{RL 调度}
}
\]

---

# 2. 当前 Way4 P3 提升不明显，真正原因已经明确

当前 P3：

\[
Way3\approx5180.6s
\]

\[
Way4\approx4983.9s
\]

提升只有约：

\[
3.8\%
\]

问题不是 belief 不够复杂。

也不是 certificate 不够强。

也不是 clear 太慢。

真正瓶颈是：

\[
\boxed{\text{移动}}
\]

当前耗时大致：

\[
T_{\rm move}\approx4000s
\]

\[
T_{\rm measure}\approx600\sim700s
\]

\[
T_{\rm switch}\approx120s
\]

\[
T_{\rm clear}\approx70s
\]

所以：

\[
\boxed{
T_{\rm move}
\gg
T_{\rm sensing},T_{\rm switch},T_{\rm clear}
}
\]

这意味着算法主目标必须首先围绕：

\[
\boxed{\min L_{\rm route}}
\]

组织。

---

# 3. 对方 V4/V6 给出的最关键经验

对方的结果证明了一件非常重要的事情：

> “定位质量更高”不等于“总时间更短”。

例如 V5：

- 首次补测达到 clear threshold 的比例提高；
- 局部定位质量更好；
- 但选择的补测点更远；
- 最终总时间没有改善。

而 V6：

- 首次补测直接 clearable 的比例反而可能更低；
- 测量次数甚至更多；
- 但移动路线更好；
- 最终总时间明显下降。

因此得到：

\[
\boxed{
\text{路线价值}
>
\text{单次定位质量}
}
\]

或者更准确地说：

\[
\boxed{
\text{定位质量必须服从整局路线成本}
}
\]

---

# 4. 我们之前 Way4 最大的数学规划错误

当前 Way4 的 REFINE 流程大致是：

\[
\text{一个 DETECTED channel}
\]

↓

\[
\text{MinimaxNBV.choose()}
\]

↓

\[
\boxed{\text{只得到一个候选位置}}
\]

↓

后面的 planner 再对不同频道之间进行选择。

这个流程存在严重问题。

因为一个频道原本可能有：

\[
q_1,q_2,\ldots,q_{50}
\]

多个有意义的补测位置。

而数学 NBV 提前：

\[
50\rightarrow1
\]

以后，后面的全局 planner 和 RL 再强，也看不到被删掉的那些位置。

所以：

\[
a^\star\notin\mathcal A_t
\]

时，

\[
\boxed{\text{RL 永远不可能学到 }a^\star}
\]

这就是“过早剪枝”。

---

# 5. 第二个重大错误：我们把任务按类别拆开了

当前 FutureCost 的思路大致是：

\[
J
=
J_{\rm route}
+
J_{\rm localization}
+
J_{\rm certificate}
+
J_{\rm exploration}
\]

看起来完整。

但实际上机器人只走：

\[
\boxed{\text{一条物理路线}}
\]

不是四条。

所以真正需要优化的是：

\[
\boxed{
Route(
SEARCH
\cup
REFINE
\cup
CLEAR
\cup
VERIFY
)
}
\]

而不是：

\[
Route(CLEAR)
+
Route(REFINE)
+
Route(VERIFY)
\]

这就是当前 Way4 future-cost 的结构性问题。

---

# 6. 一个地点不是一个“任务动作”，而是一个“多任务服务机会”

这是本次讨论最重要的概念之一。

之前：

```text
REFINE ch4
REFINE ch6
SEARCH ch10
VERIFY ch12
```

被认为是几个独立任务。

但物理世界里，如果它们都可以在同一个位置 \(q\) 完成，

机器人只付一次移动成本：

\[
C_{\rm move}(q)
=
\frac{\|q-x_t\|}{5}
\]

所以真正动作应该是：

\[
\boxed{
a=
GoToAndServe(q,S)
}
\]

其中：

\[
S=
\{
REFINE(ch4),
REFINE(ch6),
SEARCH(ch10),
VERIFY(ch12)
\}
\]

这才是真正的决策单位。

因此：

\[
\boxed{
\text{RL 应该学习位置价值，而不是抽象任务标签价值}
}
\]

---

# 7. 最终 Action 的定义

以后 RL 的候选动作不再是：

```text
REFINE ch4
CLEAR ch6
SEARCH
```

而是：

\[
a_i=
(q_i,S_i,\phi_i)
\]

其中：

- \(q_i\)：目标坐标；
- \(S_i\)：这个点可以完成的服务集合；
- \(\phi_i\)：这个点的完整数学和路线特征。

例如：

```text
candidate #17
point = (823,-417)

services:
    REFINE: ch4,ch7
    SEARCH: ch2,ch8,ch10
    VERIFY: ch12
    CLEAR: none
```

另一个：

```text
candidate #21
point = (-200,910)

services:
    CLEAR: ch6
    REFINE: ch4,ch9
    SEARCH: ch1
```

RL 需要学习：

\[
Q(B,q_i,S_i)
\]

---

# 8. 数学层应该生成多个高价值空间候选，而不是替 RL 选最终点

数学的职责不是：

\[
q^\star=\arg\min U(q)
\]

然后只给 RL 一个结果。

正确职责是：

\[
\boxed{
\text{生成一批合理、有数学依据、具有多样性的候选点}
}
\]

例如每轮：

原始池：

\[
100\sim500
\]

经过合法性、重复、完全支配筛选后：

\[
K=16\sim64
\]

交给 RL。

候选来源包括：

- clear target；
- MEC center；
- feasible polygon centroid；
- bearing perpendicular stand-off；
- 多尺度 ring；
- SEARCH point；
- certificate point；
- 其他频道 clear point；
- 其他频道 refine point；
- 当前 route corridor；
- shared waypoint；
- fallback anchor。

---

# 9. 数学筛选只能删除“明显垃圾”，不能替 RL 决策

可以删：

- 非法点；
- 完全重复；
- 一定没有新信息；
- 已经扫描过且结果确定相同；
- 被另一个 candidate 完全支配。

例如若：

\[
C(a)\ge C(b)
\]

且：

\[
G_j(a)\le G_j(b)
\]

对所有价值维度成立，至少一个严格更差，

则可以删除 \(a\)。

除此之外：

\[
\boxed{\text{不要按一个 scalar score 把候选砍成 1 个}}
\]

---

# 10. 候选必须把所有有价值的信息全部告诉 RL

这是对之前 residual 设计最重要的修正。

不能再只给：

```text
action type
distance
math score
```

而是每个候选必须完整包含：

### 空间信息

\[
\Delta x,\Delta y,d,T_{\rm travel}
\]

### 路线信息

\[
\Delta C_{\rm route}
\]

插入位置；

route rank；

距当前路线距离；

距下一个 task 距离；

### 多任务信息

它可以服务：

\[
哪些频道
\]

哪些是：

```text
SEARCH
REFINE
CLEAR
VERIFY
```

### 定位信息

对每个频道：

\[
IG
\]

\[
E[MEC']
\]

\[
P_{\rm clear}
\]

\[
worst\_case\ shrink
\]

### certificate 信息

\[
CertificateGain
\]

### P4 方向信息

\[
visibility
\]

\[
heading\ uncertainty
\]

\[
directional\ hypothesis\ entropy
\]

### 数学 heuristic

\[
J_{\rm math}
\]

也要给。

但只是 feature。

---

# 11. RL 必须看到 Candidate × Channel 的完整交互矩阵

对于 candidate \(i\) 和频道 \(c\)：

\[
M_{ic}
\]

包含：

- 是否值得测；
- 是否 guaranteed detect；
- expected information gain；
- expected MEC after；
- clear probability；
- certificate gain；
- crossing quality；
- 是否已在此测过；
- 对 P4 的 visibility gain；
- directional ambiguity。

所以 RL 输入不是：

\[
candidate_i
\]

而是：

\[
\boxed{
candidate_i
+
\{interaction(candidate_i,ch_c)\}_{c=1}^{20}
}
\]

这能让网络真正学：

> 一个位置同时对多个频道有价值。

---

# 12. 数学层仍然要计算好所有可以明确计算的东西

RL 不应该浪费容量重新发现：

\[
\pm1^\circ
\]

bearing 几何。

也不应该重新学：

\[
MEC
\]

也不应该重新猜：

\[
route marginal
\]

这些数学可算量应该全部算好。

原则：

\[
\boxed{
\text{数学负责把“局部后果”算清楚}
}
\]

\[
\boxed{
\text{RL 负责学“这些局部后果放在整局里值不值”}
}
\]

---

# 13. 当前 OutcomePredictor 的固定概率方式不应再作为核心

当前类似：

\[
P(detect)=0.85
\]

\[
P(near)=0.10
\]

\[
P(no\_signal)=0.05
\]

这种写死方式不应该继续作为主要规划预测。

P3 应该基于当前：

\[
F_c
\]

采样可能源位置：

\[
p_1,\ldots,p_N
\]

然后枚举：

\[
\epsilon\in\{-1^\circ,0,+1^\circ\}
\]

执行几何更新：

\[
F'_c
=
F_c
\cap
W(q,\theta+\epsilon,\pm1^\circ)
\]

重新计算：

\[
MEC(F'_c)
\]

得到：

\[
E[r'_{\rm MEC}]
\]

\[
P(r'_{\rm MEC}\le20)
\]

\[
E[\Delta diameter]
\]

这才是真正的 candidate feature。

---

# 14. P4 保留 Hard / Soft 双层

P4 中：

NO_SIGNAL 可能是：

- 真不在范围；
- 在范围但定向源背对；
- 其他方向因素。

所以不能像 P3 那样简单：

\[
F_c\leftarrow F_c\setminus B(q,1000)
\]

必须保持：

\[
\boxed{
Hard positive belief
+
Soft directional hypotheses
}
\]

Hard 层决定：

```text
CLEAR
ABSENT
EXIT
```

Soft 层只用于：

```text
规划
候选评分
visibility prediction
```

软层永远不能成为 hard certificate。

---

# 15. 数学保证 100% 完成，RL 不负责学习“别失败”

这是本次讨论最终最核心的原则。

数学层控制：

\[
\mathcal A_{\rm safe}(B_t)
\]

RL 只能：

\[
a_t\in\mathcal A_{\rm safe}(B_t)
\]

例如 EXIT：

只有：

\[
\forall c,\quad
CLEARED(c)
\lor
ABSENT\_CERTIFIED(c)
\]

才生成 EXIT。

否则：

\[
EXIT\notin\mathcal A_{\rm safe}
\]

RL 根本没有提前退出这个权限。

CLEAR 同理。

不满足 clear guard：

\[
CLEAR\notin\mathcal A_{\rm safe}
\]

所以：

\[
\boxed{
\text{安全靠数学动作空间保证}
}
\]

而不是：

\[
\boxed{
\text{靠把网络做小来保证}
}
\]

---

# 16. 这也是之前 tiny residual 设计最根本的问题

之前我做成了：

\[
Q=
Q_{\rm math}
+
\Delta Q_\theta
\]

而：

\[
\Delta Q_\theta
\]

还是：

```text
MLP(64,64)
REINFORCE
zero init
no critic
no GAE
no PPO
```

这相当于：

- 动作空间被数学提前砍小；
- 输入信息被压缩；
- 学习器本身又被做弱。

也就是：

\[
\boxed{
\text{候选限制}
+
\text{信息限制}
+
\text{模型容量限制}
}
\]

同时存在。

这个不应该再作为最终方案。

它只能保留为：

\[
\boxed{\text{轻量 RL 消融 baseline}}
\]

---

# 17. Way2 的失败不能推出 Way2 的网络组件没用

Way2 P4 失败只能说明：

\[
\boxed{\text{Way2 整体闭环不可靠}}
\]

不能推出：

```text
Transformer 不好
PPO 不好
critic 不好
GAE 不好
大网络不好
```

所以最终 Way4 应该复用 Way2 的：

- PPO trainer；
- GAE；
- rollout buffer；
- checkpoint；
- channel transformer；
- normalization；
- batching；
- optimizer；
- gradient clipping；
- logging；
- training infrastructure。

但不要复用：

- Way2 的任意坐标 action head；
- Way2 的旧 observation；
- Way2 的旧安全逻辑；
- Way2 的旧 end-to-end action space。

即：

\[
\boxed{
\text{复用学习能力，不复用错误控制结构}
}
\]

---

# 18. 最终 RL 不应该是 REINFORCE，而应该是 Candidate PPO

正式主学习器：

\[
\boxed{
\text{Masked Candidate Actor-Critic PPO}
}
\]

原因：

episode 长；

长期 credit assignment 强；

很多动作的价值几百秒之后才体现。

因此需要：

\[
V_\psi(B)
\]

和：

\[
GAE
\]

而不是纯 Monte-Carlo REINFORCE。

---

# 19. 强网络结构

最终网络建议：

\[
\boxed{
Channel Transformer
+
Candidate Encoder
+
Candidate-Channel Cross Attention
+
Candidate Self-Attention
+
Actor-Critic
}
\]

### Channel encoder

20 个频道：

\[
x_1,\ldots,x_{20}
\]

通过共享 MLP，再用 Transformer：

\[
H=
Transformer(x_1,\ldots,x_{20})
\]

### Candidate encoder

每个 candidate：

\[
z_i
\]

编码其：

- 坐标；
- route；
- services；
- math score；
- immediate cost；
- aggregate gains。

### Cross attention

candidate 对 20 个频道 interaction：

\[
M_{i1},\ldots,M_{i20}
\]

进行 attention。

### Candidate attention

候选之间也互相比较。

最后输出：

\[
\pi_\theta(a_i|B)
\]

和：

\[
V_\psi(B)
\]

---

# 20. Scale 不应再提前人为限制

“大网络一定更好”不能理论保证。

但：

\[
\boxed{
\text{不能没实验就把网络限制在 }64\times64
}
\]

应该跑 scaling：

小：

\[
d=64
\]

中：

\[
d=128
\]

大：

\[
d=256
\]

比较：

- validation time；
- sample efficiency；
- P95；
- paired win rate；
- inference cost。

如果更大继续提升，就继续扩。

如果饱和，就停止。

结论由数据决定。

---

# 21. Reward 必须直接对齐比赛目标

主 reward：

\[
\boxed{r_t=-\Delta t}
\]

机器人真实花多少秒：

就扣多少。

包括：

- move；
- measure；
- switch；
- clear。

不要把：

```text
IG
MEC shrink
P_clear
certificate gain
```

直接作为主要 reward。

它们应该是 observation/features。

这样网络自己学：

> 这个信息到底值不值得花时间获取。

---

# 22. Full-clear 不是靠 bonus 学出来的

数学层已经控制退出。

所以 correctness 来源是：

\[
\boxed{\text{architecture}}
\]

而不是：

\[
\boxed{\text{reward shaping}}
\]

full-clear bonus 可以保留训练层做 bug 防护。

但不能把：

\[
P(\text{full clear})=1
\]

寄希望于 PPO reward。

---

# 23. 仍然必须有 deterministic fallback

仅仅限制合法动作还不够。

因为 RL 可以合法地“磨蹭”。

因此定义 progress watchdog。

若连续若干 macro step：

- belief 不收缩；
- certificate 不增长；
- clear 数不变；
- coverage 无增长；
- 重复访问；
- 时间预算异常；

则：

\[
\boxed{\text{切回 deterministic completion policy}}
\]

最简单的第一版：

一旦 fallback：

\[
\boxed{\text{本局不再把控制权还给 RL}}
\]

这样最容易验证。

---

# 24. 数学 completion policy 才是真正的 100% 底座

所以 P4 当前：

\[
3/8
\]

或存在 5 个失败 seed，

绝不能说：

> “以后让 RL 修”。

RL 不负责修 correctness。

这些 seed 必须先修 deterministic gap。

直到：

\[
\boxed{Math/Fallback\ 100\%\ full-clear}
\]

才配得上“数学保证”。

P3 和 P4 都必须建立这样的底座。

---

# 25. P3 与 P4 可以共享架构，但不必共享同一个 policy

P3：

全向源，几何更确定。

P4：

加入 directional visibility 和 ambiguity。

所以可以：

\[
\text{shared network backbone}
\]

但建议最终：

\[
\boxed{\text{P3 policy}}
\]

和：

\[
\boxed{\text{P4 policy}}
\]

分别 fine-tune。

不要为了“统一”而损失性能。

---

# 26. 统一全局 route planner 必须成为 deterministic 主基线

在 RL 之前，先建立：

\[
\boxed{Way4-Route-Math}
\]

它必须统一：

```text
SEARCH
REFINE
CLEAR
VERIFY
```

任务。

每个 candidate 的核心 route 特征：

\[
\Delta C_{\rm route}
=
C(T_i)-C(T)
\]

路线 solver：

任务少时：

\[
Held\text{-}Karp
\]

任务多时：

```text
nearest neighbor
cheapest insertion
2-opt
Or-opt
multi-start
```

然后每执行一个 task：

重新：

\[
Observe\rightarrow Update\rightarrow Replan
\]

---

# 27. 必须吸收 V6 的经验，但不要照搬 V6 的局限

V6 正确方向：

\[
\boxed{\text{route-aware candidate}}
\]

但是它仍有长尾：

- 过度等待 SEARCH；
- task ordering；
- dedicated detour。

所以 Way4 应该从一开始加入三个保护。

### WAIT_SEARCH gate

只有等待未来 SEARCH 真正划算时才等。

### Route hysteresis

新路线只好一点点时，不要剧烈重排。

### Detour regret guard

新候选相对保守候选如果额外 route cost 太大：

直接 fallback。

这些属于 deterministic route protection。

---

# 28. FutureCostEstimator 不删除，但降级

现有：

```text
future_cost.py
```

仍然有价值。

可以提供：

- remaining localization；
- remaining certificate；
- estimated clear route；
- coverage burden。

但：

\[
\boxed{\text{不再作为唯一主调度逻辑}}
\]

而是：

\[
\boxed{\text{RL feature / auxiliary estimator}}
\]

---

# 29. Candidate 选择的核心哲学

数学层不要回答：

> “哪个 candidate 最好？”

数学层应该回答：

> “这些 candidate 各自有哪些已知优缺点？”

然后 RL 学：

\[
Q(B,a)
\]

即：

> “在当前整局状态下，这个候选长期到底值多少？”

这就是数学和 RL 最正确的接口。

---

# 30. 最终版本体系固定

以后版本命名固定：

### Way3

老安全基线。

### Way4-Legacy-Math

当前 Way4。

### Way4-Route-Math

统一空间任务和全局路线后的 deterministic 版本。

### Way4-Residual-REINFORCE

现有 tiny residual，作为消融。

### Way4-Final-PPO

最终主模型。

Way2 Full PPO：

历史对照。

不再频繁改主线名字。

---

# 31. 正确的工程实施顺序

### 第一阶段：冻结 baseline

固定：

```text
Way3
Way4-Legacy
Way2 historical
```

固定 seeds 和统计口径。

---

### 第二阶段：修 candidate 接口

最重要：

```text
SpatialCandidate
Candidate × 20 channel features
multi-service waypoint
```

如果一个 waypoint 不能同时表达：

```text
REFINE ch4
REFINE ch7
SEARCH ch10
VERIFY ch12
```

就不要训练 RL。

---

### 第三阶段：Way4-Route-Math

先不碰 PPO。

把：

```text
SEARCH
REFINE
CLEAR
VERIFY
```

统一路线化。

验证：

\[
T_{\rm RouteMath}
<
T_{\rm LegacyMath}
\]

---

### 第四阶段：修 P4 deterministic gap

所有失败 seed：

逐个 trace。

必须先达到：

\[
100\%\ full-clear
\]

---

### 第五阶段：接强 RL

复用 Way2：

```text
Transformer
PPO
GAE
trainer
buffer
normalization
checkpoint
```

接到新的 SpatialCandidate interface。

---

### 第六阶段：Scaling

小、中、大网络。

---

### 第七阶段：Independent test

最终才看 test set。

---

# 32. 统一评测原则

所有算法必须：

\[
\boxed{\text{same seed paired comparison}}
\]

先看：

\[
FullClear
\]

必须：

\[
100\%
\]

否则速度没有资格讨论。

然后看：

\[
Mean
\]

\[
Median
\]

\[
P90/P95
\]

\[
Max
\]

\[
Movement
\]

\[
PairedWinRate
\]

\[
>100s\ regressions
\]

\[
>300s\ regressions
\]

\[
WorstRegression
\]

不能再只看平均时间。

---

# 33. Lower Bound 必须继续保留

已有 evaluate 分支：

```text
LB
Oracle
Math
Hybrid
```

继续使用。

重要指标：

\[
R_{\rm LB}
=
\frac{T_{\rm policy}}{T_{\rm lowerbound}}
\]

这告诉我们：

算法到底距离物理下界还有多远。

---

# 34. 训练集、验证集、测试集必须彻底隔离

至少：

```text
train
validation
test
stress
```

最终 test seed 不得用于：

- reward tuning；
- architecture choice；
- checkpoint selection；
- hyperparameter sweep。

否则最终结果不可信。

---

# 35. 最重要的论文消融

最终至少比较：

\[
Way4\text{-Math}
\]

\[
Way4\text{-RouteMath}
\]

\[
Way4\text{-REINFORCE}
\]

\[
Way4\text{-CandidatePPO}
\]

以及：

### 去 route features

证明路线信息作用。

### 去 multi-service matrix

证明共享 waypoint。

### 去 Transformer

证明跨频道关系。

### 去 math features

证明数学先验。

### small / medium / large

证明模型容量影响。

### PPO vs REINFORCE

证明 actor-critic 和 GAE。

---

# 36. 论文最终方法叙事

整个问题可以表述为：

\[
\boxed{\text{Constrained Belief-Space Routing}}
\]

或者：

\[
\boxed{
\text{Learning-Augmented Safe Active Search and Routing}
}
\]

数学层：

构造：

\[
B_t
\]

以及：

\[
\mathcal A_{\rm safe}(B_t)
\]

RL：

\[
\pi_\theta:
(B_t,\mathcal A_{\rm safe})
\rightarrow
\Delta(\mathcal A_{\rm safe})
\]

优化：

\[
\min_\theta
\mathbb E
\left[
\sum_t\Delta t_t
\right]
\]

约束：

\[
a_t\in\mathcal A_{\rm safe}(B_t)
\]

deterministic fallback 提供 completion。

这是非常干净的论文结构。

---

# 37. 本次讨论中明确判定为错误或废弃的东西

下面这些以后不能再回到主线。

## 37.1 单频道唯一 NBV

废弃。

---

## 37.2 先局部最优，再拼全局

废弃。

---

## 37.3 把 route/localization/certificate 当独立路线相加

不能作为主目标。

---

## 37.4 固定 0.85/0.10/0.05 outcome probability

不能作为核心预测。

---

## 37.5 tiny residual REINFORCE 当最终方案

废弃为主线。

保留消融。

---

## 37.6 因为安全就把网络做弱

明确错误。

---

## 37.7 Way2 整体失败，所以 Way2 网络也不要

明确错误。

---

## 37.8 RL 任意输出坐标

最终主方案不采用。

---

## 37.9 RL 学会“什么时候可以退出”

不采用。

EXIT 由数学层控制。

---

## 37.10 用 reward 去“教育”RL必须全清

不是正确性来源。

---

# 38. 本次对话中确认的最终原则

整个项目以后只遵守下面几条。

### 原则 1

\[
\boxed{
\text{数学提供保证，不提供不必要的能力限制}
}
\]

### 原则 2

\[
\boxed{
\text{RL 看完整信息，不隐藏数学已经计算出的有价值特征}
}
\]

### 原则 3

\[
\boxed{
\text{一个地点可能同时服务多个频道和多个任务}
}
\]

### 原则 4

\[
\boxed{
\text{RL 学的是“去这个位置的长期价值”}
}
\]

### 原则 5

\[
\boxed{
\text{主要性能目标是完成时间}
}
\]

### 原则 6

\[
\boxed{
\text{安全靠 action mask / guard / fallback，不靠弱网络}
}
\]

### 原则 7

\[
\boxed{
\text{candidate generation 要丰富，不能过早剪枝}
}
\]

### 原则 8

\[
\boxed{
\text{模型容量由 scaling benchmark 决定，不提前人为封顶}
}
\]

### 原则 9

\[
\boxed{
\text{没有 same-seed benchmark，不允许换主线}
}
\]

### 原则 10

\[
\boxed{
\text{100\% full-clear 是所有速度比较的前提}
}
\]

---

# 39. 对之前设计失误的最终总结

之前 Way4 RL 的问题不是简单的：

> “REINFORCE 不如 PPO。”

真正的问题是三层一起保守：

\[
\boxed{
\text{候选动作被过早压缩}
}
\]

\[
\boxed{
\text{动作的多任务信息被隐藏}
}
\]

\[
\boxed{
\text{模型容量和训练算法又被主动削弱}
}
\]

最后变成：

\[
\text{强数学 planner}
+
\text{弱 RL 纠偏器}
\]

而本项目真正应该做的是：

\[
\boxed{
\text{强数学安全底座}
+
\text{强 RL 效率优化器}
}
\]

两者各自发挥优势。

---

# 40. 最终冻结一句话

Way4-Final 不再是：

> 数学先决定去哪儿，RL稍微改一下。

而是：

\[
\boxed{
\textbf{
数学不断生成一组安全、合理、信息丰富的空间机会；
RL 看到每个位置对全部频道、全部任务、路线和未来状态的完整价值，
学习下一步去哪里能让整局最早结束；
无论 RL 多差，数学 guard 和 deterministic fallback 都不允许任务主动失败。
}
}
\]

最终目标：

\[
\boxed{
P(\text{full-clear})=100\%
}
\]

在此前提下：

\[
\boxed{
\min T_{\rm finish}
}
\]

这就是本次对话最终得出的完整方案。