# 时间优化清单：训练路径增量验收

本记录仅覆盖已经执行的训练工程修改，不代表总清单完成。

| 项目 | 当前证据 | 未完成部分 |
|---|---|---|
| C01/C02 | `timed_policy_transitions` 对每个策略决策区间计入 -ΔT/1000；终局追加失败惩罚；末区间包含数学回退耗时 | primitive/decision 四时间桶尚未接通；真实 rollout 逐秒对账待执行 |
| C03/C04/C06 | bounded reward 函数存在 | no-progress/repeat 的真实观测与奖励接入待完成 |
| F01/F02 | CLI gamma/GAE 参数实际传入 rollout | 三组 gamma、三组 lambda 实验未执行 |
| AC02 | formal 默认累计至少 2048 transitions 后 update；保留整局；每局保存；超时未足批次不 update；未达计划 updates 时 complete=false | 正式大规模训练未执行；优化器/RNG/未更新 buffer 的恢复待完成 |
| AC03 | 每 epoch 随机 minibatch、末批保留、梯度裁剪；GNN/attention/critic 屏蔽 padding | 真实训练性能验收待执行 |
| AD01/AD02 | formal 默认 train 10000–10199、val 11000–11049 | test 12000–12049 与完整来源隔离校验待接入 |
| E03 | 已接入固定 validation seeds 的 greedy 评估；best_state 仅可由 full-clear/违规计数/完整 seed gate 通过的结果产生；checkpoint 记录 safety_qualified；统一 CLEAR 入口按执行前硬多边形或 near 圆盘证明记录 clear_audit/illegal_clear | safety_violation 仍未完整测量，gate 会拒绝授予安全资格；homing 三角清除尝试存在无证明执行路径，待修复并跑真实验证 |

验证环境：`D:/Anaconda3/envs/gesture_env/python.exe`，PYTHONPATH 包含
`SXJM/Way4`、`SXJM/Way4/src`、`SXJM`。

本轮验证：transition-batch/checkpoint/minibatch 3 passed；
relational-policy/minibatch 5 passed。测试包含真实参数更新、每 epoch 样本全覆盖、
末批保留，以及补零前后 valid logits/critic 相等。

下一依赖：完整时间/progress 审计与安全 validation gate；之后才进行正式训练和消融。

CLEAR 审计增量：pipeline 已启用 strict clear safety；无执行前证明的 CLEAR 会被拒绝且不推进时间，
并写入 rejected audit。executor/homing 回归 16 passed、3 skipped。证明在调用环境前计算；实际命中
不能反向使无证明清除合法；near 证书按实际目标距离检查。

AE01–AE07：新增统一 `compute_efficiency_metrics()` 与 episode 输出字段；当前
no-progress/unnecessary-return 仍使用保守零值，直到 primitive progress/route-regret
数据接通后才可作为正式性能统计。

E01/E02：Candidate-PPO `RewardConfig` 已改为 full-clear 额外奖励 0、失败终局
`-100-N_unresolved`；13 个 RL objective/train/rollout 回归通过。旧 `TabularSMDP`
配置仍保留独立兼容参数，尚未作为 Candidate-PPO 正式训练路径使用。
