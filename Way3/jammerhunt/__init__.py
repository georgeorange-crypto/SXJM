"""
jammerhunt —— 无线电干扰源的快速自动定位与清除（第三问 / 第四问）。

模块：
  geometry    纯几何：方位角、交会定位、定位区域(±1°楔形交)与最小包围圆、路径排序
  environment 本包自带的忠实虚拟世界（物理+计时），复现 附件2 §10 计时示例
  interface   World 抽象：LocalWorld / RunnerWorld(接 offline_sim) / HttpWorld(真机)
  coverage    扫描点覆盖保证：全向 1-覆盖、定向 3-覆盖（数值校验）
  agent       策略：覆盖扫描 → 交会定位 → 归航精定位 → 清除（第三/四问统一，参数区分）
  mc          蒙特卡洛评测：全清成功率 + 平均定位-清除时间
  main        连真实模拟器的入口（HTTP）

设计约束：agent 及其依赖(geometry/interface/coverage)仅用标准库，保证正式测试可靠；
numpy/matplotlib 仅用于离线分析与作图，不进主流程。
"""

__all__ = ["geometry", "environment", "interface", "coverage", "agent", "mc"]
