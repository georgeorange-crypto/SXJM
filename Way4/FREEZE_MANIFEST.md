# Way4 Freeze Manifest

状态：implementation-freeze candidate
日期：2026-09-12

## 权威源码根

本清单只冻结 `SXJM/Way4/`。`MathModelingCode/`、`remote_way4_20260912/` 是评测/远程产物，不作为本地运行时源码根。

## 架构入口

- Belief：`src/way4/belief/`
- Certificate：`src/way4/certificate/`
- Planner：`src/way4/planner/`
- Executor：`src/way4/executor/`
- RL augmentation：`src/way4/rl/`
- Tests：`tests/`

## 冻结规则

1. 安全状态由 simulator、set-membership belief 和 certificate 决定；RL 只重排合法 macro candidates。
2. `F_c` 是安全 outer set；`EffectiveRegion` 只用于保守规划/hypothesis 查询，不可替代 hard certificate。
3. 任何 benchmark 必须固定使用：
   `D:\Anaconda3\envs\gesture_env\python.exe -m pytest SXJM/Way4/tests -q`
4. 变更后必须重新记录完整测试输出、跳过项原因和评测 seed split。

## 当前证据

- 已有完整回归记录：`193 passed, 12 skipped`。
- 后续新增功能必须以 targeted regression + 完整回归重新验收。
- 本文件不是性能结果，也不宣称 full-clear、large-seed 或论文级统计已经完成。
- 当前 Git freeze commit 为 `8f09be4`；`scripts/freeze_fingerprint.py` 生成的 `results/way4_source_fingerprint.json` 提供当前 `src/`、`tests/`、`DESIGN.md` 和本 manifest 的逐文件 SHA-256 与 tree fingerprint。该 fingerprint 是可复现快照证据。
