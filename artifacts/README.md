# artifacts 目录说明

本目录存放训练流程产生的**可再生成产物**，全部不进入版本控制
（见根目录 `.gitignore`）。删除本目录下除本文件之外的内容是安全的，
按 README 重新执行两条命令即可重建。

## 目录约定

| 子目录 | 内容 | 生成方式 |
| --- | --- | --- |
| `reports/` | JSON 评估报告，例如 `cw001_baseline_report.json` | `python scripts/train_baseline.py --config configs/baseline.yaml` |
| `models/` | joblib 序列化的完整 Pipeline（预处理 + 模型），例如 `logistic_regression_pipeline.joblib` | 同上 |
| `runtime/` | CW-007 的运行时 SQLite 数据库（`cw007_decisions.sqlite3`）。**可变状态的运行时产物**，不是可再生的实验产物——删除它等于清空历史决策记录，服务下次启动会按冻结 schema 重建一个空库 | 由服务自身在 lifespan 启动时创建 |
| `reports/<work_package>_*/` | 各工作包的 smoke 报告（例如 `cw007_persistence/smoke_report.json`） | 对应的 `scripts/smoke_*.py` |

## 为什么不提交这些文件

- 模型二进制与报告会随数据和配置变化反复重建，提交后容易与代码版本不一致。
- 报告中的每个数值都能由脚本复现，因此不需要把结果文件当作“证据”提交。
- 控制仓库体积，避免把大文件写入 Git 历史。
- `runtime/` 下的数据库还可能包含运行时状态（已写入的决策记录），因此更不应该提交：
  它既不可复现，也不是代码或配置。

求职作品集需要展示指标时，应提交**生成指标的命令**和 `WORK_LOG.md` 中的记录，
而不是手工粘贴的截图或数字。
