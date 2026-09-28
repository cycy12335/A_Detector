# IEEE-CIS 处理中间层说明

本目录存放 `CW-002` 审计流程生成的**中间产物**。目录内容（除本说明文件外）不被 Git 跟踪。

## 与审计报告的分工

| 位置 | 内容 | 性质 |
| --- | --- | --- |
| `data/processed/ieee_cis/`（本目录） | 切分清单等中间件 | 体积可能较大，可重建 |
| `artifacts/reports/cw002_ieee_cis/` | `audit.json`、`audit.md`、`column_profile.csv`、`split_manifest.json` | 面向验收的可读+机器可读产物 |

## 边界

- 本目录**只允许**存放聚合结果与行级分区标签（如 `TransactionID → 分区`），
  **不得**存放原始逐行特征值、合并后的宽表或任何可直接还原原始交易的副本。
- 本层产物是**建模层的前置输入**：后续工作包只能依据训练分区的统计与语义角色做特征决策，
  不得依据验证集/测试集的标签相关性选列。
- 所有产物必须可重建：删除本目录后重新运行审计命令即可复原。

## 重建命令

```powershell
python scripts/audit_ieee_cis.py --config configs/ieee_cis.yaml
```

## 已知限制

- `TransactionDT` 是**相对时间偏移**，其参考起点未在公开数据中说明。因此本层不得把
  `TransactionDT` 转换为日历日期，任何切分边界都以原始数值形式记录。
- 匿名列（`V*`、`C*`、`D*`、`M*`、`id_*`）的真实业务语义无法从列名确认，本层不为其
  编造含义。
