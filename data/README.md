# data 目录说明

本目录分三层，职责不同，**任何一层都不提交真实的逐行数据**。

| 目录 | 内容 | 是否提交 Git | 说明 |
| --- | --- | --- | --- |
| `generated/` | `CW-001` 模拟交易 CSV | 否（`.gitignore` 排除） | 由 `scripts/generate_data.py` 确定性生成，可随时重建 |
| `raw/` | 用户自行下载的原始公开数据 | 否 | 只读，不修改、不复制、不提交 |
| `processed/` | 审计与切分产生的中间产物 | 否（保留说明文件） | 由 `scripts/audit_ieee_cis.py` 生成，可重建 |

## 为什么原始数据不进版本控制

- IEEE-CIS 数据受 Kaggle 竞赛条款约束，再分发可能违反条款。
- 原始文件体积大，写入 Git 历史后无法真正删除。
- 报告与切分清单都是可再生产物，提交它们只会带来与代码版本不一致的风险。

因此本仓库只提交**说明文件、配置、代码与聚合审计结果**；逐行数据一律留在本地。

## 安全删除本地数据

删除本地数据不会影响代码与测试（测试只用项目自建的小型 fixture）：

```powershell
Remove-Item -Recurse -Force data\raw\ieee_cis
Remove-Item -Recurse -Force data\processed\ieee_cis
```

删除后重新执行 `python scripts/audit_ieee_cis.py --config configs/ieee_cis.yaml`
即可重建 `processed/` 与审计报告（需已重新放置原始文件）。
