# IEEE-CIS 原始数据放置说明

本目录用于放置 **IEEE-CIS Fraud Detection** 竞赛的原始训练文件。目录内容不被 Git 跟踪。

## 来源

| 项目 | 值 |
| --- | --- |
| 数据页 | https://www.kaggle.com/competitions/ieee-fraud-detection/data |
| 规则页 | https://www.kaggle.com/competitions/ieee-fraud-detection/rules |
| 数据提供方 | Vesta Corporation |
| 竞赛组织方 | IEEE Computational Intelligence Society |

## 许可与条款提醒

- 数据受 **Kaggle 竞赛规则**与数据提供方条款约束，**必须由使用者本人**在 Kaggle 上
  同意条款后才可下载使用。
- 本仓库**不复制** Kaggle 条款全文，也不对其做任何解释或转述；请直接阅读上面的官方规则页，
  自行确认你的使用权限与允许范围。
- 不要把原始文件、其副本或任何逐行样本提交到本仓库或任何公开渠道。
- 本项目仅将其用于求职作品集的**离线研究与工程演示**，不用于任何生产或商业用途。

## 期望文件名

```text
data/raw/ieee_cis/
├── train_transaction.csv     # 必需，约 590k 行、约 394 列
└── train_identity.csv        # 必需，约 144k 行、约 41 列
```

两个文件**都必须存在**：审计流程会校验连接覆盖率与孤立键，缺任意一个都无法完成
`CW-002` 的连接审计。

## 放置步骤

1. 用浏览器登录 Kaggle，打开上面的数据页。
2. 阅读并同意竞赛规则与数据条款。
3. 手动下载 `train_transaction.csv` 与 `train_identity.csv`（无需下载 `test_*` 文件）。
4. 把两个文件放入本目录，文件名保持原样。
5. 回到项目根目录执行审计：

```powershell
python scripts/audit_ieee_cis.py --config configs/ieee_cis.yaml
```

## 执行者（AI 智能体）的边界

- 不索取、不读取、不记录用户的 Kaggle 用户名、密码、Token 或 Cookie。
- 不代替用户登录、点击同意或绕过任何条款与访问控制。
- 不把原始数据复制到仓库范围；报告中只记录**相对路径**，不记录用户主目录等机器专属路径。
- 原始文件缺失时，只报告缺失并停止，**不得**伪造哈希、行数、标签率或覆盖率。
