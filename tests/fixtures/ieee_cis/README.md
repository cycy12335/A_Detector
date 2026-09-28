# IEEE-CIS 测试 fixture

**本目录的数据是本项目自行构造的合成数据，不是 Kaggle 的 IEEE-CIS 数据，也不含任何真实交易。**

用途：在不访问网络、不依赖真实大文件的前提下，验证 `CW-002` 审计流程的
Schema 校验、连接审计、时间并列安全切分、列画像与特征角色登记。

## 文件

| 文件 | 行数 | 列数 | 说明 |
| --- | --- | --- | --- |
| `train_transaction.csv` | 40 | 10 | 交易主表；`TransactionDT` 在多个时间点上并列，用于验证切分不会拆开同一时刻 |
| `train_identity.csv` | 26 | 3 | 身份表；只覆盖前 25 笔交易，并**故意包含 1 个孤立键**（`9999999`） |

`train_transaction.csv` 里特意保留了两类边界情况：

- `V2` 列 100% 缺失 → 用于验证"训练分区全缺失 → 排除"的角色规则。
- `P_emaildomain` 含空值 → 用于验证选项缺失率统计。
- `V1`、`id_31` 是匿名风格列名 → 用于验证匿名风险登记文本。

## 与真实数据的关系

这些 fixture 的列名模仿了 IEEE-CIS 的命名风格（`TransactionID`、`TransactionDT`、
`isFraud`、`card1`、`V*`、`id_*`），以便审计代码路径与真实数据一致；
**取值全部由固定种子的合成过程生成**，与真实数据没有任何对应关系。

绝不要把真实 Kaggle 数据复制到本目录：那会把受条款约束的逐行数据带进版本控制范围。
真实数据只能放在 `data/raw/ieee_cis/`，该路径已被 `.gitignore` 排除。

## 重新生成

`train_transaction.csv` 与 `train_identity.csv` 由固定种子合成（seed=7），
可用同一脚本重建；测试不依赖重建结果之外的外部输入。
