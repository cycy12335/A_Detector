# CW-021 三组可复现对比实验

## 实验边界

本包只做安全聚合，不重新训练、不读取或导出原始交易行、不新增 test 评估、不修改服务模型或风险政策，也不调用 DeepSeek。输入冻结在 `configs/cw021_inputs.json`；运行入口是：

```powershell
python scripts/run_cw021_comparisons.py
```

脚本先逐项校验冻结输入 SHA-256，任一文件缺失或漂移即失败关闭；随后写入 `artifacts/reports/cw021/`。输出只含聚合指标、来源摘要和运行元数据，不含原始查询文本、交易行、模型输入或私人路径。

## 组一：历史模型跨来源观察性对比

输入为 CW-003 三基线与 CW-004 LightGBM 的已验收汇总。共同口径是 IEEE-CIS 时间切分、validation 冻结阈值后的一次 test 评估；199 个源特征契约只适用于 LogisticRegression 与 LightGBM，Rule 只读 2 个源字段，Dummy 不使用特征值。由于三基线和 LightGBM 来自不同工作包及候选运行，结果是跨来源观察性比较，不是同一轮重训，也不是单因素模型消融。

来源 SHA 的语义被显式区分并由脚本 fail-closed 校验：CW-003 的历史比较表 `source_report_sha256` 是报告文件 SHA；CW-004 的历史比较表该字段实际是历史运行配置 SHA `901517448045ee00efa0684d3e7a1bd86d975e9ade31828bc67cf539d75d5582`，不是报告文件 SHA。脚本同时校验当前配置 SHA、历史报告内 `config.config_sha256`、报告文件 SHA，以及比较表指标行；结果中分别输出 `source_report_sha256_actual`、`source_config_sha256_current`、`source_config_sha256_historical` 与语义字段。

逐模型差异记录如下：

| 模型 | 特征范围 | 预处理 | 训练/拟合条件 |
|---|---|---|---|
| Rule | `TransactionAmt`、`C13` 两个源字段 | train 分区各自估计 0.99 分位数；均值指示器打分 | 无监督尾部启发式，不使用标签、不拟合分类器 |
| Dummy | 不使用特征值 | 处于 CW-003 共享装配管线，但模型忽略输入特征 | `DummyClassifier(strategy=prior, random_state=42)`，只估计 train 类别先验 |
| LogisticRegression | 199 个源特征（194 数值、5 类别）→348 维稀疏输入 | train-only 筛选；数值 median/缺失指示器/标准化；类别 most-frequent/one-hot | `C=1.0`、`class_weight=balanced`、`lbfgs`、`max_iter=3000`、无交互项 |
| LightGBM | 复用同一 199→348 特征契约 | 复用 CW-003 已冻结预处理，不重新定义 | GBDT，1000 棵上限、31 叶、`learning_rate=0.05`；`scale_pos_weight` 由 train 计算，validation 早停 |

输出指标来自历史 `comparison.csv`：precision、recall、F1、FPR、ROC-AUC、PR-AUC、predicted-positive rate、冻结阈值和训练耗时；成本来自各历史报告的 test 成本场景。成本单位是研究用相对单位，不是货币损失，不代表生产收益或风险偏好。

## 组二：validation 阈值与成本

输入只使用 CW-003/CW-004 的 `threshold_curve.csv` validation 行。对每个模型和场景计算：

`total_cost = false_positive × fp_cost + false_negative × fn_cost`

每个场景选择 validation 总成本最低的阈值，成本并列时取更高阈值；同时记录该点的 Recall、FPR、FP、FN、样本数和平均成本。场景为 `fp_fn_1_1`、`fp_fn_1_5`、`fp_fn_1_20`。脚本拒绝读取或使用 test 曲线，阈值选择不回看 test 标签。

## 组三：CW-015 固定合成查询的 top-k

同一次运行使用固定的 `tests/fixtures/cw015/retrieval_queries.json`、同一来源清单、同一离线索引和确定性排序，只改变 `top-k=1/3/5`。指标沿用 CW-015 定义：正例查询平均 Recall@1、Recall@k、MRR，负例空结果率；另记录每个查询的返回证据字符数总和和唯一引用数，再汇总为全部查询与正例查询平均值。输出不保存查询文本或逐查询返回内容。

## 解释边界

模型组不是因果证明，阈值组是 validation 选择行为而不是 test 性能，检索组只说明固定合成查询和审查来源快照下的 top-k 差异。模型分数不是欺诈概率，成本不是人民币/美元损失，匿名字段不赋予业务含义。
