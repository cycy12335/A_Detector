# CW-021 三组可复现对比实验

## 实验边界

本包只做安全聚合，不重新训练、不读取或导出原始交易行、不新增 test 评估、不修改服务模型或风险政策，也不调用 DeepSeek。输入冻结在 `configs/cw021_inputs.json`；运行入口是：

```powershell
python scripts/run_cw021_comparisons.py
```

脚本先逐项校验冻结输入 SHA-256，任一文件缺失或漂移即失败关闭；随后写入 `artifacts/reports/cw021/`。输出只含聚合指标、来源摘要和运行元数据，不含原始查询文本、交易行、模型输入或私人路径。

## 组一：历史模型跨来源观察性对比

输入为 CW-003 三基线与 CW-004 LightGBM 的已验收汇总。共同口径是 IEEE-CIS 时间切分、199 个源特征契约、validation 冻结阈值后的一次 test 评估；差异项包括训练工作包、模型家族、预处理与训练实现。由于三基线和 LightGBM 来自不同工作包及候选运行，结果是跨来源观察性比较，不是同一轮重训，也不是单因素模型消融。

输出指标来自历史 `comparison.csv`：precision、recall、F1、FPR、ROC-AUC、PR-AUC、predicted-positive rate、冻结阈值和训练耗时；成本来自各历史报告的 test 成本场景。成本单位是研究用相对单位，不是货币损失，不代表生产收益或风险偏好。

## 组二：validation 阈值与成本

输入只使用 CW-003/CW-004 的 `threshold_curve.csv` validation 行。对每个模型和场景计算：

`total_cost = false_positive × fp_cost + false_negative × fn_cost`

每个场景选择 validation 总成本最低的阈值，成本并列时取更高阈值；同时记录该点的 Recall、FPR、FP、FN、样本数和平均成本。场景为 `fp_fn_1_1`、`fp_fn_1_5`、`fp_fn_1_20`。脚本拒绝读取或使用 test 曲线，阈值选择不回看 test 标签。

## 组三：CW-015 固定合成查询的 top-k

同一次运行使用固定的 `tests/fixtures/cw015/retrieval_queries.json`、同一来源清单、同一离线索引和确定性排序，只改变 `top-k=1/3/5`。指标沿用 CW-015 定义：正例查询平均 Recall@1、Recall@k、MRR，负例空结果率；另记录每个查询的返回证据字符数总和和唯一引用数，再汇总为全部查询与正例查询平均值。输出不保存查询文本或逐查询返回内容。

## 解释边界

模型组不是因果证明，阈值组是 validation 选择行为而不是 test 性能，检索组只说明固定合成查询和审查来源快照下的 top-k 差异。模型分数不是欺诈概率，成本不是人民币/美元损失，匿名字段不赋予业务含义。
