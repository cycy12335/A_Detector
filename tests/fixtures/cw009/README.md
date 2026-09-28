# CW-009 固定回归语料

`model_regression_cases.json` 是 CW-009 的 **golden master**：它冻结了当前已验收的
`cw004-lightgbm-challenger-v1` bundle 与 `cw006-risk-policy-v1` 政策在 12 个**纯合成**输入上的
模型分数与融合决策。

## 纪律

- **全是合成输入**：数值列只用 `0.0` 与少数几个合成常数；五个类别列只用
  `cw009_cat_alpha` / `cw009_cat_beta` 两个占位串（二者都在训练映射中**未见**）。
  语料里没有任何真实交易 ID、标签或原始数据行。
- **值只读一次**：期望值由生产链路（`load_runtime` + `load_policy_engine` + `fuse`）
  一次性只读求得后写入。门禁 runner **没有** `update` / `bless` / `record` / `accept-current`
  模式，失败时绝不重写本文件。
- **改了就要显式评审**：本文件的 SHA-256 冻结在 `configs/quality_gate.yaml` 的
  `inputs.corpus.sha256`。任何改动都必须同时更新配置，否则门禁在加载模型之前就以
  退出码 2 失败。
- **它不证明什么**：这是 golden master，只检测"实现或产物是否偏离已验收基线"，
  **不**证明模型正确、公平、稳定或适合真实风控，也不是在 test 集上重新评估效果。

## 12 个 case 的覆盖

| 角色 | 数量 | 作用 |
| --- | --- | --- |
| `baseline` | 1 | 未触发规则的平静基线，作为所有配对 case 的参照 |
| `boundary` | 6 | `TransactionAmt` 与 `C13` 各自 低于 / 恰好等于 / 略高于 冻结阈值 |
| `stress` | 4 | 模型分数跨越冻结阈值，并与 0 / 1 / 2 条规则命中组合 |
| `shape` | 1 | 五个类别特征换成训练映射未见的第二种合成取值 |

合计覆盖：模型 `flagged` true 与 false；规则命中 0 / 1 / 2 条；`low_signal` /
`elevated_signal` / `high_signal` 三种等级；严格 `>` 语义下"恰好等于阈值不命中"；
以及类别特征在训练映射中未见值的路径。

## 结构

```jsonc
{
  "schema_version": "cw009_model_regression_corpus_v1",
  "kind": "synthetic_golden_master_regression_corpus",
  "disclaimer": "……必须标注 synthetic / golden-master / not-business-correctness……",
  "model_version": "cw004-lightgbm-challenger-v1",
  "policy_version": "cw006-risk-policy-v1",
  "bundle_sha256": "……",          // 必须与配置冻结值一致
  "policy_sha256": "……",          // 必须与配置冻结值一致
  "source_features": ["……199 个，顺序即 bundle 顺序……"],
  "score_absolute_tolerance": 1e-12,
  "cases": [
    {
      "case_id": "case_amt_equal",          // 小写字母/数字/_/-，3–64，唯一
      "role": "boundary",                   // baseline | boundary | stress | shape
      "notes": "……",
      "baseline_case_id": "case_base_calm", // baseline 角色必须为 null
      "varied_features": ["TransactionAmt"],// boundary 必须恰好一个规则特征
      "features": { "……": "199 个键，名称与顺序必须与 source_features 完全一致……" },
      "expected": {                          // 字段集合精确，只允许这七个
        "score": 0.08124042039839917,
        "model_flagged": false,
        "matched_rule_ids": [],
        "unevaluable_rule_ids": [],
        "rule_score": 0.0,
        "signal_points": 0,
        "risk_level": "low_signal"
      }
    }
  ]
}
```

`role` 为 `boundary` 的 case 只允许改动 `varied_features` 声明的那**一个**规则特征，
其余 198 项必须与配对基线逐值一致——这样"低于 / 等于 / 略高于"三个点之间的差异归因才不含糊。

## 相关命令

```powershell
# 只跑预检 + 回归 + 报告（不含 pytest），适合诊断
python scripts/run_quality_gate.py --config configs/quality_gate.yaml --regression-only

# 默认一键门禁（包含完整 pytest）
python scripts/run_quality_gate.py --config configs/quality_gate.yaml
```
