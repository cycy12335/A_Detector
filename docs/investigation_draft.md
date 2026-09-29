# CW-016 结构化调查草稿与证据校验

CW-016 只定义一个离线、结构化的“候选草稿”契约。输入必须先通过既有
`DecisionRecord`/`StoredDecision` 字段和跨字段不变量校验；随后从 CW-015 索引绑定同一政策版本与
SHA-256 的政策引用。确定性适配器生成的对象和未来任何模型生成的对象都一律视为不可信，必须经过
独立校验器后才可得到 `verified` 或 `insufficient_evidence` 状态。

## 命令

```powershell
python scripts/evaluate_investigation.py
python -m pytest tests/unit/test_cw016_investigation.py -q
```

评测案例位于 `tests/fixtures/cw016/investigation_cases.json`，全部为人工审查的合成决策快照和候选
篡改操作。评测脚本从真实校验结果计算格式成功率、事实一致性和证据完整率及分母，不记录决策 ID、
分数、特征或候选正文。

允许的观察只有四类：模型阈值标记、规则求值状态、研究型信号等级、已知限制。每类观察都有固定的
决策字段白名单和政策 citation ID 集合；额外字段、伪造/漂移引用、事实矛盾、缺观察、版本不符都被
拒绝。规则不可评估时只能标记 `insufficient_evidence`，不能生成完整等级结论。

草稿永远标记 `disposition=research_only`，不会提供批准、拒绝、拦截、欺诈概率或任何生产处置。
`score` 仍是未校准模型分数，`C13` 仍是匿名特征；本包不接入 LLM、网络、默认数据库、真实交易数据
或新的服务接口。真实模型适配、凭据、成本和数据外发边界留待后续明确授权的工作包。
