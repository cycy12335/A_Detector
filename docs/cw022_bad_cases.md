# CW-022 典型 Bad Case、根因与改进记录

## 范围与复现入口

本包只记录合成/离线安全边界，不读取或导出原始交易行、身份信息、原始特征向量或外部模型原始载荷；不新增 DeepSeek 调用、模型重训或 test 指标选择。输入来源和代码/测试 SHA 冻结在 `configs/cw022_inputs.json`，运行入口为：

```powershell
.venv\Scripts\python.exe -X utf8 scripts/run_cw022_bad_cases.py
```

单例可用 `--case BC-001` 至 `--case BC-004`。机器结果写入 `artifacts/reports/cw022/cw022_results.json`，来源清单写入 `artifacts/reports/cw022/cw022_sources.json`。任一冻结来源漂移都会返回 `source_drift:<path>` 并停止。

“Bad Case”在这里表示可复现的边界输入或故障，不等于生产事故；“主动失败关闭”也不等于模型误报/漏报。CW-009 合成回归语料没有真实业务标签，本包不作欺诈误报/漏报结论。

## 案例卡

### BC-001：无匹配证据返回空

- 边界：检索 / 无匹配证据。
- 输入摘要：CW-015 固定夹具中的 `q-unsupported` 合成查询；查询文本不写入结果。
- 预期与观察：`RetrievalIndex.retrieve(..., top_k=5)` 返回空元组，复现结果为 `empty`。
- 根因证据：`tests/unit/test_cw015_retrieval.py::test_retrieval_returns_empty_without_fabricating_evidence`，以及 `src/ai_detector/retrieval/core.py` 的词项检索逻辑。
- 影响边界：无法生成有来源支撑的候选；不会把无证据查询扩写成事实。
- 当前防护：审查来源清单、citation 校验、空结果分支和输出上限。
- 改进状态：已实施防护；后续只需继续维护审查来源与夹具摘要。
- 复验：`.venv\Scripts\python.exe -X utf8 scripts/run_cw022_bad_cases.py --case BC-001`。

### BC-002：伪造/矛盾候选被拒绝

- 边界：候选校验 / 跨字段事实矛盾。
- 输入摘要：离线合成候选把 `flagged` 改成与已校验 DecisionSnapshot 相反；不输出候选载荷。
- 预期与观察：返回 `rejected`，错误码为 `fact_mismatch`，不产生 draft。
- 根因证据：`src/ai_detector/investigation/draft.py` 的独立事实重算与 `tests/unit/test_cw017_generation.py::test_untrusted_candidate_is_rejected`。
- 影响边界：不可信候选不能进入 verified 草稿；这不是生产模型误报/漏报证据。
- 当前防护：严格 schema、跨字段事实校验、引用集合校验和 policy 版本校验。
- 改进状态：已实施防护；后续可扩展更多合成突变，但不改变当前安全结论。
- 复验：`.venv\Scripts\python.exe -X utf8 scripts/run_cw022_bad_cases.py --case BC-002`。

### BC-003：政策摘要漂移在 transport 前失败关闭

- 边界：policy/bundle 资产完整性。
- 输入摘要：合成决策携带错误 policy SHA；不读取或输出原始决策载荷。
- 预期与观察：返回 `degraded`，错误码为 `policy_digest_mismatch`，transport 调用次数为 0。
- 根因证据：`src/ai_detector/investigation/draft.py::_policy_chunks` 与 `tests/unit/test_cw017_generation.py::test_policy_digest_mismatch_degrades_before_transport`。
- 影响边界：调查生成停止，不产生候选；这是配置/资产保护，不是业务模型效果问题。
- 当前防护：来源 manifest SHA、policy version 和 policy digest 三重绑定。
- 改进状态：已实施防护；若未来改变 bundle 格式，必须新增摘要兼容测试。
- 复验：`.venv\Scripts\python.exe -X utf8 scripts/run_cw022_bad_cases.py --case BC-003`。

### BC-004：外部模型连续超时后安全降级

- 边界：外部模型 transport / timeout。
- 输入摘要：离线 fake transport 连续两次抛出 `ProviderTimeout`；不发起真实外部调用。
- 预期与观察：在 `max_retries=1`、`max_calls=2` 下返回 `degraded`，错误码为 `provider_timeout`，尝试和调用均为 2，无 draft。
- 根因证据：`src/ai_detector/investigation/generation.py::ControlledGenerator.generate` 与 `tests/unit/test_cw017_generation.py::test_repeated_timeout_degrades_without_candidate`。
- 影响边界：没有候选草稿；系统保留降级状态，不把超时当作模型结论。
- 当前防护：调用次数、重试上限、超时错误分类和无 draft 返回。
- 改进状态：已实施防护；真实 provider 的网络/费用表现不在本包实测范围内。
- 复验：`.venv\Scripts\python.exe -X utf8 scripts/run_cw022_bad_cases.py --case BC-004`。

## 跨案例规律与改进矩阵

| 改进项 | 覆盖案例 | 状态 | 证据/复验 | 限制 |
|---|---|---|---|---|
| 来源 SHA、policy version/digest 绑定 | BC-001、BC-003 | 已实施 | CW-015 来源测试、CW-017 policy digest 测试 | 依赖来源清单和 bundle 摘要持续同步 |
| 候选独立 schema/事实/引用校验 | BC-002 | 已实施 | CW-016/CW-017 拒绝断言 | 只证明合成候选被拒，不代表真实模型质量 |
| 外部调用预算、重试与超时降级 | BC-004 | 已实施 | CW-017 连续超时测试 | 未做真实 provider 网络或费用实测 |
| 更丰富的可观测性和运营告警 | BC-001–BC-004 | 拟议 | 后续增加稳定事件计数和脱敏告警测试 | 当前不改服务协议、风险政策或外部调用 |

## 结论边界

四例覆盖检索证据、候选可信度、政策/资产完整性和外部 transport 四个系统边界。前三级是主动拒绝或失败关闭，第四例是主动降级；均不是模型预测错误样本，也不能从中推导业务误报/漏报率。
