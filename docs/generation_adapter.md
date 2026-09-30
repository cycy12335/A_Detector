# CW-017 受控生成边界

## 当前状态

本包已实现提供方中立的结构化请求/响应契约、离线假传输、有限重试、调用/费用上限、响应大小
限制、空工具白名单和安全降级。当前真实提供方为 DeepSeek API，API 模型 ID 为 `deepseek-flash`
（展示名 DeepSeek-V4.1-Flash）；凭据只从 `DEEPSEEK_API_KEY` 读取，不写入仓库、报告或日志。
CI 和默认测试仍保持离线；只有显式运行真实端到端脚本且进程环境存在凭据时才访问外部模型。

## 数据边界

`build_request()` 只允许以下两类输入：

- CW-016 `DecisionSnapshot` 的显式派生字段白名单。原始特征、交易行、默认 SQLite、身份字段、
  创建时间、请求 ID 和服务载荷不会进入请求。
- CW-015 政策 authority 的已审查片段，且必须匹配决策的政策版本和 SHA-256。请求只发送片段文本、
  citation、版本、定位和标签，不发送完整 README 或未审查路径。

请求 schema 为 `cw017_generation_request_v1`，响应 schema 为 `cw017_generation_response_v1`，
提示词版本为 `cw017_prompt_v1`。`tools` 固定为空数组；请求和响应分别受 12,000 和 8,000 字符
上限约束。

## 控制与降级

`ControlledGenerator` 在每次传输前执行凭据存在性、调用次数、滑动窗口速率和估算费用检查。只对超时和临时不可用
做有限重试；格式错误、认证失败和未知故障不重试。响应 usage 只用于上限核对，不写入业务报告。

所有响应候选都交给 CW-016 `validate_candidate`。校验结果只有 `verified`、
`insufficient_evidence` 或 `rejected`；传输/预算/解析错误统一返回 `degraded`，并只保留稳定错误码
和元数据，不保存原始请求、响应、决策标识、提示词或凭据。

## DeepSeek 真实端到端验证

负责人决策为：预算上限人民币 0.50 元，适配器采用保守 USD 0.07 上限；最多 2 次调用、最多 1 次
重试，60 秒窗口最多 2 次调用；仅发送 `DEFAULT_OUTBOUND_FIELDS` 和已审查 CW-015 政策片段，
不启用工具。`scripts/run_deepseek_generation.py` 使用 DeepSeek Responses API、JSON Schema 和
`reasoning.effort=none`，将单个合成 `complete_low` 案例交给 CW-016 校验。

最近一次真实运行：1 次调用、1 次尝试，状态 `verified`，费用 USD 0.0011514，请求/响应字符数
3050/1513；报告只保留元数据，未保存原始 payload、凭据或决策标识。报告路径为
`artifacts/reports/cw017_deepseek_e2e/report.json`，该目录被忽略，不进入 Git。

## 离线验证

```powershell
python scripts/evaluate_generation.py
python -m pytest tests/unit/test_cw017_generation.py -q
```

`tests/fixtures/cw017/generation_cases.json` 是人工审查的合成故障集，证明离线边界和失败关闭行为；
真实端到端报告只证明本次合成样本的适配与校验链路，不推断生产调查质量。真实调用后仍需独立验收，
且用户已在聊天中粘贴过 API key，建议本次使用后立即撤销并轮换该 key。
