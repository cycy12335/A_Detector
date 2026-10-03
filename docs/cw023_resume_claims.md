# CW-023 简历表述与指标复现

本页是简历文案的唯一 claim 清单。`scripts/validate_cw023_claims.py` 会逐项检查文本 token、来源文件 SHA-256、指标路径和限制；任何来源漂移或口径不符都失败退出。

## 中文简历条目

**交易风险识别与 AI 辅助调查（Python / LightGBM / Docker）**：围绕 IEEE-CIS 匿名研究数据的历史冻结评估，构建可离线复现的交易风险识别与证据检索原型；在固定合成回归集上实现 12/12 回归通过、0 个静态违规、回归 12/12（`[CW023-C01]`、`[CW023-C02]`），LightGBM 在历史 test 分区（118,108 条）取得 F1=0.38355（`[CW023-C06]`）；在 9 条固定合成检索查询、top-k=5 下 Recall@5=1.00000、MRR=0.90625、负向空结果率=1.00000（`[CW023-C03]`）。设计来源 SHA、策略版本、事实校验、超时重试上限与降级路径，并用 4 个边界案例 + 2 个负向控制验证 fail-closed 行为（`[CW023-C04]`）；Docker 演示的 6/6 HTTP 检查通过，使用固定 synthetic constant-features 输入，未使用存储或 LLM（`[CW023-C05]`）。

## Concise English version

**Transaction risk detection & AI-assisted investigation (Python / LightGBM / Docker):** Built an offline-reproducible research prototype over frozen IEEE-CIS anonymized data, with 12/12 synthetic regression cases passing and 0 static violations (`[CW023-C01]`, `[CW023-C02]`). LightGBM reported F1=0.38355 on a single historical test partition of 118,108 rows (`[CW023-C06]`). A fixed synthetic retrieval set achieved Recall@5=1.00000, MRR=0.90625, and a 1.00000 negative-empty rate over 9 queries (`[CW023-C03]`). Added source/policy digests, fact validation, bounded retries, and fail-closed degradation; 4 boundary cases plus 2 negative controls passed (`[CW023-C04]`). The Docker demo passed 6/6 HTTP checks with synthetic constant-feature input and no storage or LLM calls (`[CW023-C05]`).

## Claim 复现表

| Claim | 数值/能力 | 分母、分区与版本 | 来源与 SHA | 复现命令 | 前提与限制 |
| --- | --- | --- | --- | --- | --- |
| CW023-C01 | 回归 12/12，0 failed | CW-009 synthetic golden-master；`cw009_model_regression_corpus_v1`；score tolerance `1e-12` | `artifacts/reports/cw009_quality_gate/report.json`；由配置冻结 | `.venv\\Scripts\\python.exe -X utf8 scripts/run_quality_gate.py --config configs/quality_gate.yaml` | 只证明实现未偏离已验收基线，不证明业务准确性；不含真实交易标签 |
| CW023-C02 | 0 个静态违规、回归 12/12 | CW-012 单进程项目门禁；静态规则 `E9,F63,F7,F82,F811,F401` | `artifacts/reports/cw012_project_gate/report.json`；由配置冻结 | `.venv\\Scripts\\python.exe -X utf8 scripts/run_project_gate.py --report artifacts/reports/cw023_project_gate/report.json` | 本地门禁不等于 hosted CI；不宣称生产部署 |
| CW023-C03 | Recall@5=1.00000；MRR=0.90625；negative-empty=1.00000 | `n_queries=9`，8 positive + 1 negative；`cw015-synthetic-reviewed-v1`；固定 index/ranking | `artifacts/reports/cw021/cw021_results.json`；由配置冻结 | `.venv\\Scripts\\python.exe -X utf8 scripts/run_cw021_comparisons.py` | 合成查询与审查源快照；仅比较 top-k，不代表真实线上召回 |
| CW023-C04 | 4 个边界案例 + 2 个负向控制 | BC-001..004；仅合成/离线；无真实 provider payload | `artifacts/reports/cw022/cw022_results.json`；由配置冻结 | `.venv\\Scripts\\python.exe -X utf8 scripts/run_cw022_bad_cases.py` | 证明边界防线与失败闭合，不是生产事故、真实 FP/FN 或模型效果 |
| CW023-C05 | 6/6 HTTP 检查通过 | Docker demo；199 features；constant-features synthetic input；storage=false、llm=false | `artifacts/reports/cw018_docker_acceptance/report.json`；由配置冻结 | `docker compose -f deploy/docker-compose.yml up --build`，再按 `docs/cw020_demo_script.md` | 需要 Docker Desktop、已授权 bundle 和本地端口；验收证据是历史一次演示，不自动代表当前机器可运行 |
| CW023-C06 | F1=0.38355 | IEEE-CIS historical `test` partition，`n_samples=118108`；threshold frozen from validation；single-shot test evaluation | `artifacts/reports/cw004_lightgbm/report.json`；由配置冻结 | `python scripts/train_ieee_lightgbm.py --config configs/ieee_cis_lightgbm.yaml`（仅在具备数据前提时） | 研究指标，不是校准概率、生产收益或新训练结果；不把 CW-021 的比较写成单因素因果结论 |
| CW023-C07 | README-only onboarding；claim/source validator | `.handoff/latest.md` → 当前任务；README 演示路径；CW-018..022 交付链接 | `README.md` 及相关文档；README SHA 同步于 `configs/retrieval_sources.json` | `.venv\\Scripts\\python.exe -X utf8 scripts/validate_cw023_claims.py` | 新用户仍需按文档准备 Docker/bundle/数据前提；P5 是否退出由独立验收决定 |

## P5 交付核对结论

- CW-018 Docker 证据、CW-019 架构文档、CW-020 演示脚本、CW-021 检索比较、CW-022 bad-case 证据均有稳定路径和机器可读报告；README 是新用户入口。
- 当前技术材料已整理为可独立复核的候选 P5 交付，但本轮只置 `waiting_acceptance`。独立验收需再次核对源 SHA、文案限制、README 路径和项目门禁；本页不自行宣布 P5 `passed`。
- 任何简历版本都不得增加“校准概率”“线上收益/ROI”“生产部署”“真实误报/漏报”或“单一因素导致改进”等未被来源支持的表述。
