# CW-012 CI 就绪清单

本文件记录"把本地总门禁搬到持续集成"**还缺什么**。它刻意不做两件事：

- **不声称托管 CI 已经能跑**。写下这些行时既没有代码托管平台，也没有可信的 bundle
  供给方式，所以还没有任何一条流水线被执行过。
- **不预先创建**流水线文件。在平台与产物供给确定之前写出来的工作流，
  只是一份看起来像证据、实则从未运行过的配置。

> **CW-014 更新**：平台已确定为 **GitHub**，工作流文件已按上面的纪律落地在
> `.github/workflows/quality-gate.yml`，并新增 `constraints/cw014-ci.txt` 与
> `configs/ci_inputs.yaml`。工作流是否**真实运行过**仍以托管 run 记录为准——
> 在拿到具体 `OWNER/REPO` 与资产发布授权之前，P3 的 CI 项仍然未完成。
> 详见 `docs/ci_github.md`。

> **2026-09-28 托管验证补记**：上述未解决项是 CW-012 当时的审计结论。
> CW-014 已在私有仓库 `cycy12335/A_Detector` 完成
> [GitHub Actions 成功运行 #36414156557](https://github.com/cycy12335/A_Detector/actions/runs/36414156557)；
> bundle 由私有 Release 供给，工作流报告静态 0 违规、回归 12/12、pytest 1910/0/0/0。
> 现状态为 `waiting_acceptance`，详见 `docs/ci_github.md` 第 6 节。

登记在册的输入由 `project_gate/ci_inputs.py` 的 `CI_INPUTS` 描述；该清单与
`configs/quality_gate.yaml` 的 `inputs` 段由测试断言逐项一致，因此新增必填输入却忘了登记
会被测试抓住。

---

## 1. 固定入口

```powershell
python scripts/run_project_gate.py
```

按 **static → quality_gate** 顺序运行：

| 阶段 | 做什么 | 失败时退出码 |
| --- | --- | --- |
| `static` | 用冻结规则集跑 ruff（`--isolated`，规则显式传入） | 2=发现违规；3=工具失败；4=超时 |
| `quality_gate` | 原样执行 `python scripts/run_quality_gate.py --config configs/quality_gate.yaml`，即完整 preflight → regression → pytest | 5（其自身退出码记在报告的 `quality_gate.exit_code`） |
| 报告 | 原子写入 `artifacts/reports/cw012_project_gate/report.json` | 6=写入失败 |

全量 pytest 只跑一次（由 CW-009 阶段跑）。报告只含阶段、命令形状、退出码、耗时、
违规计数、CW-009 状态/计数与工具版本；不含完整 stdout/stderr、测试节点 ID、
请求/特征值、密钥、绝对路径、bundle 内容或数据库行。

静态规则集（**完整**选择集，没有 ignore / per-file ignores / noqa 兜底）：
`E9`（语法错误）、`F63`（无效转义 / 元组断言 / 可疑比较）、`F7`（语句位置错误）、
`F82`（未定义名称）、`F811`（重复定义）、`F401`（未使用的导入）。
范围：`src/ai_detector`、`scripts`、`performance`、`project_gate`、`tests`、`conftest.py`。

---

## 2. 需要的输入：哪些随代码走，哪些不会

| 输入 | 路径 | 是否提交 | 校验方式 |
| --- | --- | --- | --- |
| 服务配置 | `configs/service.yaml` | ✅ 提交 | 门禁配置冻结 SHA-256，预检按字节比对 |
| 政策 | `configs/risk_policy.yaml` | ✅ 提交 | 同上 |
| 固定回归语料 | `tests/fixtures/cw009/model_regression_cases.json` | ✅ 提交 | 同上 |
| 门禁配置 | `configs/quality_gate.yaml` | ✅ 提交 | 自身即冻结输入清单与容差 |
| 源码 / 门禁实现 | `src/ai_detector`、`project_gate`、`scripts` | ✅ 提交 | 静态检查 + 全量 pytest |
| 依赖与规则 | `pyproject.toml`（extra `static`/`service`/`tree`/`test`） | ✅ 提交 | ruff 用 `==` 精确锁版本；缺失时静态阶段失败关闭 |
| **推理 bundle** | `artifacts/models/cw004/lightgbm_bundle.joblib` | ❌ **被 `.gitignore` 排除** | 门禁配置冻结 SHA-256 `f02bb50e…35a1c`（55,619 B），预检按字节比对 |

默认数据库 `artifacts/runtime/cw007_decisions.sqlite3` **不是**门禁输入：preflight 只做
字节比对，regression 走"规则求值 → 模型评分 → 融合"而不触碰存储。

**结论：干净检出直接跑总门禁一定会失败**，而且失败点在 preflight —— 因为 bundle 不在
版本控制里。这是设计使然（模型二进制不该提交），不是缺陷；但它是 CI 必须先解决的前置条件。

### 实测的两种失败形态

在项目内被忽略的 `artifacts/tmp_tests/cw012/` 里用暂存配置（只改 `bundle.path` 与
`report.path`）实测，真实 bundle 与真实报告**逐字节未动**：

| 场景 | 门禁退出码 | `error_code` | `inputs.bundle` |
| --- | --- | --- | --- |
| bundle 不存在（干净检出） | 2 | `input_missing` | `match: false`，`actual_sha256: null` |
| bundle 存在但字节不符 | 2 | `input_hash_mismatch` | `match: false`，`actual_sha256` 为实际值 |

两种情况下 `regression` 都不运行、`pytest` 为 `{"status": "not_run"}`，
其余三个已提交输入仍为 `match: true` —— 失败精确指向被忽略的 bundle。

### 静态阶段不会伪造整体通过

同一个暂存配置下跑总门禁：`static` 如实 `passed`、`violations: 0`，`quality_gate`
`failed`（exit 2），整体 `status: failed`、退出码 5。静态阶段只报告自己那一格，
不会把整体结论刷成绿的。

---

## 3. CI 可运行前必须先确定的两项（**未解决**）

### 3.1 代码托管与执行平台
当前目录**不是 Git 仓库**（没有 `.git`），也没有任何平台凭据或组织/仓库归属决定。
需要项目负责人确定：托管在哪（GitHub / GitLab / 自建）、由谁拥有、流水线在哪种 runner 上跑、
允许哪些网络出口。在这些确定之前不写任何流水线文件。

> **CW-013 补充**：上表"bundle 是唯一被忽略的**门禁**输入"只对 **preflight** 成立。
> 隔离源码副本实测发现**全量 pytest** 还需要两个被忽略产物：
> `artifacts/models/cw003/rule_baseline_state.json` 与
> `artifacts/runtime/cw007_decisions.sqlite3`（默认库）。它们的可信供给方式同样**未确定**。
> 详见 `docs/bundle_delivery.md` 第 6 节。

### 3.2 已验收 bundle 的可验证供给方式

这是更硬的一条：CI 需要一份与 `f02bb50e…35a1c` **逐字节相同**的 bundle，而当前唯一可信
来源是"在本仓库用 CW-004 训练入口在本机产出"。可选路径各有代价，**均未选**：

| 路径 | 代价 / 风险 |
| --- | --- |
| CI 里重训 bundle | 需要 IEEE-CIS 原始数据（其许可证禁止再分发），且训练确定性、时长与资源都未验证 |
| 把 bundle 提交进仓库 | 与"模型二进制不入库"的既有约定冲突；需要显式评审并调整 `.gitignore` |
| 作为 CI 产物/制品上传下载 | 需要平台侧的制品存储、访问控制与保留策略；且必须先有一次可信产出 |
| 外部对象存储 + 校验 | 需要凭据管理（本包明确禁止引入秘密凭据），并确定谁有权写入 |

无论选哪条，落地时都必须保留**预检的字节级 SHA-256 校验**：CI 只能从可信来源取得，
校验失败必须像现在一样以退出码 2 失败关闭，而不是"找不到就用别处的模型顶替"。

---

## 4. 就绪判定

| 项目 | 状态 |
| --- | --- |
| 冻结、可复现的静态检查 | ✅ 已完成（本地） |
| 单命令总入口 + 稳定退出码 + 安全报告 | ✅ 已完成（本地） |
| CI 输入清单与缺失行为审计 | ✅ 已完成 |
| 干净检出 bundle 缺口失败关闭的实测证据 | ✅ 已完成 |
| 代码托管/执行平台 | ❌ **未确定** |
| 可信 bundle 供给方式 | ❌ **未确定** |
| 在本平台真实运行过一次的流水线 | ❌ **尚不存在** |

因此：**P3 的 CI 项未完成**。本包交付的是"本地总门禁"与"CI 前置条件审计"。
