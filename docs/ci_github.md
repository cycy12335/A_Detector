# CW-014 GitHub Actions 托管质量门禁

本文件记录 GitHub CI 的操作方式、权限边界、资产发布清单，以及**尚未具备的外部条件**。
它在托管 run 真实成功之前不声称 P3 的 CI 项已完成。

- 工作流：`.github/workflows/quality-gate.yml`
- 依赖约束：`constraints/cw014-ci.txt`（全部 `==` 精确锁定）
- 输入清单：`configs/ci_inputs.yaml`
- 输入校验 + 放置：`scripts/ci_fetch_inputs.py`
- 总门禁（工作流里原样运行）：`python scripts/run_project_gate.py`

## 1. 工作流做什么

```text
checkout(fetch-depth 1, 不保留凭据)
  → setup-python 3.14（windows-latest）
  → pip install -c constraints/cw014-ci.txt -e ".[test,service,tree,static]"
  → gh release download <私有 Release 资产> → artifacts/models/cw004/lightgbm_bundle.joblib
  → python scripts/ci_fetch_inputs.py        # 逐字节校验三项输入，失败关闭
  → python scripts/run_project_gate.py       # 静态检查 + 完整 CW-009 链路
  → 上传安全报告（always()）
```

**runner 为什么是 windows-latest**：CW-009 golden master 用 **1e-12** 的绝对容差比较模型
分数，而 LightGBM 的预测不保证跨平台逐位一致。本机已验收基线是 Windows + CPython 3.14；
换平台必须先单独验证回归，不能凭本机结果推断托管通过。

**为什么固定 3.14 而不是 3.14.6**：`actions/setup-python` 的 `3.14` 取该系列最新补丁；
真正决定数值的是约束文件里精确锁定的 numpy/pandas/scikit-learn/lightgbm。
工作流第一步会打印实际解释器版本，run 记录里可核对。

## 2. 权限边界

| 项目 | 取值 | 理由 |
| --- | --- | --- |
| `permissions` | 仅 `contents: read` | 读私有 Release 资产所必需；不写仓库 |
| 引用的 secret | **只有**自动提供的 `secrets.GITHUB_TOKEN` | 不需要也没有配置任何长期凭据 |
| 触发 | `workflow_dispatch` + 默认分支 `push` | 受信任路径 |
| 未使用 | `pull_request` / `pull_request_target` | 外部 PR 因此**拿不到 token、读不到模型资产** |

不受信任 PR 的完整门禁需要一套单独的设计（无模型读取权、或对模型资产只读且隔离的
runner），本包**没有**实现它，也不假装它已经存在。

## 3. 三项被忽略输入的供给

| 输入 | 供给方式 | 缺失/不符时 |
| --- | --- | --- |
| CW-004 bundle（55,619 B，`f02bb50e…35a1c`） | **私有 Release 资产**，`ci_fetch_inputs.py` 逐字节校验 | 退出码 2，任务失败 |
| CW-003 规则基线状态（508 B，`bf152b4b…51eca3`） | **经审查入库的夹具**：`tests/fixtures/cw014/rule_baseline_state.json`，由脚本原子放到 `artifacts/models/cw003/rule_baseline_state.json` | 夹具或目标摘要不符即退出码 2 |
| CW-007 默认决策数据库 | **不再需要**（测试夹具消除依赖） | 若意外存在，仍校验摘要，不符即退出码 2 |

bundle 的路径与摘要来自 `configs/quality_gate.yaml`；`configs/ci_inputs.yaml` 只是**转述**，
脚本会交叉核对，不一致就退出码 3——清单不是第二授权来源。

**CW-014 关闭的干净检出缺口**（前两包遗留）：

1. 全量套件曾依赖开发机上的**默认数据库**。现在"默认库不被改写、也不被凭空创建"
   由根 `conftest.py` 的会话级守卫 `guard_default_database` 负责，在有无该文件的两种
   环境下都成立。落地时发现并修掉了真正的副作用源：`test_cw005_cli.py` 与
   `test_cw006_cli.py` 的探针配置从真实 `configs/service.yaml` 派生却**没有隔离
   `storage.path`**，于是真实 CLI 进程会在干净检出里凭空建出默认库。
2. 全量套件还需要 CW-003 规则基线状态；现在由随源码提交的夹具供给（508 B 纯合成 JSON，
   无路径、无数据、无可执行内容）。

## 4. 资产发布清单（上传前需项目负责人确认）

| 项目 | 值 |
| --- | --- |
| 文件 | `artifacts/models/cw004/lightgbm_bundle.joblib` |
| 大小 | 55,619 B |
| SHA-256 | `f02bb50e61c5f70c85448430c8bb653bd6e347f400220771500ee84a70135a1c` |
| 建议位置 | 目标 GitHub 仓库的**私有** Release，资产名 `lightgbm_bundle.joblib` |
| 可读范围 | 仅该私有仓库有读权限者（含 Actions 的 `GITHUB_TOKEN`） |
| 必需仓库变量 | `CW014_RELEASE_TAG`（Release 标签；缺失时工作流立即失败） |
| **不发布** | IEEE-CIS 原始/处理数据、默认数据库、任何凭据 |

**尚未授权执行**：本包没有创建仓库、没有上传任何资产、也没有配置远端。

## 5. 重跑方式

- 手动：仓库 **Actions → quality-gate → Run workflow**（需 `workflow_dispatch`）。
- 自动：向默认分支 push。
- 本地等价预演（不需要 GitHub）：
  `python scripts/ci_fetch_inputs.py` 然后 `python scripts/run_project_gate.py`；
  完整隔离复现见 `docs/bundle_delivery.md` 的
  `python scripts/verify_isolated_bundle_gate.py --full`。

报告位置（工作流会作为 artifact 上传，保留 14 天）：

- `artifacts/reports/cw012_project_gate/report.json`（阶段与计数）
- `artifacts/reports/cw009_quality_gate/report.json`（回归与 pytest）
- `artifacts/reports/cw014_ci/inputs_report.json`（三项输入的校验结果）

三份报告都只含相对路径、摘要、大小、状态与计数，不含 token、模型内容、原始数据、
请求特征或数据库行。

## 6. 仍未完成的条件（**阻断托管验证**）

| 条件 | 状态 |
| --- | --- |
| CI 平台 | ✅ 已确定：GitHub |
| 仓库 `OWNER/REPO` 与可见性 | ❌ **未提供**（用户已选"私有仓库"，但未给出具体仓库名） |
| 模型资产发布授权（私有 Release） | ✅ 原则同意，❌ **尚未指定目标仓库** |
| 本机 `gh` CLI | ❌ 未安装（创建仓库/上传 Release/触发 run 都需要它或等价的网页操作） |
| 真实托管 run | ❌ **尚不存在** |

**在拿到具体 `OWNER/REPO` 之前无法推送或运行**；按验收标准，本包状态记为 `blocked`
而不是 `waiting_acceptance`——缺少托管 run 记录时不得声称 P3 hosted CI 已完成。
