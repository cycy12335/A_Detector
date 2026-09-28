# CW-014 GitHub Actions 托管质量门禁

本文件记录 GitHub CI 的操作方式、权限边界、资产发布清单和真实托管运行证据。
CW-014 已实施并等待独立验收。

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

## 4. 资产发布记录

| 项目 | 值 |
| --- | --- |
| 文件 | `artifacts/models/cw004/lightgbm_bundle.joblib` |
| 大小 | 55,619 B |
| SHA-256 | `f02bb50e61c5f70c85448430c8bb653bd6e347f400220771500ee84a70135a1c` |
| 发布位置 | 私有仓库 `cycy12335/A_Detector` 的 Release `cw014-ci-inputs-v1`，资产名 `lightgbm_bundle.joblib` |
| 可读范围 | 仅该私有仓库有读权限者（含 Actions 的 `GITHUB_TOKEN`） |
| 必需仓库变量 | `CW014_RELEASE_TAG`（Release 标签；缺失时工作流立即失败） |
| **不发布** | IEEE-CIS 原始/处理数据、默认数据库、任何凭据 |

项目负责人已指定仓库并确认上传。下载已复核大小与 SHA-256；模型二进制未进入 Git。

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

## 6. 真实托管验证

| 项目 | 结果 |
| --- | --- |
| 仓库 | 私有 `cycy12335/A_Detector`，默认分支 `main` |
| 触发与提交 | `push`，`dd2f4de074b440ae1f1c87db61e530435c38380b` |
| 成功运行 | [Actions #36414156557](https://github.com/cycy12335/A_Detector/actions/runs/36414156557) |
| 环境 | Windows Server 2025，Python 3.14.7；ruff 0.14.4，pytest 9.1.1，LightGBM 4.7.0，NumPy 2.5.3 |
| 输入 | bundle `verified`，CW-003 状态夹具 `placed`，默认库 `not_required`；预检四项冻结摘要全部匹配 |
| 阶段 | 静态 `passed` / 0 违规；CW-009 `passed` / 回归 12/12 / 最大偏差 0.0 / 确定性 true；pytest 1910 passed / 0 skipped / 0 failed / 0 errors；总门禁 exit 0 |
| 报告 | 工作流上传 `cw014-reports`，含 CW-009、CW-012、CW-014 三份安全 JSON |

首次 [Actions #36413240635](https://github.com/cycy12335/A_Detector/actions/runs/36413240635)
在全量 pytest 出现 9 个夹具准备错误。原因是 CW-013 隔离副本测试把默认 SQLite
当作必需文件，和 CI 的 `not_required` 输入约定冲突。提交 `dd2f4de` 将默认库快照改为
可选：存在时比较字节摘要，不存在时检查运行后仍不存在；其它冻结输入继续强制校验。

托管计数相对本地 1907 passed / 3 skipped 多 3 项通过、少 3 项跳过，总用例数相同。
本机已提供真实 IEEE-CIS 数据，因此三个“数据缺失时应失败”的用例按条件跳过；
干净检出的托管环境没有这些数据，三个用例正常执行并通过。托管报告仅含状态、摘要、版本与计数。

当前实施状态为 `waiting_acceptance`；独立验收后方可将 CW-014 标记为 `passed`。
