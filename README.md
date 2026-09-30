# AI Detector

面向求职作品集的“交易风险识别与 AI 辅助调查”研究原型。

本目录当前以任务文档驱动开发。执行者开始工作前，必须按以下顺序阅读：

1. [`CLAUDE.md`](./CLAUDE.md)：Claude Code 跨工作包稳定规则与最小上下文流程。
2. [`.handoff/latest.md`](./.handoff/latest.md)：唯一的当前协议任务入口和状态摘要。
3. 当前任务目录的 `status.json`、`next.md`；新任务或整改任务再按协议读取其他任务文件。

`PROJECT_PLAN.md`、`WORK_BOUNDARIES.md` 和 `WORK_LOG.md` 分别保存长期路线、边界与历史；实施者按任务引用读取，
不需要每次全部载入。

## 文档职责

- `PROJECT_PLAN.md` 只描述长期规划，不作为“全部立即实现”的指令。
- `.handoff/latest.md` 和当前任务目录是当前唯一实施范围；没有列入其中的功能不得顺手开发。
- `WORK_BOUNDARIES.md` 的限制优先于实现便利。
- `WORK_LOG.md` 采用追加记录方式，完成工作后必须写入验证命令和实际结果。

## 当前状态

- 项目阶段：P5 作品集交付
- 当前任务：`CW-018` / `TASK-20260930-001`
- 状态：`CW-018` 已完成 Compose 演示实现和本机非容器链路验证，等待 Docker 实测与独立验收；`CW-017` 已独立验收通过
- 规范入口：`.handoff/latest.md` 与 `.handoff/tasks/TASK-20260930-001/`

## 这是什么

`CW-001`（已验收）实现了一个**完全离线、确定性、可复现**的最小分类闭环。
`CW-002`（已验收）接入了 IEEE-CIS 公开匿名数据，建立数据指纹、契约、连接、
时间切分与泄漏审计流程。`CW-003`（已验收）在该固定数据边界上建立了真实数据基线：
显式特征契约、train-only 预处理、透明度规则/Dummy/Logistic Regression 三类基线，
以及只在验证集上冻结的 F1 阈值与相对成本曲线。`CW-004` 在同一数据与特征契约上增加了
**参数预先冻结**的 LightGBM 挑战者、两次无 test 候选运行（A/B）的确定性核对、一次受控
test 评估、四模型来源化对比与源特征级聚合重要性。

```text
CW-001（已验收）：
  生成模拟交易 CSV  →  按时间切分 60/20/20  →  训练 Dummy / LogisticRegression
                    →  验证集选阈值  →  测试集评估一次  →  JSON 报告 + joblib 产物

CW-002（已验收）：
  放置 IEEE-CIS 原始 CSV  →  流式 SHA-256 指纹  →  Schema 与连接审计
                          →  时间并列安全切分  →  列画像与特征角色登记  →  四类审计产物

CW-003（已验收）：
  复用 CW-002 固定切分  →  train-only 列决策与预处理  →  Rule / Dummy / Logistic 基线
                         →  validation 冻结 F1 阈值与相对成本  →  test 一次性评估

CW-004（已验收）：
  复用 CW-003 特征契约  →  两次 train/validation LightGBM 候选运行  →  确定性核对
                        →  单次 test 评估  →  四模型对比 + 聚合重要性 + 推理 bundle

CW-005（已验收）：
  加载 CW-004 bundle（先验 SHA-256）  →  只读 FastAPI 五接口  →  严格 199 列契约
                                    →  请求 ID + 安全错误 + 无载荷日志  →  进程内 smoke

CW-006（已验收）：
  CW-003 train-only 透明阈值  →  版本化两规则引擎  →  模型/规则确定性融合
                              →  研究型风险等级 + policy/risk API  →  独立 smoke

CW-007（已验收）：
  CW-006 求值结果  →  最小化派生记录  →  标准库 sqlite3 单事务写入
                   →  按 UUID4 decision_id 读回存储快照（不重算）  →  独立 smoke

CW-008（已验收）：
  可选 Idempotency-Key  →  同键同规范请求重放 / 异请求 409
                         →  SQLite v1→v2 原子迁移  →  并发安全提交 + 独立 smoke

CW-009（已验收）：
  12 个纯合成固定 case  →  冻结模型/政策 golden master  →  确定性漂移比较
                         →  一条命令运行回归 + 全量 pytest  →  安全 JSON 报告

CW-010（已验收）：
  故障矩阵  →  模型/政策/存储故障注入  →  降级隔离与写入原子性回归

CW-011（已验收）：
  本机回环固定负载  →  Locust 测量  →  P95 与失败率报告

CW-012（已验收）：
  冻结规则集静态检查  →  本地单命令总门禁  →  CI 输入可复现性审计
CW-013（已验收）：
  离线导出交付包  →  按冻结摘要导入  →  隔离源码副本里复现总门禁
CW-014（已验收）：
  GitHub Actions 干净检出  →  校验三项输入  →  原样运行总门禁
```

两段链路使用**不同性质的数据**，不要混淆：

- `CW-001` 使用**模拟数据**，欺诈标签由人为设定的生成模型产生。其指标只用于验证工程
  闭环是否可复现。
- `CW-002` / `CW-003` 使用使用者自行下载的 **IEEE-CIS 公开匿名竞赛数据**。它来自真实机构
  但已匿名化发布，**不能等同于真实银行生产数据**。

> **重要声明**：本仓库的任何指标与结论都**不代表真实银行或真实交易场景的风控效果**，
> 也不能外推到生产环境。`CW-003` 与 `CW-004` 给出的是公开匿名数据上的**基线**，用于证明
> 建模数据流无泄漏、可复现，而不是效果声明；两份报告都未设置任何效果门槛。

本仓库目前**不含**：XGBoost / CatBoost 等其他提升框架、深度学习、**超参数搜索**、
SHAP / Permutation Importance / PDP 或特征重要性的业务解释、概率校准、结果列表与分页查询、
更新与删除、**通用迁移框架**（只有 `cw007_storage_v1`→`cw008_storage_v2` 这一条单向边）、
PostgreSQL、认证/限流/TLS、Docker、CI、压测、漂移监控、模型再训练、RAG 或任何 LLM 调用。

请求幂等**已经实现**（`CW-008`）：两个持久化 POST 接受可选 `Idempotency-Key`，同键同请求重放已提交
快照、同键异请求返回 409 `idempotency_conflict`。但它**不是 exactly-once**——同键并发可能让模型
重复计算，只保证最终只有一组结果落库；key 也没有 TTL、撤销或管理接口。

`CW-005` 提供**只读评分服务**：五个固定接口、严格契约与安全错误/日志，默认仅监听本机。
`CW-006` 在同一服务上加入了版本化透明规则引擎与模型/规则融合（研究型风险等级），并新增
`policy` / `risk` 三个接口。`CW-007` 再加一层**最小化 SQLite 持久化**与按 ID 读回，
新增三个 `/v1/decisions/*` 接口。`CW-008` 为两个持久化 POST 增加可选 `Idempotency-Key`（同键重放、异请求 409），并把存储原子升级到 `cw008_storage_v2`。

需要强调的是：`CW-007` 的持久化**只保存派生结果**（版本、分数、规则求值事实、研究型等级），
不保存 features、请求体、header 或原始 URL。`CW-007` 本身不含请求幂等——**无 key 时至今仍然如此**：
重复提交会创建新的 decision ID 与新记录；幂等只在调用方**显式提供** `Idempotency-Key` 时生效
（`CW-008`）。两个版本都不含列表/分页/更新/删除。整条链路都不产生任何生产处置。

`CW-004` 引入 LightGBM，但**只有一个固定参数的模型**：没有搜索、没有多组参数对比、
不依据任何指标调参；`gain`/`split` 重要性只聚合回源列，并被明确标注为**非因果、非业务含义**。

`CW-009` 已验收：它添加了一组 12 个**纯合成**固定回归 case 与一条一键质量门禁
命令。该 golden master 只用于发现已验收 bundle/policy 的行为漂移，**不**能证明模型正确、公平、
稳定、适合真实风控或生产就绪——它回答的是"实现或产物有没有偏离基线"，不是"模型准不准"。

## 环境要求

- Python 3.11 或更高（本机验证版本：3.14.6）
- Windows PowerShell（命令以 PowerShell 为准；同时给出 Git Bash 等价写法）
- `CW-004` 需要额外的可选依赖 `tree`（LightGBM 二进制 wheel）；`CW-001` 至 `CW-003`
  不需要它，未安装时它们的命令与测试仍可正常运行
- `CW-005` 需要额外的可选依赖 `service`（FastAPI / Pydantic / Uvicorn / HTTPX）；
  未安装时服务专项测试清晰跳过，其余功能与测试不受影响

## 快速开始（Windows PowerShell）

在项目根目录（即本 README 所在目录）执行。**所有命令都必须从项目根目录运行。**

### 1. 创建虚拟环境并安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

要做 `CW-004` 的 LightGBM 实验，再装 `tree` extra（**只用官方 PyPI 的二进制 wheel**）：

```powershell
python -m pip install --only-binary=:all: "lightgbm>=4.5,<5"
python -m pip install -e ".[test,tree]"
```

只做 `CW-001` 至 `CW-003` 时不需要这一步。若你的平台没有兼容的 LightGBM wheel，
请把任务置为阻塞并说明，**不要**从源码编译、降级 Python、另建环境或换用其他模型框架。

若 PowerShell 因执行策略拒绝激活脚本，可不激活，直接使用 `.\.venv\Scripts\python.exe`
替代下文所有 `python`（例如 `.\.venv\Scripts\python.exe -m pytest`）。

### 2. 生成模拟交易数据

```powershell
python scripts/generate_data.py --seed 42 --rows 5000 --output data/generated/transactions.csv
```

参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--seed` | `42` | 随机种子。相同种子必然产出相同内容 |
| `--rows` | `5000` | 生成行数 |
| `--output` | `data/generated/transactions.csv` | 输出路径（相对项目根目录） |
| `--start-timestamp` | `2025-01-01T00:00:00` | 首笔交易时间 |
| `--missing-rate` | `0.02` | 数值特征缺失注入比例，用于验证缺失值处理路径 |

输出列：`transaction_id, timestamp, user_id, device_id, amount,
transaction_count_1h, is_new_device, is_fraud`，按 `timestamp` 升序排列。

### 3. 训练基线模型

```powershell
python scripts/train_baseline.py --config configs/baseline.yaml
```

配置文件 [`configs/baseline.yaml`](./configs/baseline.yaml) 控制随机种子、输入/输出路径、
时间切分比例、阈值候选网格和两个模型的参数。所有相对路径都相对项目根目录解析。

### 4. 运行测试

```powershell
python -m pytest
```

单独运行单元测试（更快）：

```powershell
python -m pytest tests/unit
```

### Git Bash / macOS / Linux 等价命令

```bash
python -m venv .venv
source .venv/Scripts/activate      # macOS/Linux 下为 source .venv/bin/activate
python -m pip install -e ".[test]"
python scripts/generate_data.py --seed 42 --rows 5000 --output data/generated/transactions.csv
python scripts/train_baseline.py --config configs/baseline.yaml
python -m pytest
```

## P1 / IEEE-CIS 数据准备

`CW-002` 接入 **IEEE-CIS Fraud Detection** 公开竞赛数据，建立数据指纹、Schema 与连接审计、
时间安全切分、列画像与泄漏风险登记。**本工作包只做数据工程与审计，不训练任何模型。**

### 1. 自行下载并放置数据

| 项目 | 值 |
| --- | --- |
| 数据页 | https://www.kaggle.com/competitions/ieee-fraud-detection/data |
| 规则页 | https://www.kaggle.com/competitions/ieee-fraud-detection/rules |
| 数据提供方 | Vesta Corporation；竞赛组织方为 IEEE Computational Intelligence Society |

数据受 Kaggle 竞赛规则与数据提供方条款约束，**必须由你本人在 Kaggle 上同意条款后手动下载**。
本仓库不下载数据、不转发数据、不代替任何人同意条款，也不索取或读取你的 Kaggle 账号、密码、
Token 或 Cookie。请直接阅读上面的官方规则页确认你的使用权限。

需要两个训练文件（无需 `test_*`）：

```powershell
# 放置到（相对项目根目录）
#   data\raw\ieee_cis\train_transaction.csv
#   data\raw\ieee_cis\train_identity.csv
```

### 2. 检查本地文件是否存在

```powershell
Test-Path data\raw\ieee_cis\train_transaction.csv
Test-Path data\raw\ieee_cis\train_identity.csv
```

两项都应返回 `True`。任一为 `False` 时审计会以退出码 2 快速失败，并列出缺失项与放置步骤。

### 3. 运行审计

```powershell
python scripts/audit_ieee_cis.py --config configs/ieee_cis.yaml
```

可选参数 `--skip-row-partitions` 用于跳过行级分区表（体积较大，只在需要逐行归属时生成）。

### 4. 查看四类产物

```powershell
Get-ChildItem artifacts\reports\cw002_ieee_cis
```

| 文件 | 内容 |
| --- | --- |
| `audit.json` | 机器可读：文件指纹、Schema、连接审计、切分、特征角色、限制、环境版本 |
| `audit.md` | 面向面试官的可读摘要，区分「已验证事实 / 设计决策 / 无法确认事项」 |
| `column_profile.csv` | 聚合列画像，每列一行（含缺失率、唯一值数、分位数、角色） |
| `split_manifest.json` | 切分算法版本、边界值、各分区行数/比例/标签率 |

行级分区表另写入 `data\processed\ieee_cis\split_manifest_rows.csv`（已被 Git 忽略）。

### 5. 从其他工作目录调用

配置里的相对路径永远**相对项目根目录**解析，因此从别的目录调用时请给绝对配置路径：

```powershell
python D:\ai_detector\scripts\audit_ieee_cis.py --config D:\ai_detector\configs\ieee_cis.yaml
```

此时若原始数据位于项目根目录之外，报告只记录文件名（不记录绝对路径），并用
`path_within_root: false` 显式标注。

### 6. 安全删除本地数据

```powershell
Remove-Item -Recurse -Force data\raw\ieee_cis
Remove-Item -Recurse -Force data\processed\ieee_cis
Remove-Item -Recurse -Force artifacts\reports\cw002_ieee_cis
```

删除后代码与测试不受影响：测试只使用 `tests/fixtures/ieee_cis/` 下**项目自建的合成 fixture**，
不依赖真实数据、不访问网络。

### 关键设计取舍

- **两遍分块读取**：第一遍只读 `TransactionID`/`TransactionDT`/`isFraud` 三列完成契约与连接
  审计，第二遍读全部列做聚合画像。内存占用与 394 列的文件宽度基本解耦。
- **时间并列安全切分**：边界取 `TransactionDT` 的**时间值**而非行号，同一时间点的所有交易必然
  进入同一分区；代价是实际比例可能略微偏离 60/20/20，**边界安全优先于比例精确**。
- **`TransactionDT` 不转换日期**：它是相对时间偏移，参考起点未公开，报告只保留原始数值。
- **角色判定只用训练分区**：`excluded` 的数据驱动规则只看训练分区缺失率，不使用验证集或测试集
  统计，避免把评估分布泄漏进特征选择。
- **匿名列不编造含义**：`V*`/`C*`/`D*`/`M*`/`id_*` 一律登记为「语义匿名，无法仅凭列名排除
  潜在代理或泄漏风险」。
- **只输出聚合统计**：列画像不含任何逐行取值，也不输出邮箱、设备串等原始内容。

### 数据门禁

没有真实数据时可以完成离线实现与 fixture 测试，但**不得**声称工作包完成：

- fixture 测试通过 ≠ 工作包完成。
- 不得生成或填写虚假的真实数据哈希、行数、标签率或连接覆盖率。
- 缺少真实数据时任务状态为 `blocked`，必须等你放置两个文件后重新运行审计。

## P1 / 真实数据基线（CW-003）

在 `CW-002` 已经固定的数据与切分边界上，建立**无泄漏、可复现**的建模流程与三类基线：
透明尾部异常规则、`DummyClassifier`（先验基线）、`LogisticRegression`。
本工作包**不做**超参数搜索、不做特征重要性解释、不训练梯度提升或深度模型。

### 前置条件

必须先完成上面的 `CW-002` 步骤：两个原始文件已放置、且审计已跑过（生成行级切分清单）。
`CW-003` 会校验两个文件的 SHA-256 与切分边界；**任一不符即以退出码 2 失败**，任务应置为
`blocked`，不得替换文件或重算切分。

### 一条命令完成实验

```powershell
python scripts/train_ieee_baselines.py --config configs/ieee_cis_baseline.yaml
```

可选参数 `--chunk-size` 只影响读取分块与内存占用，不影响任何结论。

从其他工作目录调用时，配置里的相对路径仍**相对项目根目录**解析，因此请给绝对配置路径：

```powershell
python D:\ai_detector\scripts\train_ieee_baselines.py --config D:\ai_detector\configs\ieee_cis_baseline.yaml
```

**资源需求（实测）**：在 590,540 × 394 的真实数据上，本机（Windows 11 / Python 3.14.6）
一次完整运行约 **5.5 分钟**、峰值内存约 **4.5 GiB**；其中 `LogisticRegression` 拟合约占 4.5 分钟。
内存不足时可减小 `--chunk-size`，或**降低**配置里的 `train_only_filters.max_missing_rate`
以减少保留列数。注意方向：规则是「缺失率 **大于** `max_missing_rate` 才排除」，因此**降低**
阈值会排除更多列、缩小设计矩阵；提高阈值反而会保留更多列。改变该阈值会改变特征清单，
必须重新记录结果。

### 查看产物

```powershell
Get-ChildItem artifacts\reports\cw003_ieee_cis_baselines
Get-ChildItem artifacts\models\cw003
```

| 文件 | 内容 |
| --- | --- |
| `report.json` | 机器可读：门禁指纹、切分、特征决策、三类基线参数/耗时/收敛、验证集冻结证据、test 一次性指标、相对成本、限制、环境版本 |
| `report.md` | 面向面试官的可读摘要（含成本口径与限制声明） |
| `threshold_curve.csv` | 三个基线在**验证集**上的完整阈值曲线（3 × 99 行） |
| `feature_manifest.json` | 允许清单、训练期排除列及原因、最终维度与稀疏性、派生字段风险 |
| `artifacts/models/cw003/*.joblib` | 完整 Pipeline（预处理 + 模型），可复用训练期统计量 |
| `artifacts/models/cw003/rule_baseline_state.json` | 规则基线的冻结分位数阈值 |

### 安全清理

只删除 `CW-003` 的产物，**不影响**原始数据与 `CW-002` 的审计产物：

```powershell
Remove-Item -Recurse -Force artifacts\reports\cw003_ieee_cis_baselines
Remove-Item -Recurse -Force artifacts\models\cw003
```

重新执行上面的命令即可重建（需原始数据仍在位且指纹未变）。

### 关键设计取舍

- **切分不复用即失败**：分区归属完全来自 `CW-002` 的 `split_manifest_rows.csv`。清单之外的
  连接键会直接报错，而不是就近归类或重新推导边界。
- **所有数据驱动决策只 fit train**：保留/排除列（缺失率、常量、类别基数阈值）、中位数众数、
  缩放参数、规则分位数全部只来自训练分区。有测试通过扰动 validation/test 来守住这条边界。
- **明确特征允许清单**：`configs/ieee_cis_baseline.yaml` 逐列枚举 431 个候选特征并给出理由
  代码，不使用"全部剩余列"。目标列、连接键与原始时间列通过硬校验排除，测试会检查源特征、
  转换后特征名与模型输入三处。
- **稀疏 one-hot 不被稠密化**：`ColumnTransformer` 的 `sparse_threshold` 设为 1.0。
  注意该判定式是 `density < threshold`，设成 0.0 反而会稠密化——这一点在代码注释中写明。
- **阈值只在验证集冻结**：F1 阈值与三个相对成本场景的阈值都在验证集上选出并冻结，
  之后 test 只评估一次。报告中的 `test_access_log` 记录该次访问的时机。
- **成本是研究用相对单位**：`total = FP × fp_cost + FN × fn_cost`，不是货币，也不代表任何
  机构的风险偏好。
- **未收敛不会被隐藏**：逻辑回归的 `ConvergenceWarning` 会被转成报告里的结构化诊断。
  首次运行 `max_iter=1000` 未收敛，已按"与收敛直接相关的有限调整"提高到 3000，调整理由
  写入配置与报告；首次运行的产物保留在 `_run1_nonconverged/` 下，未被覆盖。

## P1 / LightGBM 挑战者（CW-004）

在 `CW-003` 已验收的数据指纹、时间切分、199 列特征契约与 348 维稀疏表示之上，增加**一个参数
预先冻结、无超参数搜索**的 CPU LightGBM 挑战者，并与 `CW-003` 的规则 / Dummy / 逻辑回归
基线做**可追溯的跨来源对比**。

### 前置条件

必须先完成上面的 `CW-002` 与 `CW-003`：两个原始文件在位、审计与基线实验已跑过。
`CW-004` 会硬校验 CW-003 的协议任务状态、建模配置 SHA-256、`feature_manifest.json` SHA-256
与三类正式模型产物；并在重算特征清单后与已验收 manifest **逐项核对**（194 数值 + 5 类别 →
348 维）。任一不符即以退出码 2 失败，任务应置为 `blocked`，不得放宽门禁。

### 一条命令完成实验

```powershell
python scripts/train_ieee_lightgbm.py --config configs/ieee_cis_lightgbm.yaml
```

可选参数 `--chunk-size` 只影响读取分块与内存占用，不影响任何结论。

从其他工作目录调用时，配置里的相对路径仍**相对项目根目录**解析，因此请给绝对配置路径：

```powershell
python D:\ai_detector\scripts\train_ieee_lightgbm.py --config D:\ai_detector\configs\ieee_cis_lightgbm.yaml
```

**资源需求（实测）**：本机（Windows 11 / Python 3.14.6，15.7 GiB 物理内存）三次完整运行的
总耗时为 **80.2 / 93.0 / 103.1 秒**，峰值内存 **4457 / 4566 / 4570 MiB**（进程峰值 RSS）。
以交付那次运行为例：前置门禁 1.0 秒、特征决策 17.5 秒、三个分区装配 9.8 秒、候选运行 A
25.7 秒、候选运行 B 23.0 秒、受控 test 评估 2.1 秒。完整日志见 `WORK_LOG.md`。

单线程（`n_jobs=1`）与 1000 棵树上限都是冻结参数的一部分，**不能**靠加线程或减树来“优化”；
`--chunk-size` 只影响读取分块与内存占用，不改变任何结论。

### 查看产物

```powershell
Get-ChildItem artifacts\reports\cw004_lightgbm
Get-ChildItem artifacts\models\cw004
```

| 文件 | 内容 |
| --- | --- |
| `report.json` | 机器可读：前置门禁指纹、切分与特征契约、完整参数与 LightGBM 版本、两次候选运行的确定性比较、早停证据、test 生命周期门禁与评估历史、规定指标、成本、重要性聚合校验、资源与限制 |
| `report.md` | 面向面试官的可读摘要 |
| `threshold_curve.csv` | **仅 LightGBM 在验证集上**的 99 行阈值曲线 |
| `comparison.csv` | 四模型汇总（无逐行数据），含 CW-003 来源指纹 |
| `feature_importance.csv` | 聚合到**源特征**的 gain / split 重要性（不含类别取值） |
| `artifacts/models/cw004/lightgbm_bundle.joblib` | 推理 bundle：预处理器 + 模型 + best_iteration + 冻结阈值 + 指纹 + 批量打分接口 |

### 安全清理

只删除 `CW-004` 的产物，**不影响**原始数据、`CW-002` 审计产物与 `CW-003` 基线产物：

```powershell
Remove-Item -Recurse -Force artifacts\reports\cw004_lightgbm
Remove-Item -Recurse -Force artifacts\models\cw004
```

### 关键设计取舍

- **参数在第一次真实训练前冻结**：`configs/ieee_cis_lightgbm.yaml` 逐项声明 16 个参数，
  代码里再核对一次键集合与取值。**不做超参数搜索**，也不试多组参数挑最好结果；LightGBM
  即使不优于逻辑回归也照实报告。
- **特征契约不复制、只引用**：`CW-004` 不重写允许清单，而是引用 `CW-003` 的配置（校验
  SHA-256）并复用其 allowlist、train-only 列决策与预处理语义；重算结果与已验收
  `feature_manifest.json` 逐项比对，不一致立即失败。
- **两阶段确定性**：候选运行 A 与 B 都用相同数据、配置与种子重跑一遍 train/validation
  全流程（含预处理器拟合），逐项比较 kept 列、转换维度、`scale_pos_weight`、
  `best_iteration_`、validation 分数、冻结阈值、混淆矩阵与聚合重要性；容差为绝对 `1e-12`。
  **只有比较通过才冻结模型并开放 test**。
- **test 只有带追踪器的受控入口**：装配阶段就把 test 目标列剥离并封存。对 test 分区，
  `features()` 与 `target()` 都**必须**接收生命周期追踪器；冻结检查与访问记账在
  **返回任何数据之前**由入口自己完成，因此"冻结前读不到"与"读到了就一定有日志"同时成立。
  `assembled["test"]` 整体取出被**直接拒绝**；只读元信息走 `structural_ids()` /
  `structural_columns()` / `structural_row_count()`（只有连接键、列名与行数，没有取值）。
  训练流程里不存在"先取数据、再补一条日志"的路径。
- **`single_shot_test` 是单次运行内的语义**：本工作包每一次完整运行都在该次运行内部恰好评估
  一次 test，但这**不**表示整个任务历史只评估过一次 test。报告 `evaluation.test_evaluation_history`
  会列出报告目录下已归档的历史运行，历史原因与命令见 `WORK_LOG.md`。
- **三基线只读不重评**：Rule / Dummy / Logistic 的 test 指标直接从已验收的
  `CW-003/report.json` 读取，并记录该报告的 SHA-256 作为来源指纹。因此四模型对比是
  **跨来源汇总**，不是同一轮重训——报告里明确写出这一点。
- **重要性聚合回源特征**：348 个转换维度按**已拟合预处理器的结构**（不是解析特征名）
  映射回 199 个源列：数值缺失指示维度并入对应数值列，one-hot 维度并入对应类别列；聚合前后
  gain / split 总量在声明容差内守恒。只做 `gain` 与 `split`，输出不含任何 one-hot 类别取值。
- **模型分数不是校准概率**：`scale_pos_weight` 会系统性抬高正类分数（只由 **train** 标签
  计算），因此报告一律称其为“模型分数（score）”，本轮不做概率校准。
- **推理 bundle 不装原始数据**：只封装预处理器、模型、`best_iteration`、有序源特征清单、
  指纹与冻结阈值。批量打分接口对列做严格集合校验：**缺列与多列都会立即失败**，不会静默
  按位置对齐。

## P2 / 只读风险评分服务（CW-005）

把 `CW-004` 已验收的推理 bundle 封装成一个**只读、可测试、默认仅本机监听**的 FastAPI 服务
MVP。它只回答"这笔交易在当前模型下的分数是多少、是否越过冻结阈值"，**不**给出批准/拒绝等
生产处置。

### 安装

```powershell
python -m pip install -e ".[test,tree,service]"
# 等价的最小安装（只跑服务）：
python -m pip install --only-binary=:all: "fastapi>=0.115,<1" "pydantic>=2.10,<3" "uvicorn>=0.30,<1" "httpx>=0.27,<1"
```

未安装 `service` extra 时，服务专项测试会**清晰跳过**（其余测试不受影响）；安装后服务专项
不得跳过。

### 启动

```powershell
python scripts/run_api.py --config configs/service.yaml
```

默认只监听 `http://127.0.0.1:8000`。配置里的相对路径永远相对项目根目录解析，因此从其它
工作目录调用时给绝对配置路径即可：

```powershell
python D:\ai_detector\scripts\run_api.py --config D:\ai_detector\configs\service.yaml
```

> **本服务没有认证、限流与 TLS，且只允许绑定回环地址**（配置层会拒绝 `0.0.0.0` 等对外地址）。
> 它只能在本机或受信内网后面使用，**不得直接暴露到公网**。

### 接口

| 方法 | 路径 | 行为 |
| --- | --- | --- |
| GET | `/health/live` | 进程存活即 200，**不依赖模型** |
| GET | `/health/ready` | bundle 已校验并加载为 200，否则 503 |
| GET | `/v1/model` | 模型版本、bundle SHA-256、分数语义、冻结阈值与 199 个必需特征名 |
| POST | `/v1/score` | 单笔评分 |
| POST | `/v1/score/batch` | 1–100 笔**原子**批量评分，响应顺序与请求一致 |

### curl 示例

先取必需的 199 个特征名（**不要在客户端硬编码列名**）：

```powershell
curl.exe -s http://127.0.0.1:8000/v1/model
```

单笔评分（`X-Request-ID` 可选；缺省时服务端生成 UUID4，响应 header 会回显）：

```powershell
curl.exe -s -X POST http://127.0.0.1:8000/v1/score `
  -H "Content-Type: application/json" -H "X-Request-ID: demo-1" `
  -d "{\"features\": {\"<特征名>\": 0.0}}"
```

批量评分（`client_item_id` 只是响应关联键，**不进入模型、不写日志**）：

```powershell
curl.exe -s -X POST http://127.0.0.1:8000/v1/score/batch `
  -H "Content-Type: application/json" `
  -d "{\"items\": [{\"client_item_id\": \"a\", \"features\": {\"<特征名>\": 0.0}}]}"
```

响应固定包含 `request_id`、`model_version`、`score`、
`score_semantics="uncalibrated_model_score"`、`threshold`、`threshold_policy="validation_f1"`
与 `flagged = score >= threshold`。**没有** `probability` / `fraud_probability` / `approve` /
`decline` 这类字段。

### 状态码与错误

| 状态码 | 含义 |
| --- | --- |
| 200 | 正常 |
| 400 | `X-Request-ID` 不合法（只允许 1–64 位 ASCII 字母数字与 `._:-`）；非法值**不会**被回显 |
| 422 | 请求体 / 契约不满足：额外 envelope 字段、取值类型非法、199 列集合不符、批量超限或重复 `client_item_id` |
| 503 | 模型未就绪（bundle 缺失、指纹不符、依赖缺失或契约校验失败） |
| 500 | 未预期错误（只给通用文案，不含堆栈或内部异常文本） |

错误体统一为 `{"error": {"code": "...", "message": "...", "request_id": "..."}}`，
**不回显**原始 input、特征值、header 集合、绝对路径或堆栈。
校验失败时**零次**评分调用——批量是原子的，不会出现"部分成功"。

> 日志只有两类：每请求一条 `request_completed`（白名单字段、模板路径、无载荷），
> 以及启动时的 `service_startup`。Uvicorn 默认 access log 已被显式关闭，因此进程输出里
> **不会**出现原始 path 或 query string。

### 进程内 smoke

```powershell
python scripts/smoke_service.py --config configs/service.yaml
```

用 `TestClient` 与**真实 bundle** 在进程内跑一遍五个接口，并写出
`artifacts/reports/cw005_service/smoke_report.json`（默认被 Git 忽略）。输入是按 bundle schema
生成的**合成常量特征**，不读取任何真实 CSV 或 test；报告里的分数只是**接口连通性证据**，
**不是**模型效果指标。

### 安全清理

```powershell
Remove-Item -Recurse -Force artifacts\reports\cw005_service
```

不会影响 `CW-004` 的 bundle 或任何 P1 产物。

### 关键设计取舍

- **先验哈希，后反序列化**：`joblib.load` 等价于 `pickle.load`，对不可信文件可以执行任意代码。
  因此服务**先流式校验 SHA-256**（与 `configs/service.yaml` 里写死的 55,619 B 产物指纹比对），
  只有完全一致才反序列化；指纹不符时反序列化**一次都不会发生**（有测试守住）。
  bundle 被视为**受信任的本地产物**——不要用它加载来源不明的 joblib 文件。
- **导入无副作用、每个 app 生命周期只加载一次**：`import ai_detector.service` 不读配置、不加载
  模型、不起服务；加载只发生在应用 lifespan 启动阶段，且只发生一次。
- **运行时可注入**：`create_app(config, runtime_loader=...)`；测试注入 fake runtime，
  因此可以在没有模型的情况下精确断言"零次评分调用"。
- **降级而不崩溃**：bundle 出问题时应用仍可创建，`/health/live` 仍 200，其余接口 503；
  服务**不下载、不重训、不回退**到 Dummy/Logistic 或任何替代模型。
- **199 列严格集合**：features 的键集合必须与 bundle 记录的源特征**完全相等**，缺列、多列
  都返回 422；错误信息只说缺少/多出的**数量**，不回显列名或取值。
- **无载荷日志**：每个请求恰好一条 JSON completion log，只含 request_id、method、route
  （模板路径）、status_code、duration_ms、model_version、batch_size 与 outcome；
  字段走**白名单**过滤，写入 body、features、分数或 client_item_id 都不会被输出。
  未匹配到路由时 route 记为 `<unmatched>`，不写原始 URL。
- **Uvicorn 默认 access log 已显式关闭**（`access_log=False`）：它会原样记录请求行，
  包括未知 path 与 query string，从而绕过上面的模板路径脱敏。关闭后每个请求只剩本服务
  自己的那一条结构化日志——这一点由**真实进程**测试用 sentinel URL 守住。
- **启动诊断**：启动成功与失败各写一条 `service_startup` 结构化日志（成功带模型版本与
  源特征数；失败只带安全分类码如 `bundle_hash_mismatch`，不含路径、配置载荷、异常文本
  或堆栈）。它是独立事件，不计入“每请求一条 completion log”。
- **安全契约写在代码里，配置不得放宽**：批量上限（1–100）、阈值策略（`validation_f1`）、
  分数语义（`uncalibrated_model_score`）与请求 ID 字符规则（1–64 位 `[A-Za-z0-9._:-]`）
  都由模块常量定义并在 `ServiceConfig.__post_init__` 中校验——该钩子在 `from_mapping`、
  `from_yaml` 与 `dataclasses.replace` 三条路径上都会执行，因此改 YAML 或造一个变体配置
  都无法把它们放宽；请求 ID 的校验式还始终编译代码常量。

## P2 / 版本化规则引擎与研究型风险分级（CW-006）

在 CW-005 的只读评分服务之上，加入一个**无状态、确定性、版本化、可解释**的规则层：把模型
的阈值信号与两条**透明尾部规则**融合成研究型 `risk_level`。CW-005 的五个接口与它们的
成功响应契约**逐字保留**。

> 这一层回答的是"模型信号与规则信号如何可解释地组合"，**不是**"该不该批准这笔交易"。
> 它没有数据库、没有持久化、没有幂等，也没有任何生产处置。

### 两条规则从哪来

| 规则 ID | 条件 | 阈值来源 |
| --- | --- | --- |
| `transaction_amt_train_q99_tail` | `TransactionAmt > 1092.0` | CW-003 透明规则基线状态，**train 分区** 0.99 分位 |
| `c13_train_q99_tail` | `C13 > 580.0` | 同上 |

阈值取自 `artifacts/models/cw003/rule_baseline_state.json` 在**训练分区**上算出的分位数，
不是在 validation/test 上挑出来的，也**不是**任何机构的业务规则。`C13` 是匿名特征，
本项目**不为它赋予任何业务含义**。两条规则只读取政策声明的那两列，不修改 199 列模型输入。

### 冻结语义（改配置也放宽不了）

- **运算符是严格 `>`**：恰好等于阈值（`1092.0` / `580.0`）**不命中**；
- `rule_score = 命中规则数 / 2`；
- 融合：模型 `flagged` 计 **2** 点，每条命中规则计 **1** 点，共 0–4 点：

  | 信号点 | `risk_level` |
  | --- | --- |
  | 0 | `low_signal` |
  | 1–2 | `elevated_signal` |
  | 3–4 | `high_signal` |

- **不可评估 → `indeterminate`**：任一规则字段为 `null`、字符串、非有限数或整列缺失时，
  **不做字符串到数字的隐式转换**，该规则记为 `evaluable=false`，整体 `risk_level` 为
  `indeterminate` 且 `signal_points` 为 `null`，并在 `unevaluable_rule_ids` 中列出。
  这是**合法且显式**的求值结果（HTTP 200），**不是** 422——"无法判断"和"请求写错了"是两回事；
- `policy_semantics` 固定为 `research_signal_band`：这是研究型信号分级，**不是**概率，
  也**不是**批准/拒绝建议。响应里刻意没有 `approve` / `decline` / `block` / `allow` /
  `fraud_probability` 这些字段。

### 三个新接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/v1/policy` | 政策版本、SHA-256、固定语义、规则元数据、融合区间与免责声明（不含路径/载荷） |
| POST | `/v1/risk/evaluate` | 单笔：模型评分 + 规则求值 + 风险等级 |
| POST | `/v1/risk/evaluate/batch` | 1–100 笔**原子**批量，响应顺序与请求一致 |

请求 envelope、199 列严格集合与取值类型契约与 CW-005 完全一致。

```powershell
# 政策元数据
curl.exe http://127.0.0.1:8000/v1/policy

# 单笔风险求值（features 需为完整的 199 列；此处省略）
curl.exe -X POST http://127.0.0.1:8000/v1/risk/evaluate `
  -H "content-type: application/json" `
  -d '{\"features\": { \"TransactionAmt\": 2000.0, \"C13\": 0.0, ...其余 197 列... }}'
```

单笔成功响应含：`request_id`、`model_version`、`score`、`score_semantics`、`threshold`、
`threshold_policy`、`flagged`、`policy_version`、`policy_sha256`、`policy_semantics`、
`rule_score`、`matched_rule_ids`、`unevaluable_rule_ids`、`rule_evaluation_complete`、
`signal_points`、`risk_level`。批量顶层只给公共模型/政策元数据，逐项给各自的等级结果，
**不回显** `features` 或任何特征取值。

### 政策故障时的降级

政策缺失、SHA-256 不符或 schema 违规（额外字段、错误版本、被改过的阈值等）时：

- 应用**仍然启动**，CW-005 的 `/v1/score`、`/v1/score/batch`、`/v1/model` 照常可用；
- `GET /v1/policy`、两个 `/v1/risk/*` 接口与 `/health/ready` 返回 **503 `policy_not_ready`**；
- **绝不**回退到内置默认政策、旧政策或其它阈值——宁可明确不可用，也不给出来源不明的判定依据；
- 启动诊断会报出安全分类码（如 `policy_hash_mismatch`），只写分类码，不写路径或配置内容。

> 政策**未配置**（`configs/service.yaml` 里没有 `policy` 段）与政策**加载失败**在对外行为上
> 都是 503，但 `config.public_dict()` 里的 `policy_configured` 能把两者区分开。

### 政策文件与安全加载

`configs/risk_policy.yaml` 是**冻结政策**，`configs/service.yaml` 用固定 SHA-256 把它钉住。
加载顺序与 bundle 同源：**先流式校验 SHA-256，通过后才 `yaml.safe_load`**；指纹不符时
YAML 解析**一次都不会发生**（有测试用 monkeypatch 守住）。政策路径必须是项目相对路径，
绝对路径、盘符路径、`..` 逃逸与疑似凭据路径一律拒绝。

schema 是**严格**的：多一个字段、少一个字段、规则顺序变化、重复 rule ID 或重复 feature、
未知 operator、非有限阈值、与 CW-003 不一致的阈值、被放松的融合真值表，全部在加载阶段
被拒。代码里的硬不变量（`src/ai_detector/rules/config.py`）是唯一权威，政策文件只能**复述**它。

### 验证

```powershell
python -m pytest
python scripts/smoke_service.py --config configs/service.yaml
python scripts/smoke_decision_service.py --config configs/service.yaml
python scripts/run_api.py --config configs/service.yaml
```

`smoke_decision_service.py` 用**真实 CW-004 bundle + 冻结政策**在进程内跑六步
（live / ready / policy / 旧 score / 单笔 risk / 两笔 batch risk），写出
`artifacts/reports/cw006_decision/smoke_report.json`（默认被 Git 忽略）。报告只记录版本、指纹、
状态码、字段名、一致性与耗时：**不含** 199 列载荷、任何特征取值、绝对路径或凭据。
其中 `risk_level` 只作为"模型信号 + 规则信号"链路的**一致性证据**，不是模型效果指标。

### 安全清理

```powershell
Remove-Item -Recurse -Force artifacts\reports\cw006_decision
```

不会影响 `CW-004` 的 bundle 或任何 P1 产物。

### 关键设计取舍

- **纯函数求值**：`PolicyEngine.evaluate_rules()` 无 I/O、无全局状态、无随机性，不修改输入
  映射，也不访问标签或训练/test 数据（有测试在删除政策文件后仍要求结果不变）。
- **融合是独立纯函数**：`fuse(result, flagged=...)` 只依赖冻结真值表，因此 0–4 点可以逐例
  断言，而不是靠端到端抽查。
- **没有动态规则平台**：运算符写死为严格 `>`，没有运算符分派、没有表达式求值、没有
  `eval`/`exec`、没有热更新。政策文件无法被扩展成一个可执行 DSL。
- **引擎输出要被校验**：引擎若给出"既命中又不可评估"、未知或重复规则 ID，或 `matched` /
  `evaluable` 不是**真正的布尔值**（整数 `1`/`0`、字符串、`null` 都算），服务返回 500
  `policy_contract_violation` 而不是把可疑结果当成合法事实——`1` 在真值判断里会被静默当成
  "命中"，这类结构异常必须在**模型调用之前**就失败（批量因此保持零次评分、无部分结果）。
- **政策路径边界在加载函数自身**：`load_policy()` **无条件**先做项目相对路径校验，且先于任何
  文件 I/O；判定与平台无关（盘符、UNC、扩展长度前缀、POSIX 绝对、根相对、`..` 逃逸、疑似凭据
  一律拒绝）。生产入口 `load_policy_engine()` 传的是**项目相对路径**，且加载函数**没有**任何
  放宽该检查的开关——不存在"底层可绕过"的第二条路径。测试需要项目外文件时也不放宽生产入口，
  而是把临时政策写进项目内、已被 Git 忽略的 `artifacts/tmp_tests/`（会话结束自动清理）。
- **批量三段式**：先全部契约校验，再全部规则求值，最后**恰好一次** `score_many`。任一契约
  错误时模型评分**零次**、规则求值零次，且不返回部分结果。
- **政策每个 app 生命周期只加载一次**，请求期间零次读文件；`create_app` 本身仍然无副作用
  （不读配置、不加载模型或政策），且允许分别注入 fake runtime 与 fake policy engine。
- **日志仍走白名单**：政策/风险路由额外记 `policy_version`、`risk_level`（单笔）与
  `rule_match_count`。**不记**规则 ID 列表、`rule_score`、特征值或载荷；批量没有"一个"风险
  等级，因此不记 `risk_level`，只记命中总数。CW-005 五个接口的日志字段集合逐字不变。

### 研究限制

- 两条阈值是公开匿名数据上的**统计演示**，不是业务规则、不是欺诈概率、不是因果结论；
- `risk_level` 只是把"模型信号 + 规则信号"组合起来的一种**透明方式**，
  不能读成模型效果、风险排序能力或生产适用性；
- 本工作包**不含**真实机构规则、外部规则源、动态规则 DSL、数据库与持久化、结果查询、
  幂等键、认证/限流/TLS、生产处置建议、Docker/CI 与压测。

## P2 / 决策结果持久化与按 ID 查询（CW-007）

在 CW-006 的无状态风险求值之上，加一层**标准库 SQLite、最小化记录、原子事务、可重启读取**的
持久化边界：成功求值的结果获得服务端 UUID4 `decision_id`，之后可按 ID 读回**存储快照**。
查询**不重新评分、不重新运行规则**。

> 这一层只回答"结果如何最小化、原子地落盘，并在进程重启后按 ID 原样读回"。
> 它**不做幂等**，也没有列表/分页/更新/删除/迁移，更不产生任何生产处置。

### 三个新接口

| 方法 | 路径 | 正常状态 | 行为 |
| --- | --- | --- | --- |
| POST | `/v1/decisions/evaluate` | 201 | 复用单笔 risk 请求，求值并提交一条记录 |
| POST | `/v1/decisions/evaluate/batch` | 201 | 复用 1–100 批量请求，一次模型调用、单事务写入全部记录 |
| GET | `/v1/decisions/{decision_id}` | 200 / 404 | 读取存储快照，不调用模型/规则 |

响应统一嵌套：单笔 POST / GET 为 `{"request_id": ..., "decision": {...}}`，
批量为 `{"request_id": ..., "decisions": [...]}`。

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/decisions/evaluate `
  -H "content-type: application/json" `
  -d '{\"features\": { \"TransactionAmt\": 2000.0, \"C13\": 0.0, ...其余 197 列... }}'

curl.exe http://127.0.0.1:8000/v1/decisions/<decision_id>
```

### 只存派生结果

记录恰好包含 20 个字段，全部是**求值产生的派生结果**：

`record_schema_version`、`decision_id`（服务端小写 UUID4）、`created_at`（UTC RFC 3339 `Z`）、
`evaluation_request_id`、`client_item_id`（单笔为 `null`）、`model_version`、`score`、
`score_semantics`、`threshold`、`threshold_policy`、`flagged`、`policy_version`、
`policy_sha256`、`policy_semantics`、`rule_score`、`matched_rule_ids`、
`unevaluable_rule_ids`、`rule_evaluation_complete`、`signal_points`、`risk_level`。

**不存**：`features`、199 列名集合、任何特征值、请求/响应体、header 集合、原始 path/query、
数据库或配置路径、凭据、异常文本、真实 CSV 行、标签，以及批准/拒绝/拦截等处置字段。

其中 **`flagged` 是独立的模型阈值判定**，写作 `flagged = (score >= threshold)`（边界是 `>=`）。
它与规则是否可评估**无关**：规则因缺列不可评估时 `signal_points` 为 `null`、`risk_level` 为
`indeterminate`，但 `flagged` 仍然是一个已知的真值。因此记录构造要求调用方**显式传入**这个
布尔值，并与交给 `fuse()` 的是**同一个**值——不允许从可空的融合点数反推，也不允许在两侧各算一次。
`flagged == (score >= threshold)` 是**写入与读回都会强制**的跨字段不变量：违反它的记录构造不出来，
被直接篡改的行读回时折叠为 `storage_contract_violation`（HTTP 500），不会被真值转换或静默修补。

### 冻结的存储契约

| 项目 | 固定值 |
| --- | --- |
| storage schema | `cw007_storage_v1`（同时写入元数据表） |
| record schema | `cw007_decision_record_v1` |
| SQLite `user_version` | `1` |
| 连接 PRAGMA | `foreign_keys=ON`、`busy_timeout=2000`、`journal_mode=DELETE`、`synchronous=FULL` |
| 数据库位置 | 项目相对 `artifacts/runtime/<文件名>.sqlite3`（唯一允许的形状） |

`artifacts/runtime/` 之外、其它扩展名、盘符/UNC/根相对/`..` 逃逸、疑似凭据路径，
以及经**符号链接或重解析点**解析后逃出项目根的路径，一律在初始化前被拒绝。
配置解析与 `create_app()` **不产生任何文件 I/O**：目录与数据库只在 app lifespan 的存储
初始化阶段创建。

已有数据库按**结构完全相等**校验，而不是"必需的东西在不在"：

- `user_version` 必须等于冻结值；
- 用户表集合必须**精确等于** `decisions` 与 `storage_metadata`；
- 两张表的**完整规范 DDL** 必须逐字相等——`sqlite_master.sql` 只折叠空白后与冻结常量
  `CANONICAL_DECISIONS_DDL` / `CANONICAL_METADATA_DDL` 做精确比较，因此**拿掉**一个原有约束与
  **增加**一个额外 CHECK / UNIQUE / 表级选项同样会被识破（建表语句与比对基准出自同一份常量）；
- 两张表的列**顺序/类型/NOT NULL/主键**逐项匹配；
- 元数据表只能含冻结的 `schema_version` 一条记录；
- 不得出现任何视图、触发器或手工索引（SQLite 为 PRIMARY KEY 自建的 `sqlite_autoindex_*`
  是内部对象，予以放行）。

任一不符都判定为结构漂移：**失败关闭**，不删除、不覆盖、不静默迁移（文件字节保持不变）。

### 原子性与顺序

单笔/批量的处理顺序是冻结的：

1. Pydantic envelope 与取值类型校验；
2. 组件就绪检查（模型 → 政策 → 存储）；
3. 全部 199 列集合与规则输出契约校验；
4. 批量**恰好一次** `score_many`；
5. 构造并严格校验全部记录（含跨字段不变量）；
6. **一个** SQLite 事务写入并 commit；
7. **commit 成功之后**才构造响应。

任一请求/特征/规则/记录契约错误：模型评分与数据库写入都是零。任一批量 insert/commit 失败：
事务全部回滚、返回统一错误、**没有**部分记录，也**不会**退化成"只返回未持久化结果"。

### 故障隔离与错误码

| 状态码 | 错误码 | 含义 |
| --- | --- | --- |
| 503 | `storage_not_ready` | 存储未配置、路径非法、库打不开或 schema 不符 |
| 503 | `storage_unavailable` | 运行期读写失败（磁盘/锁/约束冲突） |
| 500 | `storage_contract_violation` | 存储中的记录结构已损坏 |
| 404 | `decision_not_found` | ID 不存在**或形状非法**（不回显原始 path 参数） |

- 仓库配置**启用了**存储，因此 `/health/ready` 只有在 model + policy + storage 全部就绪时才 200；
- 存储故障时 **CW-005/CW-006 的八个旧接口按原能力继续工作**，只有三个 `/v1/decisions/*`
  与 readiness 降级；**不会**改用内存数据库或临时文件；
- 已有记录在**模型或政策未就绪时仍可读取**（GET 只依赖存储）；
- schema 被篡改（版本、表、列、CHECK 约束）时**失败关闭**：不删除、不覆盖、不静默迁移。

### 无 key 时不幂等

相同输入、甚至相同 `X-Request-ID` 的重复提交会创建**新的 UUID4** 与**新记录**（`X-Request-ID`
至今不是幂等键、也不参与请求指纹）。本包不对 request ID 建唯一约束、不缓存响应、
也不声称 exactly-once。

> 后续进展：`CW-008` 已在**本包冻结的记录契约**之上实现了可选的 `Idempotency-Key` 幂等
> （见上文 CW-008 章节）。它只改变"调用方显式给出 key"时的行为；**无 key 路径与本节的描述
> 完全一致**，未做任何改动。

### 验证

```powershell
python -m pytest
python scripts/smoke_service.py --config configs/service.yaml
python scripts/smoke_decision_service.py --config configs/service.yaml
python scripts/smoke_persistence_service.py --config configs/service.yaml
python scripts/run_api.py --config configs/service.yaml
```

`smoke_persistence_service.py` 用**真实 bundle + 冻结政策**和一个**本次运行专用**的
一次性数据库跑六步：live/ready → 单笔持久化并读回 → **重建 app** 后仍可读 → 两笔批量与
逐条查询 → 旧 `/v1/risk/evaluate` 契约不变 → 清理。报告写到
`artifacts/reports/cw007_persistence/smoke_report.json`（默认被 Git 忽略），其中**不含**
decision ID、request/client ID、分数、记录内容、features、特征取值或数据库路径。
运行结束后该一次性数据库及其 sidecar 会被删除。

### 安全清理

```powershell
Remove-Item -Recurse -Force artifacts
untime          # 运行时数据库（可再生成）
Remove-Item -Recurse -Force artifacts
eports\cw007_persistence
```

删除数据库**不会**影响模型、政策或既有报告；服务下次启动会按冻结 schema 重新建库。

### 关键设计取舍

- **数据库是可变运行时产物**：不使用 SHA-256 固定；可信度来自固定 schema、CHECK 约束与
  **读取时重新校验**（UUID、UTC 时间、有限数、真布尔、JSON 列表、`flagged == score >= threshold`、
  等级与点数的跨字段一致性）。损坏的行不会被真值转换或静默修补，只会得到
  `storage_contract_violation`。
- **模型标记只算一次、不从融合点数反推**：`_build_record()` 计算一次 `flagged = score >= threshold`，
  把**同一个布尔值**同时交给 `fuse()` 与 `StoredDecision`。`RiskAssessment.signal_points` 在规则
  不可评估时是 `null`，用它反推会把"标记但不可评估"错写成"未标记"——这条路径现在有专门的
  `indeterminate` 单笔/批量反例守着（旧 risk、新 POST、数据库行、GET 四者逐字段一致）。
- **已有库按结构完全相等校验**：表集合精确等于两张冻结表、**两张表的完整规范 DDL 逐字相等**、
  列顺序/类型/NOT NULL/主键逐项匹配、元数据仅一条冻结记录、无视图/触发器/手工索引（SQLite 内部
  `sqlite_autoindex_*` 放行）。多一张表、改一个列属性、或**多一个 CHECK/UNIQUE/表级选项**都
  失败关闭，且失败时文件字节不变。
- **连接是逐操作短生命周期的**：每次读写自建连接并在 `finally` 关闭，不保存全局可变连接、
  不跨线程复用；写入用 `BEGIN IMMEDIATE` … `commit()` 的显式事务。
- **SQL 全部参数化**：唯一拼进语句的是冻结的列名元组与冻结的整数 `user_version`。
  带引号、SQL 关键字或超长 decision ID 只会得到安全 404，不会改变 schema。
- **列表只编码为确定性 JSON**：不使用 pickle/joblib/YAML，读回时严格校验元素类型与唯一性。
- **ID 与时钟可注入**：`create_app(..., decision_id_factory=..., utc_clock=...)`，
  因此记录在测试里是逐字段可断言的常量。
- **日志仍走白名单**：新接口最多增加 `storage_operation`（`read`/`write`）与 `record_count`；
  **不记** decision ID、request/client ID、数据库路径、SQL、分数、规则 ID 或记录内容。
  CW-005/CW-006 八个旧接口的日志字段集合逐字不变。

### 研究限制

- 存储里的字段都是研究型信号，不是概率、不是业务规则、也不是生产处置依据；
- 单机作品集服务：不承诺多进程高并发、跨主机一致性、灾备或加密；
- 本工作包**不含**幂等、列表/分页/搜索、更新/删除、保留期、通用迁移、PostgreSQL、
  ORM、认证/限流/TLS、Docker/CI 与压测。

## P2 / 持久化请求幂等与 SQLite v2 迁移（CW-008）

在 `CW-007` 已验收的决策快照之上，为两个持久化 POST 增加**可选、持久化、并发安全**的请求幂等，
并把存储从 `cw007_storage_v1` 原子升级到 `cw008_storage_v2`。**不新增任何路径**：仍然只有 11 条。

> 本包保证的是**已提交结果的幂等**，不是模型只运行一次，也不是端到端 exactly-once。
> 同键并发可能让模型重复计算，但 SQLite 事务保证只有一组结果能落库。

### 可选请求头

| 场景 | 状态 | body | `Idempotency-Replayed` |
| --- | --- | --- | --- |
| 无 `Idempotency-Key` | 201 | CW-007 原 body，每次新建记录 | 不返回 |
| 合法 key 首次提交 | 201 | CW-007 原 body | `false` |
| 同键同请求重放 | 201 | 与首次逐字段一致 | `true` |
| 同键异请求或异 operation | 409 | 统一错误 envelope | 不返回 |
| 非法 / 重复 key header | 400 | `invalid_idempotency_key` | 不返回 |

- header 可选；`Idempotency-Key` 合法值必须是 **16–128 个 ASCII 字符**，正则
  `^[A-Za-z0-9][A-Za-z0-9._:-]{15,127}$`。取值**大小写敏感**，不 trim、不做 Unicode 归一化，
  同一请求出现多次即 400（"用哪一个"没有合理默认）。
- 建议调用方使用高熵随机值（如 UUID）；服务**不替调用方生成**幂等键，也**没有 TTL**。
- `X-Request-ID` **不是**幂等键、不参与请求指纹：换它仍是同一个请求；无 key 时相同的 request ID
  依旧各建一条记录。

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/decisions/evaluate `
  -H "Content-Type: application/json" `
  -H "Idempotency-Key: 3f1c9a7e-52b4-4d18-9c33-0a7e6d1b8f20" `
  -d '{\"features\": { \"TransactionAmt\": 2000.0, \"C13\": 0.0, ...其余 197 列... }}'
```

### 请求指纹

规范化规则写死在代码里（`ai_detector.storage.idempotency`），因为幂等的全部语义都压在
"这两个请求是不是同一个请求"上：

- 对象 key 按 Unicode code point 排序，JSON 用 UTF-8、`ensure_ascii=False`、无多余空白；
- features 的**输入顺序无关**；批量 item 顺序、`client_item_id`、字符串与 `null` **强相关**；
- 有限数统一转 float（`1` ≡ `1.0`），`-0.0` 规范为 `0.0`；`true`/`false` 是布尔，**绝不**与
  `1`/`0` 归一；NaN/Infinity 拒绝；
- `operation`（`decision_single` / `decision_batch`）进入指纹，单笔与批量不共享命名空间；
- 不含 `X-Request-ID`、`Idempotency-Key` 或任何其它 header/path/query。

只持久化两个 64 位摘要（`key_digest`、`request_fingerprint`）与**有序** decision ID 关联。
**不存**原始 key、features、请求体或 header。摘要会泄漏"两个输入是否相等"，**不是**匿名化、
不是加密，也不是不可逆的安全凭据。

### SQLite v2 与唯一允许的迁移

| 项目 | 值 |
| --- | --- |
| storage schema | `cw008_storage_v2`（`user_version = 2`） |
| legacy storage schema | `cw007_storage_v1`（`user_version = 1`） |
| record schema | `cw007_decision_record_v1`（**未变**） |
| 用户表 | `decisions`、`storage_metadata`、`idempotency_requests`、`idempotency_results` |

`decisions` 的 DDL、20 个字段与既有行**逐字未变**；升级只做三件事：建两张幂等表、把元数据唯一那行
改成 `cw008_storage_v2`、把 `user_version` 提升到 `2`——全部在**一个** `BEGIN IMMEDIATE` 事务里，
commit 之后还要按完整 v2 契约**复验**一次（"写完了"不等于"写对了"）。

只接受三种启动状态：**新文件**（直接建严格 v2）、**严格 v2**（只验证，零写入）、
**完整合法的 v1**（迁移一次）。其它版本、结构漂移或损坏文件一律失败关闭——不猜测、不覆盖、不删除、
**不跳迁**。迁移前会逐行确认既有决策记录**读得出来**：结构完全相等救不了被改坏的行。
事务中途失败只回滚自己那个事务，绝不留下"半迁移"（不会出现只有一张新表或版本已升而表未建）。
**不是**通用迁移框架：没有 registry、没有 v0、没有降级命令、没有 Alembic。

### 新增错误

| 状态码 | 错误码 | 含义 |
| --- | --- | --- |
| 400 | `invalid_idempotency_key` | key 缺失字段以外的形状/重复问题 |
| 409 | `idempotency_conflict` | 同键已用于另一个请求 |
| 500 | `idempotency_contract_violation` | 已持久化的幂等映射或关联损坏 |

SQLite 锁/磁盘/事务错误仍是 503 `storage_unavailable`；迁移或 schema 错误仍是
`storage_not_ready`。**重放只依赖存储**：模型或政策未就绪时，已提交的 keyed 请求仍可重放；
未命中的 key 则按原语义降级（503）。冲突在已有映射可读时**发生在模型与规则之前**。

### 并发边界

- 同键同请求并发：可能**重复评分**，但只有一个请求能把候选记录写进去；其余在事务内看到已提交映射
  后改为安全重放。所有成功调用返回同一组快照，至多一个结果标记为 `created`。
- 同键异请求并发：一方创建，另一方必须 409；数据库只含胜者的完整结果。
- 一句话：**只保证"提交结果唯一"，不保证"计算唯一"**。

### 验证

```powershell
python -m pytest
python scripts/smoke_service.py --config configs/service.yaml
python scripts/smoke_decision_service.py --config configs/service.yaml
python scripts/smoke_persistence_service.py --config configs/service.yaml
python scripts/smoke_idempotency_service.py --config configs/service.yaml
```

`smoke_idempotency_service.py` 先建一个**严格 v1** 的一次性数据库并种入一条由真实政策与真实模型
求值产生的决策，再让服务把它迁移到 v2，然后走完 keyed 单笔/批量首次与重放、冲突、无 key 兼容与
清理；报告写到 `artifacts/reports/cw008_idempotency/smoke_report.json`（Git 忽略），其中**不含**
幂等键、摘要、指纹、request/client/decision ID、分数、规则 ID、features 或数据库路径。

### 安全清理

删除运行时数据库**不会**影响模型、政策或报告；服务下次启动会按冻结 schema 重新建库
（已迁移过的库不会被再次迁移，因为它是严格 v2）。

```powershell
Remove-Item -Recurse -Force artifacts\runtime
Remove-Item -Recurse -Force artifacts\reports\cw008_idempotency
```

### 研究限制

- 幂等只覆盖**已成功提交**的数据库结果，不改变"分数不是概率、等级不是处置"的研究语义；
- 单机 SQLite，不承诺跨进程/跨主机强一致、多进程高并发或灾备；key 无 TTL、无撤销、无清理；
- 本工作包**不含**列表/分页/搜索、更新/删除、通用迁移、PostgreSQL、ORM、Redis、认证/限流/TLS、
  Docker/CI 与压测。

## P3 / 固定模型回归集与一键质量门禁（CW-009）

### 一条命令

```powershell
python scripts/run_quality_gate.py --config configs/quality_gate.yaml
```

它按固定顺序做三件事，**全部成功才退出 0**：

1. **preflight** —— 以**二进制**方式核对 `configs/service.yaml`、`artifacts/models/cw004/lightgbm_bundle.joblib`、
   `configs/risk_policy.yaml`、`tests/fixtures/cw009/model_regression_cases.json` 四个输入的 SHA-256。
   它先**收齐**四个实际摘要并逐个记录 `actual_sha256` / `match`，再统一判定；只要有一个不符就
   **不加载模型、不解析 YAML/JSON、不推理、不跑 pytest**，直接以退出码 2 失败关闭。
   配置与语料在**任意层级**出现重复映射键（相同值或冲突值）同样在这里失败关闭——解析器不会
   静默保留"最后一个值"；
2. **regression** —— 用生产顺序（规则求值 → 规则结果结构校验 → 模型评分 → `flagged = score >= threshold`
   → 融合）对 12 个纯合成 case 求实际结果，与语料冻结的期望逐值比较，并在同一进程里**再跑一遍**
   做确定性核对；
3. **pytest** —— 用当前解释器 `sys.executable -m pytest` 运行**完整**测试套件（固定 cwd 为项目根、
   有限超时、不使用 shell）。

诊断时可以用 `--regression-only`：它仍然执行完整的 preflight + regression + report，只是跳过 pytest。
**没有**跳过哈希、放宽容差、更新 oracle 或"接受当前结果"的参数。

### 退出码

| 退出码 | 含义 |
| --- | --- |
| 0 | 全部通过 |
| 2 | 配置 / 预检 / 语料契约失败 |
| 3 | 回归或确定性失败 |
| 4 | pytest 非零（含子进程无法启动） |
| 5 | pytest 超时（子进程被 kill 并回收） |
| 6 | 报告写入失败（含报告脱敏断言不通过） |

未预期的内部错误**也失败关闭**：它按发生阶段返回该阶段的退出码，报告里的 `error_code` 为
`internal_error`，且只记录异常**类别名**——绝不写 traceback 或消息正文。

### 报告字段

报告只写 [artifacts/reports/cw009_quality_gate/report.json](artifacts/reports/cw009_quality_gate/report.json)（Git 忽略），
用临时文件 + `os.replace` **原子写入**：

| 字段 | 内容 |
| --- | --- |
| `work_package` / `schema_version` / `generated_at_utc` | 工作包、报告 schema（`cw009_quality_gate_report_v1`）、UTC 时间 |
| `overall_status` / `exit_code` / `error_code` | 总体状态、冻结退出码、稳定错误码 |
| `stages` | 三个阶段各自的 `passed` / `failed` / `skipped` 与错误码 |
| `inputs` | 四个输入的**相对路径**、冻结摘要、实际摘要与是否匹配（失败时同样列出全部存在输入） |
| `versions` | 模型版本、政策版本、分数语义、阈值策略 |
| `corpus` | case 数、源特征数、容差、各角色 case 数 |
| `coverage` | flagged true/false 数、命中规则数分布、出现的风险等级、每条规则的阈值探针 |
| `regression` | 通过/失败 case 数、失败 case ID 与**字段名**、最大分数绝对偏差、双跑确定性、分布核对 |
| `pytest` | 命令的安全形状（解释器与 cwd 为占位符）、退出码、耗时、子进程是否回收、通过/跳过/失败/错误计数 |
| `environment` | Python、平台与关键依赖版本 |

报告**不含**任何 feature 取值、完整 feature 映射、请求体、数据库行、UUID、幂等摘要、绝对路径、
bundle 内容、pytest 节点 ID 或捕获的 stdout/stderr。case ID 与安全字段名允许出现。这条约束不靠
"写代码时小心"：报告在落盘前会被再查一遍，查出违规就失败关闭（退出 6）。

### Golden master 是什么意思

`tests/fixtures/cw009/model_regression_cases.json` 里的期望值是**一次性地、只读地**用当前已验收的
bundle + policy 求出来再冻结的。runner 里**没有** `update` / `bless` / `record` / `accept-current`
模式——失败时它绝不会重写语料或配置。语料的字节哈希写在 `configs/quality_gate.yaml` 里，因此
任何改动都必须显式评审后同时更新两边，而不是被门禁悄悄"接受"。

12 个 case 全部是**人工合成的占位取值**（数值列为 0.0 或少数几个合成常数，五个类别列只用两个
合成占位串），不含任何真实交易 ID、标签或原始数据行。它们合计覆盖：模型 flagged true/false、
规则命中 0/1/2 条、`low_signal`/`elevated_signal`/`high_signal` 三种等级、`TransactionAmt` 与 `C13`
各自低于/等于/略高于冻结阈值（**严格 `>`**：恰好等于不命中）、以及类别特征在训练映射中**未见**的取值。

> 这份 golden master 检测的是"实现或产物是否偏离了已验收基线"。它**不是**模型效果、公平性、
> 稳定性或业务正确性的证据，也不是在 test 集上重新评估效果。

### 失败排查

| 症状 | 先看哪里 |
| --- | --- |
| 退出码 2，`error_code` 为 `input_hash_mismatch` | `inputs` 里哪个输入 `match: false`；**四个存在输入的 `actual_sha256` 都会列出**，确认没有人改动过 bundle / policy / 服务配置 / 语料 |
| 退出码 2，`error_code` 为 `corpus_schema_violation` / `corpus_contract_violation` | 语料被改坏（重复映射键、字段集合、特征顺序、case ID、边界配对）——门禁不会自动修复它 |
| 退出码 2，`error_code` 为 `config_schema_violation` | 配置被改坏（重复映射键、未知/缺失字段、错类型、宽松容差、非冻结的 pytest 参数） |
| 退出码 2，`error_code` 为 `agreement_violation` | 配置与语料/运行时对同一件事说法不一致（容差、case 数、版本、摘要） |
| 退出码 3，`error_code` 为 `regression_mismatch` | `regression.failed_case_ids` 与 `failed_fields`：哪些 case 的哪个字段漂移了 |
| 退出码 3，`error_code` 为 `regression_nondeterministic` | 同一进程两次评分结果不同——这是独立的失败类别，不会被逐值比较掩盖 |
| 退出码 3，`error_code` 为 `regression_baseline_mismatch` | 逐 case 都对，但聚合分布与 frozen expected 不一致 |
| 退出码 4 / 5 | `pytest` 阶段的 `exit_code`、`counts`、`child_reaped` 与超时设置 |

报告只给 case ID 与**字段名**，不给取值；要复现具体差异，请在本机用生产链路对同名 case 手工求值。

### 验证

```powershell
# CW-009 专项
python -m pytest tests/unit/test_cw009_quality_config.py tests/unit/test_cw009_corpus.py `
  tests/unit/test_cw009_regression.py tests/unit/test_cw009_report.py `
  tests/unit/test_cw009_pytest_runner.py tests/unit/test_cw009_gate.py `
  tests/integration/test_cw009_quality_gate.py

# 默认一键门禁（真实运行完整 pytest）
python scripts/run_quality_gate.py --config configs/quality_gate.yaml
```

### 安全清理

报告是可再生成的产物，删除不影响任何冻结输入：

```powershell
Remove-Item -Recurse -Force artifacts\reports\cw009_quality_gate
```

### 已知限制

- 门禁是**单机、单进程**的：不承诺并发、分布式或跨机器一致性；
- 语料只覆盖 12 个合成点，**不代表**输入空间；它检测漂移，不度量覆盖率或鲁棒性；
- 分数比较用固定的**绝对**容差（`1e-12`），配置层面还锁死上限 `1e-6`——不接受相对容差；
- 本工作包**不含** CI、容器、压力/并发测试、漂移监控与模型再训练。

## P3 / 故障注入与降级回归（CW-010）

CW-005 到 CW-008 各自验证过"自己那部分坏掉时会降级"。CW-010 补的是**跨组件、跨接口**
的那一层：多个组件**同时**坏掉时，readiness 报哪个阻塞项、哪些接口必须继续工作、
请求期的失败会不会留下半条记录。本包**不修改生产代码**——所有场景都由测试证明既有实现
已经满足契约；没有任何一处需要修复。

### 故障矩阵

矩阵写在 `tests/cw010_fixtures.py` 的 `FAULT_MATRIX` 里，是**可被测试遍历的数据**：
`tests/unit/test_cw010_matrix.py` 校验它的完整性（必需注入点一个不少、状态码与错误码
出自冻结词表、日志字段全在白名单内、受影响与未受影响接口不相交、readiness 优先级可由
行数据推导），集成测试则直接**消费**同一份数据去发请求。因此"文档里写的"和"测试跑的"
不会各自漂移。

| 阶段 | 注入点 | 受影响接口 | 状态码 / 稳定码 |
| --- | --- | --- | --- |
| 启动 | bundle 缺失/指纹不符/损坏 | `/health/ready`、`/v1/model`、评分、风险、decisions 写入 | 503 `model_not_ready` |
| 启动 | 政策加载失败 | `/health/ready`、`/v1/policy`、风险、decisions 写入 | 503 `policy_not_ready` |
| 启动 | 存储加载失败 | `/health/ready`、三个 `/v1/decisions/*` | 503 `storage_not_ready` |
| 启动 | 模型 + 政策 | `ready` 报 `model_not_ready`；`/v1/policy` 仍报**自己的** `policy_not_ready` | 503 |
| 启动 | 政策 + 存储 | `ready` 报 `policy_not_ready`（政策先于存储） | 503 |
| 启动 | 模型 + 存储 | `ready` 报 `model_not_ready`（模型最优先） | 503 |
| 请求 | `score_many` 抛 `TimeoutError`/未预期异常 | 评分、风险、decisions 写入 | 500 `internal_error` |
| 请求 | `evaluate_rules` 抛 `TimeoutError`/未预期异常 | 风险、decisions 写入（**纯模型评分照常 200**） | 500 `internal_error` |
| 请求 | 存储裸异常（insert/get/lookup/commit） | 对应的 decisions 接口 | 500 `internal_error` |
| 请求 | 存储**已分类**失败 | 对应的 decisions 接口 | 503 `storage_unavailable` |
| 请求 | 提交事务内真库回滚 | decisions 写入 | 503 `storage_unavailable` |

### 三条被重点验证的不变量

1. **隔离**：一个组件坏掉不得连带停用无关接口。政策坏掉时 `/v1/score` 与 `/v1/model`
   照常；模型坏掉时 `/v1/policy` 照常；存储坏掉时前八个接口全部照常。每个接口报的码
   必须指向**真正坏掉**的组件，不能张冠李戴。
2. **原子性**：批量中途失败不返回"看起来完整"的部分结果，且**零记录落库**。规则求值
   在模型调用之前，因此求值失败时模型**一次都没被调用**。带 `Idempotency-Key` 的提交
   失败会把 `decisions`、`idempotency_requests`、`idempotency_results` **同时**回滚为空。
3. **可恢复**：失败**不毒化**幂等键。故障解除后用同一个 key 重试正常创建（而不是误报
   409），随后重放返回首次快照、不重跑模型/政策、也不追加写入。

### 验证

```powershell
# CW-010 专项（93 个用例）
python -m pytest tests/unit/test_cw010_matrix.py `
  tests/integration/test_cw010_startup_faults.py `
  tests/integration/test_cw010_request_faults.py

# 默认一键门禁（真实运行完整 pytest）
python scripts/run_quality_gate.py --config configs/quality_gate.yaml

# 四套 smoke
python scripts/smoke_service.py --config configs/service.yaml
python scripts/smoke_decision_service.py --config configs/service.yaml
python scripts/smoke_persistence_service.py --config configs/service.yaml
python scripts/smoke_idempotency_service.py --config configs/service.yaml
```

### 已知限制

- 注入的 `TimeoutError` 只代表**依赖报告了超时**；本包**不**对阻塞调用施加真实时间上限，
  也**不**新增真实网络超时、自动重试、熔断器或后台恢复流程；
- 故障通过 fake loader / fake engine / 包装式 repository 在**调用边界**注入，
  因此验证的是"收到该异常后的行为"，不是特定驱动或文件系统错误的成因；
- 存储故障的原子性在**真实 SQLite** 上用真库回滚验证；模型与政策用测试替身，
  不涉及重训或重新评估；
- 未覆盖并发/多进程下的竞态（CW-008 只覆盖了单进程内的提交竞态）；
- 本包**不含** CI、容器、压力测试、认证、RAG/LLM，也不进入 `CW-011`。

## P3 / 本机性能基线与 P95 报告（CW-011）

CW-011 使用可选依赖 `performance` 中的 Locust，在 `127.0.0.1` 上启动短时服务，
以 CW-009 提交的**合成**特征分别请求单笔评分、风险融合和持久化决策。服务配置和 SQLite
数据库使用临时副本；默认库不参与负载。每位虚拟用户按固定 4:3:3 顺序循环三类请求。

```powershell
# 用户已授权安装本包所需依赖；只需安装一次
python -m pip install -e ".[performance,service]"

# 短时链路检查；只生成 smoke_report
python scripts/run_performance.py --smoke

# 两档正式测量；生成 report.json、report.md 和各档位事件流
python scripts/run_performance.py
```

正式档位分别为 1 用户 / 5 秒预热 / 25 秒测量、4 用户 / 8 秒预热 / 30 秒测量。
预热请求不进入统计；每档按接口核对最小有效样本数。报告从 Locust 请求事件计算
请求数、失败数/失败率、RPS 和最近秩 P50/P95/P99。失败和低样本均失败关闭，
保留安全分类报告。报告写入 `artifacts/reports/cw011_performance/`，事件流只有
接口名、测量窗口内时间、耗时和成功标志，不含特征值或请求体。

这些 P95 是本机、单进程、合成请求下的观测值。硬件、后台负载、Python/Locust 版本及
测量档位都会影响结果；它们不构成生产 SLO、容量承诺或真实交易负载结论。

### 每轮事件文件隔离与新鲜度

每档的统计**只能**来自本轮 Locust 写出的事件流，这一点由路径隔离保证，而不是靠比较
时间戳：本轮事件写入一个独占的临时文件（`.<name>_events.jsonl.<uuid>.round`），
因此上一轮留在正式文件名上的样本在物理上不可能被读到。进程退出后再做一次新鲜度校验，
要求文件存在、非空且修改时间不早于本轮开始（留 1 秒容差抵消文件系统时间戳精度）。

- 本轮没有写出新样本时**稳定失败**：退出码非 0 归类为 `locust_failed`，
  退出码为 0 则报 `no_new_events`；时间戳早于本轮报 `stale_events`。
  三种情况都**不会**回退去读任何既有事件文件。
- 只有完整成功的这一轮才用原子替换发布到正式文件名，因此 `low_events.jsonl` /
  `concurrent_events.jsonl` 始终与**最近一次成功测量**逐字段对应；失败的一轮只清理
  自己的临时文件，不破坏上一条可追溯的原始事件。
- 临时文件的清理是**单一出口**：spawn 失败、等待超时（`locust_timeout`）、新鲜度不足、
  统计失败（`invalid_measurement`）与请求失败（`request_failures`）都会删除本轮
  `.round`。清理失败也**不会**掩盖原始失败分类——宁可留下一个临时文件，也不把一次
  超时变成 `PermissionError`。

## P3 / 静态检查与本地总质量入口（CW-012）

CW-012 增加一条**从项目根执行**的总质量入口，把静态检查与已验收的 CW-009 默认门禁串成
一条命令，并审计"把这条命令搬到干净 CI"还缺什么。

### 一条命令

```powershell
python scripts/run_project_gate.py
```

按 **static → quality_gate** 顺序运行：

| 阶段 | 做什么 | 失败时退出码 |
| --- | --- | --- |
| `static` | 冻结规则集跑 ruff | 2=发现违规；3=工具失败；4=超时 |
| `quality_gate` | 原样执行 `python scripts/run_quality_gate.py --config configs/quality_gate.yaml`（完整 preflight → regression → pytest） | 5（其自身退出码记在报告的 `quality_gate.exit_code`） |
| 报告 | 原子写入 `artifacts/reports/cw012_project_gate/report.json` | 6=报告写入失败 |

全量 pytest **只跑一次**（由 CW-009 阶段跑）；总入口不重复启动独立 pytest，
也不改 CW-009 的配置、语料与退出码语义。任一阶段失败即失败关闭，
静态失败时质量门禁阶段**一次都不运行**。

### 静态规则集

只用**正确性**与**导入卫生**规则，不用风格规则；没有 `ignore`、没有 per-file ignores、
没有 `noqa` 兜底：

| 规则 | 抓什么 |
| --- | --- |
| `E9` | 语法错误（文件根本不能编译） |
| `F63` | 无效转义序列、对元组做断言、可疑比较 |
| `F7` | 语句位置错误（循环外 `break`/`continue` 等） |
| `F82` | 未定义名称（含 `__all__` 与局部变量先引用后赋值） |
| `F811` | 未使用名称被重复定义 |
| `F401` | 未使用的导入 |

范围：`src/ai_detector`、`scripts`、`performance`、`project_gate`、`tests`、`conftest.py`。
规则集在 `project_gate/static_checks.py::FROZEN_RULES` 里冻结并在命令行上显式传入
（`--isolated`，忽略工作区与用户级配置）；`pyproject.toml` 的 `[tool.ruff.lint] select`
声明同一份集合供手工运行，测试断言两者逐项相等。ruff 用 `==0.14.4` 精确锁版本
（可选 extra `static`，不影响服务运行时依赖）。

本包实测时该仓库有 24 处 `F401`（重构残留的未使用导入），已逐项做**最小修复**——
只删除未使用的导入名，未做任何重写或格式化；其余规则零违规。
`F841`/`F402` 刻意**不在**规则集内：它们在本仓库指向无测试覆盖的训练代码，
"修掉"一个看似多余的赋值反而可能改变被追踪的 holdout 访问计数，收益为负。

手工查看带文件位置的结果（总门禁报告只保留计数）：

```powershell
python -m ruff check --isolated --select E9,F63,F7,F82,F811,F401 `
  src/ai_detector scripts performance project_gate tests conftest.py
```

### 报告字段

`artifacts/reports/cw012_project_gate/report.json`（继续被 Git 忽略）只含：
阶段状态、**命令形状**（丢解释器绝对路径）、退出码、耗时、静态违规计数（按规则码）、
CW-009 的状态/回归/pytest 计数与四个输入的 `match` 标志、环境工具版本。
**不含**完整 stdout/stderr、测试节点 ID、请求/特征值、密钥、绝对路径、bundle 内容或
数据库行。

### CI 输入限制（未完成项）

**干净检出直接跑总门禁一定失败**：`artifacts/**` 被 `.gitignore` 排除，而 CW-009 预检
**必须**有已验收的 `artifacts/models/cw004/lightgbm_bundle.joblib`。实测两种失败形态
（在项目内被忽略的暂存目录里做，真实 bundle 与真实报告逐字节未动）：

| 场景 | 退出码 | `error_code` | `inputs.bundle` |
| --- | --- | --- | --- |
| bundle 不存在（干净检出） | 2 | `input_missing` | `match: false`，`actual_sha256: null` |
| bundle 存在但字节不符 | 2 | `input_hash_mismatch` | `match: false`，`actual_sha256` 为实际值 |

两种情况下回归与 pytest 都不运行，其余三个已提交输入仍 `match: true`——
失败精确指向被忽略的 bundle。**静态阶段不会伪造整体通过**：同一配置下静态如实
`passed`，整体仍 `failed`、退出码 5。

**CW-012 当时的结论**：P3 的 CI 项尚未完成；当时没有 `.git`、托管平台或可信的 bundle
供给方式，因此该包没有创建流水线文件或声称 hosted CI 已运行。CW-014 已完成后续托管落地。
完整清单、四种可选 bundle 供给路径及其代价见 [docs/ci_readiness.md](docs/ci_readiness.md)。

## P3 / 可信模型产物交付与隔离复现（CW-013）

CW-013 建立**本地、离线、可验证**的 bundle 搬运路径：从已验收 bundle 导出交付包，
在另一处按**门禁配置的冻结摘要**校验后导入，并在**不携带 `artifacts/**`** 的隔离源码
副本里证明"缺它必失败、导入后门禁能通过"。

### 三条命令

```powershell
# 1) 导出交付包（先校验冻结摘要，不符就不导出）
python scripts/export_bundle.py

# 2) 导入到门禁配置要求的目标路径（原子写入；同摘要目标幂等成功）
python scripts/import_bundle.py

# 3) 隔离复现
python scripts/verify_isolated_bundle_gate.py          # 快：解析路径 + 缺 bundle 必失败
python scripts/verify_isolated_bundle_gate.py --full   # 再加：导入后跑原样总门禁
```

交付包默认在 `artifacts/delivery/cw013_lightgbm_bundle/`（`lightgbm_bundle.joblib` +
`manifest.json`），报告在 `artifacts/reports/cw013_delivery/`；两者都被 `.gitignore`
的 `artifacts/**` 排除，**模型二进制不进版本控制、不上传**。

### 信任锚只有一个

导出的来源与导入的验收标准都取自 `configs/quality_gate.yaml` 的
`quality_gate.inputs.bundle`（路径 + SHA-256）。包内 `manifest.json` 只是自描述：
即便有人把载荷改坏、同时把清单改成与新字节自洽，导入仍然失败——有专门的区分性测试
守住这一点。目标已存在时：摘要相同 → 幂等成功；不同 → **拒绝覆盖**且原文件字节不变。

退出码（稳定）：`0` 成功；`2` 信任摘要不符；`3` 交付包不合格；`4` 路径不安全；
`5` 目标已存在且字节不同；`6` 写入或报告失败。

**目录边界（越界即失败关闭）**：`--package-dir` 只能指向被 Git 忽略的
`artifacts/delivery/` 子树（否则 `package_dir_not_ignored`），因为交付包里是模型二进制，
绝不能写进会被提交的目录；`--stage` 只能指向被 Git 忽略的 `artifacts/tmp_tests/` 之下的
专用子目录（否则 `stage_not_ignored`），且脚本只创建并清除本轮自己的 `run-<uuid>/`——
递归删除之前一定先校验边界。

**`already_present` 的含义**：导出时若目录里已有内容，它必须先是**完整、合法、与冻结
配置一致**的包；只有载荷在、清单缺失或损坏都会失败关闭（`package_incomplete` 等），
以免"导出成功"产出一个根本导入不了的包。

### 隔离为什么必须新建解释器

本项目以 editable 方式安装在 `.venv`，其 `.pth` 注册的是 **MetaPathFinder**，
优先级高于 `sys.path`——只设 `PYTHONPATH` 无法让代码从副本解析。隔离脚本新建独立 venv，
只放两个**普通路径** `.pth`（副本 `src` + 宿主 `site-packages`），因此宿主的 editable
finder 不会注册；脚本会断言 `ai_detector.__file__` 与 `PROJECT_ROOT` 都落在副本内。
**隔离的是代码解析，不是依赖**：第三方包复用宿主 venv 的同版本。

### 隔离实测结果

| 阶段 | 结果 |
| --- | --- |
| 缺 bundle | 总门禁 **exit 5**；静态如实 passed；`bundle` 输入 `false`；`input_missing` |
| 导入交付包 | **exit 0 / imported** |
| 导入后原样总门禁 | **exit 0**：静态 0 违规、回归 12/12、偏差 0.0、全量 pytest **1839 passed / 9 skipped / 0 failed** |
| 真实工作目录 | 冻结文件与默认库逐字节未变 |

9 个 skip 是隔离测试在副本内的**结构性跳过**（`python -m venv` 不能从 venv 内再建 venv，
嵌套验证既无意义也不可行），真实仓库里的默认运行不受影响。

### 新发现：全量套件还需要两个被忽略的产物

CW-012 的审计说"bundle 是唯一被忽略的**门禁**输入"——这对 **preflight** 成立；
隔离验证发现**全量 pytest** 还需要 `artifacts/models/cw003/rule_baseline_state.json`
与 `artifacts/runtime/cw007_decisions.sqlite3`。它们在真实工作目录里存在，干净检出里
没有，因此首次隔离运行实测 3 项失败（另一项是副本漏了已提交的 `.gitignore`，已修正）。
在 CW-013 阶段，这两个产物的可信供给方式**尚未确定**；CW-014 已用已审查夹具供给
CW-003 状态，并消除全量测试对默认数据库的依赖。

### CW-013 结束时的未完成项（现由 CW-014 处理）

CW-013 当时尚无 Git 仓库、托管平台、交付包的分发与保管决定或独立 CI 依赖环境；
该包只交付本地可信搬运与隔离复现。后续托管落地见下一节；当时的操作与限制见
[docs/bundle_delivery.md](docs/bundle_delivery.md)。

## P3 / GitHub Actions 托管质量门禁（CW-014）

CW-014 把已有的本地总门禁搬上 **GitHub Actions**：干净检出 → 独立安装受约束依赖 →
按冻结摘要校验三项被忽略输入 → 原样运行 `python scripts/run_project_gate.py`。
平台为 GitHub；工作流已在私有仓库 `cycy12335/A_Detector` 的 `main` 分支真实运行成功。
CW-014 已独立验收通过；托管门禁目前覆盖受信任的 `main` 推送和手动触发。

### 关键文件

| 文件 | 作用 |
| --- | --- |
| `.github/workflows/quality-gate.yml` | 托管工作流（runner `windows-latest`、Python 3.14） |
| `constraints/cw014-ci.txt` | 依赖约束，全部 `==` 精确锁定 |
| `configs/ci_inputs.yaml` | 三项被忽略输入的路径/摘要/供给方式 |
| `scripts/ci_fetch_inputs.py` | 逐字节校验并放置输入，缺失或不符即失败关闭 |
| `docs/ci_github.md` | 操作方式、权限边界、资产发布清单、未完成条件 |

### 权限与触发边界

`permissions` 只有 `contents: read`；只引用自动提供的 `secrets.GITHUB_TOKEN`；
触发只有 `workflow_dispatch` 与默认分支 `push`——**没有** `pull_request` /
`pull_request_target`，因此外部 PR 拿不到 token、读不到模型资产。
不受信任 PR 的完整门禁需要单独设计，本包没有实现它。

runner 固定 `windows-latest`：CW-009 golden master 用 **1e-12** 绝对容差比较模型分数，
LightGBM 预测不保证跨平台逐位一致，换平台必须先单独验证回归。

### 三项被忽略输入的供给

| 输入 | 供给方式 |
| --- | --- |
| CW-004 bundle（55,619 B，`f02bb50e…35a1c`） | **私有 Release 资产**，工作流下载后逐字节校验 |
| CW-003 规则基线状态（508 B） | **经审查入库的夹具** `tests/fixtures/cw014/rule_baseline_state.json` |
| CW-007 默认决策数据库 | **不再需要**（测试夹具消除依赖） |

第三项是 CW-014 关闭的干净检出缺口：全量套件过去依赖开发机上的默认库。现在
"默认库不被改写、也不被凭空创建"由根 `conftest.py` 的会话级守卫
`guard_default_database` 负责。落地时抓到了真正的副作用源——`test_cw005_cli.py` 与
`test_cw006_cli.py` 的探针配置从真实 `configs/service.yaml` 派生却**没有隔离
`storage.path`**，真实 CLI 进程会在干净检出里凭空建出默认库；两处已改为使用临时库。

### 验证

```powershell
# 本地等价预演（不需要 GitHub）
python scripts/ci_fetch_inputs.py
python scripts/run_project_gate.py

# GitHub：Actions → quality-gate → Run workflow；报告作为 artifact 上传
```

完整隔离复现（无 `artifacts/**` 的源码副本里跑完整门禁）见
[docs/bundle_delivery.md](docs/bundle_delivery.md)。

### 托管运行记录

私有 Release `cw014-ci-inputs-v1` 提供冻结 bundle；仓库变量 `CW014_RELEASE_TAG` 指向该标签。
GitHub Actions [成功运行 #36415890918](https://github.com/cycy12335/A_Detector/actions/runs/36415890918)
在提交 `34ba5cc0800907ba4998606ca3cdb014e6cee91f` 上完成：静态 0 违规、模型回归
12/12、全量 pytest **1910 passed / 0 skipped / 0 failed / 0 errors**。第一次托管运行发现
CW-013 隔离夹具仍假定默认库必然存在，现已修正。详细证据和限制见
[docs/ci_github.md](docs/ci_github.md) 第 6 节；CW-014 已独立验收通过。

## P4 / 可追溯规则知识库与离线检索（CW-015）

CW-015 建立一个只读、完全离线的规则证据入口：`configs/risk_policy.yaml` 是规则语义的唯一权威，
README 与操作文档只能解释限制，不能覆盖政策。来源清单会固定项目相对路径、SHA-256、版本和原文
行范围；构建时来源缺失、篡改、版本漂移或定位越界都会失败关闭。

```powershell
python scripts/build_retrieval_index.py
python scripts/retrieve_rules.py "严格大于阈值 等于不命中" --top-k 3
python scripts/evaluate_retrieval.py --top-k 5
```

检索只返回带 `citation_id`、来源摘要、版本和定位的已审查片段；无匹配时返回空结果，不编造引用。
评测集是合成且人工审查的，指标由脚本真实计算。检索分数不是模型分数、概率或风险等级；本包不接入
LLM、外部向量服务或真实交易数据，也不生成批准/拒绝/拦截建议。完整 schema、限制和重建方式见
[docs/retrieval.md](docs/retrieval.md)。

## P4 / 结构化调查草稿与证据校验（CW-016）

CW-016 在 CW-015 检索基线之上定义有限的结构化候选草稿：输入只接受既有 `DecisionRecord` 的派生
字段，观察只能引用同一政策版本和摘要的已审查 citation。候选必须经过独立事实/引用校验；规则不可
评估时只能标记 `insufficient_evidence`，不会把分数当概率，也不会给出批准、拒绝或拦截建议。

```powershell
python scripts/evaluate_investigation.py
python -m pytest tests/unit/test_cw016_investigation.py -q
```

评测使用人工审查的合成决策与恶意候选，不读取真实交易数据、默认数据库，不调用 LLM 或外部服务。
格式、事实一致性和证据完整率由脚本从实际校验结果计算。详细 schema、失败行为和限制见
[docs/investigation_draft.md](docs/investigation_draft.md)。

## P4 / 受控生成边界（CW-017）

CW-017 增加提供方中立的结构化请求/响应契约和受控编排器：请求只包含 CW-016
`DecisionSnapshot` 的显式白名单派生字段，以及 CW-015 已审查的政策片段；工具白名单为空，
请求/响应均有 schema 版本和大小上限。任何响应都作为不可信 `cw016_draft_v1` 候选，必须重新
经过 CW-016 `validate_candidate`；失败只会降级为 `degraded`、`rejected` 或
`insufficient_evidence`，不会生成可信处置结论。

```powershell
python scripts/evaluate_generation.py
python -m pytest tests/unit/test_cw017_generation.py -q
```

离线验证覆盖超时有限重试、调用/费用上限、凭据缺失、格式错误、额外字段、事实冲突和伪造引用。
负责人已决定使用 DeepSeek API 的 `deepseek-flash`（展示名 DeepSeek-V4.1-Flash），凭据只从
`DEEPSEEK_API_KEY` 读取；适配器通过 Responses API 的 JSON Schema 输出完成一次合成端到端调用，
结果经 CW-016 校验为 `verified`。本次预算上限按人民币 0.50 元保守换算为 USD 0.07，实际费用
USD 0.0011514；工具白名单为空，最多 2 次调用、最多 1 次重试、60 秒窗口最多 2 次调用。
配置、日志脱敏、真实调用与降级边界见 [docs/generation_adapter.md](docs/generation_adapter.md)。

## P5 / Docker Compose 本地演示（CW-018）

CW-018 提供本地只读容器演示：镜像不包含被忽略的模型 bundle，启动时通过只读 bind mount 提供，
并在监听端口前校验 bundle/policy 的冻结 SHA-256 和运行时契约。容器不启用 DeepSeek、默认 SQLite、
持久化卷或真实交易数据，使用非 root、只读根文件系统和 `127.0.0.1:8000` 宿主端口。

需要 Docker Desktop 和本地已授权的 CW-004 bundle。新克隆若 bundle 不存在，先按
[docs/docker_demo.md](docs/docker_demo.md) 的 CW-013 交付步骤导入，再运行：

```powershell
docker compose -f compose.demo.yaml up --build -d
python scripts/verify_demo.py
docker compose -f compose.demo.yaml down --remove-orphans
```

验证脚本使用合成常量特征检查健康、模型元数据、只读评分和错误特征契约。缺失或篡改 bundle 时入口
失败关闭；本机没有 Docker 时只能运行契约测试，不能把静态配置当作容器端到端证据。

## 产物位置

| 路径 | 内容 | 是否提交 Git |
| --- | --- | --- |
| `data/generated/transactions.csv` | 生成的模拟交易数据 | 否（`.gitignore` 排除） |
| `artifacts/reports/cw001_baseline_report.json` | 机器可读评估报告 | 否 |
| `artifacts/models/dummy_pipeline.joblib` | Dummy 完整 Pipeline | 否 |
| `artifacts/models/logistic_regression_pipeline.joblib` | 逻辑回归完整 Pipeline | 否 |
| `data/raw/ieee_cis/` | 使用者自行下载的 IEEE-CIS 原始数据 | 否 |
| `data/processed/ieee_cis/split_manifest_rows.csv` | 行级分区表 | 否 |
| `artifacts/reports/cw002_ieee_cis/` | 审计四类产物 | 否 |
| `artifacts/reports/cw003_ieee_cis_baselines/` | 真实数据基线的四类报告产物 | 否 |
| `artifacts/models/cw003/` | 两个 Pipeline 与规则基线冻结状态 | 否 |
| `artifacts/reports/cw004_lightgbm/` | LightGBM 挑战者的五类报告产物 | 否 |
| `artifacts/models/cw004/lightgbm_bundle.joblib` | 推理 bundle（预处理器 + 模型 + 冻结阈值） | 否 |
| `artifacts/reports/cw005_service/smoke_report.json` | 服务五接口的进程内 smoke 报告 | 否 |
| `artifacts/reports/cw006_decision/smoke_report.json` | 真实 bundle + 冻结政策下六步链路的 smoke 报告 | 否 |
| `artifacts/runtime/cw007_decisions.sqlite3` | 决策记录数据库（运行时产物，可再生成） | 否 |
| `artifacts/reports/cw007_persistence/smoke_report.json` | 持久化六步链路的 smoke 报告 | 否 |
| `artifacts/reports/cw008_idempotency/smoke_report.json` | v1→v2 迁移 + 幂等七步链路的 smoke 报告 | 否 |
| `artifacts/reports/cw009_quality_gate/report.json` | 一键质量门禁报告（预检/回归/pytest 三阶段） | 否 |
| `tests/fixtures/cw009/model_regression_cases.json` | CW-009 固定回归语料（12 个纯合成 case） | **是** |
| `configs/quality_gate.yaml` | CW-009 门禁配置（冻结四个输入指纹与命令形状） | **是** |

保存的是**完整 Pipeline**（预处理 + 模型），因此推理时会复用训练集拟合出的填充
统计量与缩放参数。删除 `artifacts/` 与 `data/generated/` 下的内容后，重新执行上面
两条命令即可完全重建。

## 目录结构

```text
ai_detector/
├── src/ai_detector/
│   ├── config.py            # 配置加载与路径解析
│   ├── training.py          # CW-001 训练编排
│   ├── ieee_training.py     # CW-003 真实数据训练编排（含 test 一次性评估入口）
│   ├── lightgbm_training.py # CW-004 前置门禁、两阶段确定性、受控 test 与报告
│   ├── service/             # CW-005 只读评分服务 + CW-006 政策/风险接口
│   │   ├── app.py           # 应用工厂、八个接口、请求 ID、安全错误、按组件降级
│   │   ├── config.py        # 服务配置、路径安全约束与政策段硬不变量
│   │   ├── schemas.py       # 严格 Pydantic 契约（含政策与风险响应）
│   │   ├── runtime.py       # 先验哈希后反序列化的运行时 + 先验哈希后解析的政策加载
│   │   ├── errors.py        # 统一错误结构（含 policy_not_ready）
│   │   └── logging.py       # 无载荷结构化日志（字段白名单）
│   ├── rules/               # CW-006 版本化透明规则引擎
│   │   ├── config.py        # 冻结政策：路径安全、先验哈希后解析、严格 schema
│   │   └── engine.py        # 纯规则求值与 0–4 点信号融合
│   ├── storage/             # CW-007 最小化决策记录持久化（标准库 sqlite3）
│   │   ├── config.py        # 冻结版本/PRAGMA 常量与安全的项目相对数据库路径
│   │   ├── records.py       # 不可变记录与严格编码/解码
│   │   ├── idempotency.py   # CW-008 请求指纹与幂等键摘要
│   │   └── sqlite.py        # schema 初始化、参数化读写、显式事务
│   ├── quality/             # CW-009 固定回归集与一键质量门禁
│   │   ├── config.py        # 严格配置 schema、规范相对路径与冻结指纹
│   │   ├── corpus.py        # 固定回归语料的严格 schema、特征顺序与边界配对契约
│   │   ├── regression.py    # 生产顺序求值、逐值比较、确定性与分布核对
│   │   ├── report.py        # 报告脱敏断言与 os.replace 原子写入
│   │   ├── errors.py        # 稳定错误码
│   │   └── gate.py          # 三阶段编排、pytest 子进程与冻结退出码
│   ├── data/
│   │   ├── schema.py        # CW-001 列契约
│   │   ├── simulate.py      # CW-001 确定性模拟数据生成
│   │   ├── fingerprint.py   # CW-002 流式 SHA-256 与路径上报策略
│   │   ├── ieee_cis.py      # CW-002 配置、发现、Schema 与连接审计
│   │   ├── temporal_split.py# CW-002 时间并列安全切分
│   │   ├── profiling.py     # CW-002 聚合列画像与特征角色登记
│   │   ├── audit.py         # CW-002 审计编排与四类产物
│   │   └── modeling.py      # CW-003 门禁、按需读取、左连接、复用 CW-002 清单
│   ├── features/
│   │   ├── split.py         # CW-001 时间切分
│   │   ├── pipeline.py      # CW-001 预处理管线
│   │   └── ieee_cis.py      # CW-003 特征契约与 train-only 决策
│   ├── models/
│   │   ├── baseline.py      # CW-001 两个基线
│   │   ├── threshold.py     # CW-001 阈值选择
│   │   ├── rule_baseline.py # CW-003 透明尾部异常规则
│   │   ├── ieee_baselines.py# CW-003 Dummy / Logistic 规格与收敛诊断
│   │   └── lightgbm_challenger.py # CW-004 冻结参数、早停、推理 bundle
│   └── evaluation/
│       ├── metrics.py       # CW-001 指标
│       ├── report.py        # 指标报告与环境版本
│       ├── cost.py          # CW-003 阈值曲线与相对成本
│       └── feature_importance.py # CW-004 逐维重要性聚合回源特征
├── tests/
│   ├── fixtures/ieee_cis/   # 项目自建合成 fixture（非 Kaggle 数据）
│   ├── fixtures/cw009/      # CW-009 固定回归语料（12 个纯合成 case，提交）
│   ├── cw003_fixtures.py    # CW-003 合成夹具与配置构造器
│   ├── cw004_fixtures.py    # CW-004 合成夹具（含 CW-003 已验收产物）
│   ├── cw005_fixtures.py    # CW-005 fake runtime 与合成 bundle
│   ├── cw006_fixtures.py    # CW-006 冻结政策映射与 fake policy engine
│   ├── cw007_fixtures.py    # CW-007 fake repository、可注入时钟与 ID factory
│   ├── cw008_fixtures.py    # CW-008 v1→v2 建库夹具与幂等常量
│   ├── cw010_fixtures.py    # CW-010 跨组件故障注入、故障矩阵与泄漏哨兵
│   ├── unit/
│   └── integration/
├── configs/
│   ├── baseline.yaml
│   ├── ieee_cis.yaml
│   ├── ieee_cis_baseline.yaml   # CW-003 建模配置（含 431 列显式允许清单）
│   ├── ieee_cis_lightgbm.yaml   # CW-004 冻结参数与前置指纹
│   ├── service.yaml             # CW-005/006/007 服务配置（bundle、政策与存储）
│   ├── quality_gate.yaml        # CW-009 门禁配置（冻结四个输入指纹与命令形状）
│   └── risk_policy.yaml         # CW-006 冻结政策（两条 train-only q0.99 规则）
├── data/                    # 三层数据目录，逐行数据一律不提交
├── artifacts/               # 可再生成产物（不提交）
├── scripts/
│   ├── generate_data.py
│   ├── train_baseline.py
│   ├── audit_ieee_cis.py
│   ├── train_ieee_baselines.py
│   ├── train_ieee_lightgbm.py
│   ├── run_api.py
│   ├── smoke_service.py
│   ├── smoke_decision_service.py   # CW-006 真实 bundle + 政策六步 smoke
│   ├── smoke_persistence_service.py # CW-007 真实 bundle + 政策 + 一次性库 smoke
│   ├── smoke_idempotency_service.py # CW-008 v1→v2 迁移 + 幂等七步 smoke
│   └── run_quality_gate.py          # CW-009 一键质量门禁入口
├── pyproject.toml
└── .gitignore
```

## 方法说明

### 为什么用时间切分而不是随机切分

交易数据具有强烈的时间相关性。随机切分会让同一时间窗口内的交易同时出现在训练集
和测试集，模型只要记住“这个时间段整体风险偏高”就能拿到虚高指标，掩盖真实的时间外
泛化差距。本工作包按时间先后切分 60% / 20% / 20%，并断言三个区间之间不发生时间穿越。

### 阈值怎么定的

阈值的候选集合是**固定网格** `0.05 ~ 0.95`（步长 0.05），不依赖任何数据分布。
选择规则：在**验证集**上取 F1 最大的阈值；并列时先取 Recall 更高者，再取阈值更高者。

**测试集只用于最终一次评估**，不参与训练、特征拟合、阈值选择或任何调参。报告中的
`test_metrics` 使用的阈值与 `models.*.threshold.threshold` 完全一致（有测试守护）。

### 关于 Dummy 基线的零指标

`DummyClassifier(strategy="prior")` 在验证集上输出常数概率 `c`（等于训练集欺诈率，
本次约 0.047）。常数预测器选出哪个阈值，取决于 `c` 与候选网格的相对位置。记
`A = {t ∈ 网格 : t <= c}`（不超过 `c` 的候选阈值集合）：

- `A` 为空（`c` 低于网格下界）：所有候选都只预测负例，F1 全为 0，完全并列，
  按 tie-break 选出**网格最高阈值**。
- `A` 非空且验证集含正例：`A` 中每个候选都把全部样本判为正例，Recall 全为 1、F1 也
  完全相同，这些候选彼此并列，按 tie-break 选出其中**最高**者，即 `max(A)`。结果一般
  **不是**网格下界；例如 `c = 0.12`、网格为 `0.05, 0.10, 0.15, ...` 时 `A = {0.05, 0.10}`，
  实际选中 `0.10`。
- `A` 非空但验证集无正例：F1 与 Recall 全为 0，同样完全并列，选出**网格最高阈值**。

本次数据属于第一种情形（`c` 约 0.047 < 网格下界 0.05）：选出阈值 0.95，
Precision/Recall/F1 退化为 0，ROC-AUC 为 0.5，PR-AUC 等于测试集基准欺诈率 0.0680。
这是基线应有的表现，不是流程错误。报告中的 `models.dummy.diagnostics` 字段会显式记录
该常数概率、判正率和说明，避免读者误判。

### LightGBM 的分数为什么不能叫“概率”

`CW-004` 的 LightGBM 使用了 `scale_pos_weight = 训练集负例数 / 训练集正例数`（本次约
`28.56`）。这是处理类别不平衡的常用手段，但它会**系统性抬高正类分数**：模型输出不再是
校准过的欺诈概率，只是同一套排序下的**模型分数**。因此本工作包：

- 报告与 bundle 一律把输出称为“模型分数（score）”，配置与文档中不出现“概率”断言；
- **不做**概率校准（Platt / Isotonic 等）——那属于后续阶段；
- 冻结阈值（F1 阈值与三个成本场景阈值）仍然只在**验证集**上按同一套固定网格与 tie-break
  选出，test 只应用这些冻结值一次。

### 特征重要性怎么读（以及不能怎么读）

`gain` / `split` 只描述“模型在训练数据上用哪些列降低了损失”，**不是**因果关系、不是业务
含义，也不是合规解释。348 个转换维度会聚合回 199 个源列：数值缺失指示维度并入对应数值列，
one-hot 维度并入对应类别列，聚合前后总量守恒。报告只给源列名与聚合数值，**不含任何
one-hot 类别取值**。本工作包**不做** SHAP、Permutation Importance、PDP，也**不为匿名
列（`V*`/`C*`/`D*`/`M*`/`id_*`）编造业务语义**。

## 已知限制

- 数据是模拟的，生成模型的系数由人为设定，不代表任何真实机构的数据分布。
- 三个特征（`amount`、`transaction_count_1h`、`is_new_device`）共同由隐变量 `risky`
  条件生成，因此在边际上并不独立——`risky` 会同时推高这三者。数据缺少真实用户行为
  序列、商户网络和长期时序依赖，只是一个简化的条件生成过程。
- 用户级隐风险会影响标签但不进入特征，用于保留不可学习的噪声；模型指标因此不会
  接近完美。这是刻意的设计，用来避免产生误导性的漂亮数字。
- 测试集与训练集的欺诈率不同（约 6.8% vs 4.7%），这只是本次有限样本下的分区比例
  波动。本模拟器是一个平稳生成过程，**没有**让标签概率或特征分布随时间变化，因此
  不能据此断言数据中存在真实的概念漂移或数据漂移。时间切分的作用是让评估以“未来
  时间外”的方式进行，而不是因为本数据里已经存在漂移。
- 当前只有 3 个特征、2 个基线模型，没有阈值成本曲线、没有模型解释、没有服务化。
- `CW-004` 的 LightGBM **只有一个固定参数配置**，没有超参数搜索，也没有多组参数对比；
  因此报告里的数字是“这一组参数的结果”，不是“调优后的最好结果”。参数在第一次真实训练前
  就已冻结，未依据任何 validation 或 test 指标修改。
- `CW-004` 的四模型对比是**跨来源汇总**：三基线指标读取自已验收的 `CW-003/report.json`，
  没有重新评估它们的 test。因此表内数值不是同一轮重训的产物，报告已显式标注来源指纹。
- `CW-004` 未做概率校准，模型输出只能当分数用；也未做 SHAP / Permutation / PDP，
  特征重要性不构成因果或业务解释。
- 匿名列是否包含未知代理变量或时间邻近泄漏风险**未解决**，只做了风险登记。
- `CW-009` 的固定回归集是 **golden master**：它只覆盖 12 个纯合成点，用于发现"实现或产物
  是否偏离已验收基线"，**不**度量模型效果、公平性、稳定性、现实漂移或业务正确性，也不代表
  输入空间。语料与配置的冻结指纹必须经显式评审后一起更新。
- 中文输出在某些终端下可能显示为乱码（控制台代码页非 UTF-8），可设置
  `$env:PYTHONIOENCODING="utf-8"` 规避。

## 免责声明

本项目是研究与工程原型，用于展示数据处理、建模流程和工程实践，**不宣称**达到银行
生产环境、监管合规或真实线上风控效果。任何指标都只在本仓库的模拟数据范围内成立。
