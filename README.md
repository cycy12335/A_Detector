# AI Detector

一个完整的、可复现的交易风险识别与 AI 辅助调查研究原型。

项目从模拟交易数据和 IEEE-CIS 公开匿名数据开始，提供数据审计、时间安全切分、基线模型、规则融合、
风险评分 API、决策结果持久化、离线检索，以及本地 Docker 演示。

## 项目效果

核心链路如下：

```text
交易数据
  → 数据契约与泄漏审计
  → 时间安全切分
  → train-only 特征处理与模型训练
  → 验证集冻结阈值
  → 模型评分与透明规则融合
  → JSON 报告、模型 bundle 与只读 API
```

当前可以演示：

- 固定随机种子生成模拟交易数据，并重复得到相同结果；
- 训练 Dummy、Logistic Regression 和 LightGBM 基线；
- 对交易请求执行模型评分、规则求值和研究型风险分级；
- 通过 FastAPI 提供只读评分、政策和决策查询接口；
- 使用 Docker Compose 完成健康检查、模型信息、政策、评分和错误输入验证；
- 对关键模型、政策、存储和检索行为运行固定回归与质量门禁。

> 项目中的分数和指标用于验证工程链路与数据处理方法，不代表真实银行生产风控效果，
> 也不是生产批准、拒绝或拦截依据。

## 环境要求

- Python 3.11 或更高；
- Windows PowerShell、macOS 或 Linux；
- 运行基础训练和测试只需要 Python 依赖；
- LightGBM 实验需要 tree extra；
- 本地 API 需要 service extra；
- Docker 演示需要 Docker Desktop 和 Docker Compose v2；
- Docker 演示还需要已按冻结 SHA-256 校验的 CW-004 模型 bundle。

所有命令默认从项目根目录执行。

## 快速运行

### 1. 安装基础依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

如果 PowerShell 禁止激活脚本，可以直接使用 .\.venv\Scripts\python.exe 替代下文的 python。

### 2. 生成模拟数据并训练基线

```powershell
python scripts/generate_data.py --seed 42 --rows 5000 --output data/generated/transactions.csv
python scripts/train_baseline.py --config configs/baseline.yaml
```

### 3. 运行测试

```powershell
python -m pytest
```

这条路径不需要真实交易数据，也不需要 Docker。

## 使用方式

### 运行本地只读评分服务

安装服务依赖：

```powershell
python -m pip install -e ".[test,tree,service]"
```

启动服务：

```powershell
python scripts/run_api.py --config configs/service.yaml
```

默认监听 http://127.0.0.1:8000，服务只监听本机，不具备认证、限流或 TLS 能力，不应直接暴露到公网。

### 运行 Docker 演示

先确保 Docker Desktop 已启动，并且项目中已有模型 bundle，然后执行：

```powershell
docker compose -f .\compose.demo.yaml up --build -d
python scripts/verify_demo.py
```

验证通过后停止服务：

```powershell
docker compose -f .\compose.demo.yaml down --remove-orphans
```

验证脚本会自动检查健康状态、模型、政策、评分接口和错误输入。

完整前置条件和失败路径见 [docs/docker_demo.md](docs/docker_demo.md)。

### 用外部 CSV 测试

将 CSV 文件复制到：

```text
data/incoming/external.csv
```

先测试前 1000 行：

```powershell
docker compose -f .\compose.demo.yaml up --build -d risk-demo
docker compose -f .\compose.demo.yaml --profile batch run --rm csv-runner `
  --input /data/incoming/external.csv `
  --output /data/output/external_scores_1000.csv `
  --base-url http://risk-demo:8000 `
  --id-column TransactionID `
  --label-column isFraud `
  --max-rows 1000
```

结果在：

```text
data/output/external_scores_1000.csv
data/output/external_scores_1000.csv.meta.json
```

确认 1000 行测试成功后，去掉 `--max-rows 1000`，即可处理完整 CSV。输入数据需要包含当前模型要求的 199 个特征；CSV 可以有额外列，批处理程序会自动忽略。`isFraud` 只用于本地评估，不会发送给模型。

### 运行质量门禁

```powershell
python scripts/run_project_gate.py
```

质量门禁覆盖静态检查、固定回归、服务 smoke 和项目级输入校验。具体输入要求以
[configs/quality_gate.yaml](configs/quality_gate.yaml) 为准。

## 示例输出

基础训练命令会输出数据规模、时间切分、验证集阈值和测试集指标，例如：

```text
输入数据：data/generated/transactions.csv
切分策略：timestamp_ascending_60_20_20
验证集选出的阈值与测试集指标：
  logistic_regression  阈值=0.50 P=... R=... F1=...
JSON 报告：artifacts/reports/cw001_baseline/report.json
模型产物（logistic_regression）：artifacts/models/cw001_baseline/...
```

具体数值会随着配置和依赖版本变化；指标只用于验证这次运行的可复现性，不应解读为生产效果。

训练和演示产物通常位于：

| 路径 | 内容 |
| --- | --- |
| data/generated/ | 模拟输入数据 |
| artifacts/reports/ | JSON、Markdown、CSV 报告 |
| artifacts/models/ | 模型和推理 bundle |
| artifacts/reports/cw018_docker_acceptance/ | Docker 演示验证报告 |

## 常见问题

### PowerShell 无法激活虚拟环境怎么办？

不必激活环境，直接使用：

```powershell
.\.venv\Scripts\python.exe -m pytest
```

### 没有 IEEE-CIS 数据能运行吗？

可以。快速运行使用项目生成的模拟数据，不需要下载外部数据。
IEEE-CIS 相关审计和训练流程需要用户自行从 Kaggle 下载并遵守数据集条款。

### 没有 LightGBM 能运行吗？

可以。基础数据生成、基线训练和测试不依赖 LightGBM。只有 CW-004 挑战者实验需要安装：

```powershell
python -m pip install -e ".[test,tree]"
```

### Docker 演示为什么提示 bundle 缺失？

Docker 演示不会自动下载、复制或替换模型 bundle。请先按照 [docs/docker_demo.md](docs/docker_demo.md) 的授权流程导入已验收 bundle。

### 这些分数是不是概率？

不是。模型输出是未校准的模型分数，风险等级是研究型规则结果，不能直接作为生产决策。

## 详细文档

- [实现过程：从零到一的工作包与验收记录](docs/implementation.md)
- [系统架构、模块说明与技术选型](docs/architecture.md)
- [开发记录、踩坑与后续计划](docs/development.md)
- [Docker 本地演示](docs/docker_demo.md)
- [外部 CSV 自动批处理](docs/external_csv_workflow.md)
- [3–5 分钟演示讲解脚本](docs/cw020_demo_script.md)
- [项目收尾、证据与复现入口](docs/project_closeout.md)
