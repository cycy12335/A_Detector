# CW-018 Docker Compose 本地演示

## 范围与前提

演示只启动已验收的只读评分/政策服务，不启用 DeepSeek、默认 SQLite、持久化卷、真实交易数据或任何工具。
合成请求只使用按 bundle schema 生成的常量特征；返回的分数是未校准模型分数，不是欺诈概率或生产处置。

需要 Docker Desktop（包含 Docker Compose v2）和本地已授权的 CW-004 bundle。bundle 不复制进镜像，
而是以只读 bind mount 提供；镜像构建不会偷偷下载或替换模型。

新克隆环境如果没有 `artifacts/models/cw004/lightgbm_bundle.joblib`，先从已授权的私有 Release 获取
CW-013 交付包并运行：

```powershell
python scripts/import_bundle.py
```

该步骤会按 `configs/quality_gate.yaml` 的冻结 SHA-256 校验后再写入被忽略的目标目录。不要把 bundle
提交到 Git，也不要把它放进 Dockerfile 的 `COPY` 范围。

## 三条命令

在项目根目录执行：

```powershell
docker compose -f compose.demo.yaml up --build -d
python scripts/verify_demo.py
docker compose -f compose.demo.yaml down --remove-orphans
```

验证脚本依次检查 `/health/live`、`/health/ready`、`/v1/model`、合成 `/v1/score` 和错误特征契约。
它只输出安全摘要，不保存请求特征值、响应载荷或数据库内容。演示启动失败时，入口会在监听端口前
校验 bundle/policy 的冻结摘要和运行时契约，并输出稳定的 `demo_preflight_failed:<code>`。

Compose 将容器端口映射到宿主机 `127.0.0.1:8000`，容器使用非 root 用户、只读根文件系统、只读
bundle 挂载、无新增 Linux capabilities 和 `no-new-privileges`。配置里不声明 storage，因此不会创建
默认 SQLite；容器销毁后没有业务数据需要清理。

## 失败路径与限制

- 缺失 bundle：容器入口退出并报告 `demo_preflight_failed:bundle_missing`。
- bundle 字节被篡改：入口退出并报告 `demo_preflight_failed:bundle_hash_mismatch`。
- policy 缺失、摘要不符或 schema 不符：入口退出并报告对应稳定政策错误码。
- 本地真实 Docker/Compose 启动、健康检查、请求和停机清理必须在装有 Docker 的环境复验；没有 Docker 的机器只能运行 CW-018 契约测试，不能宣称容器端到端通过。
- 托管质量门禁继续覆盖静态/回归/pytest；它不自动等价于 Docker 容器实测，除非工作流明确加入 Docker job。
