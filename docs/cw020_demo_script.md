# CW-020 3–5 分钟讲解脚本

## 使用口径

- **现场步骤**：演示者可在项目根目录、Docker Desktop 可用且已授权 bundle 存在的环境执行。
- **历史证据**：本包不重新运行，只引用 CW-018/CW-019 已验收材料；不能说成“本次运行结果”。
- **源码边界**：来自当前源码和文档的设计/失败关闭规则；不等同于生产效果。
- **安全边界**：全程只使用匿名或合成输入，不展示真实交易、原始特征、个人信息、私有地址、凭据或 API key。

## 0:00–0:25｜项目定位

这是一个面向作品集的 AI Detector 研究原型。演示默认只使用匿名或合成输入，重点是评分边界、冻结资产和失败关闭，不把研究型分数包装成生产风控效果。

**来源与口径**：`docs/architecture.md` 第 1 节“交付边界”；历史证据为 CW-018/CW-019 验收材料。现场操作：无。图示口径：架构图属于 CW-019 已验收材料，本包不声称重新生成。

## 0:25–0:55｜系统边界

评分服务只读加载冻结的 LightGBM bundle 和版本化政策。调查生成是可选路径，默认服务、默认测试和 CI 都不调用 DeepSeek。进入外部模型的内容限于决策快照和已审查上下文。

**来源与口径**：`docs/architecture.md` 第 1、4、7、8.6 节，当前源码 `src/ai_detector/investigation/generation.py`；这是当前源码边界，不是现场外部模型调用。现场操作：无。

## 0:55–1:30｜离线模型

离线链路先做匿名数据审计和时间切分，再用 train-only 规则拟合预处理。LightGBM 用 validation 早停，随后用 validation 分数冻结 F1 阈值，test 只评估一次，最后交付 bundle。服务端不读取原始 CSV。

**来源与口径**：`docs/architecture.md` 第 3 节、`src/ai_detector/lightgbm_training.py`；“validation 早停、F1 阈值、test 一次评估”是模型训练规则，“服务只读 bundle”是运行时边界。历史图示为 CW-019 已验收架构 PNG，现场操作：无。

## 1:30–2:05｜请求时评分与调查

请求先经过 199 列严格契约和 bundle 摘要校验，再得到未校准模型分数和政策结果。只有显式调查路径才会构造 DeepSeek 请求；外部候选返回后还要经过 CW-016 独立事实和引用校验。候选不能更新决策。

**来源与口径**：`docs/architecture.md` 第 4、7、8.6 节，`src/ai_detector/investigation/generation.py`；199 列是当前请求契约，SHA-256 是运行时资产校验，CW-016 是候选校验边界。历史图示为 CW-019 已验收架构 PNG，现场操作：无，不重放付费调用。

## 2:05–2:45｜现场 Docker 演示

现场前提是 Docker Desktop、Docker Compose v2、Docker 引擎可用，并且本地已有按冻结 SHA-256 校验过的 CW-004 bundle。若 bundle 不存在，先按 `docs/docker_demo.md` 的授权流程运行 `python scripts/import_bundle.py`；不要把 bundle 提交 Git 或复制进镜像。

从项目根目录执行三条命令：

```powershell
docker compose -f compose.demo.yaml up --build -d
python scripts/verify_demo.py
docker compose -f compose.demo.yaml down --remove-orphans
```

**现场预期**：`verify_demo.py` 依次检查 live、ready、model、policy、合成 score 和错误特征契约；错误输入返回稳定 422。**历史证据**：CW-018 `artifacts/reports/cw018_docker_acceptance/report.json` 记录六项检查通过、199 列合成常量特征、无持久化和未调用 LLM；本包没有新增容器实测，也不把 CI 绿色当 Docker 实测。

## 2:45–3:15｜失败关闭

缺列、类型或批量不合法时返回 422，不执行部分评分。bundle 或 policy 摘要不匹配时，服务不就绪或 preflight 失败，不静默替换模型。候选证据不足时返回 rejected 或 insufficient_evidence。

**来源与口径**：`docs/architecture.md` 第 5、7 节、`docs/docker_demo.md`“失败路径与限制”、CW-018 报告中的 `invalid_features=422`。这些是源码和历史验收定义的失败边界；现场可按文档复现，但本包不新增错误实测。

## 3:15–3:45｜证据与限制

CW-018 提供 Docker 合成演示证据，CW-019 提供架构图、数据流和关键取舍，Actions 提供冻结门禁和安全报告。输入是研究数据，模型分数未校准，不是欺诈概率，研究型等级不是生产处置，默认不调用 DeepSeek。

**来源与口径**：CW-018/CW-019 `acceptance.md`、`docs/ci_github.md`、`docs/generation_adapter.md`；“Actions 通过”只说明对应托管门禁，不替代 Docker 容器实测。现场操作：无。

## 3:45–4:10｜复现入口

最后指向 README、docs/docker_demo.md 和 docs/architecture.md。总结：评分链路可复现，调查链路可审查，失败边界可解释。全程不展示真实交易、原始特征、个人信息、私有地址或凭据。

**来源与口径**：`README.md` 的 CW-018/CW-019/CW-020 入口、`docs/docker_demo.md` 的三条命令、`docs/architecture.md` 第 9 节复现命令。现场操作：结束；脚本独立可读，不依赖视频或未制作材料。
