# 项目收尾与交付索引

2026-10-03，CW-023 最终独立复验 `passed`，P5 达到 [`PROJECT_PLAN.md`](../PROJECT_PLAN.md) 的退出条件。验收对象为提交 `51611e97e18933d594c253533b075185861f65bf`；对应 [GitHub Actions #37103879967](https://github.com/cycy12335/A_Detector/actions/runs/37103879967) 成功。本页记录已验收范围和继续维护时的入口；后续文档维护提交不能替代该历史验收证据。

## 从哪里开始

| 目的 | 入口 | 已验收范围 |
| --- | --- | --- |
| 本地合成演示 | [README](../README.md) → [Docker 说明](docker_demo.md) | CW-018/CW-023：Compose 构建、健康、评分与错误输入 6/6，结束后清理容器 |
| 理解架构 | [架构与取舍](architecture.md) | CW-019：离线数据/模型、只读服务、调查候选和信任边界 |
| 面试讲解 | [3–5 分钟脚本](cw020_demo_script.md) | CW-020：分段脚本；PPTX/PDF 为可选附件，视频不要求 |
| 阅读对比 | [三组对比说明](cw021_comparisons.md) | CW-021：历史四模型、validation 阈值成本、固定合成检索 `top-k` |
| 查看故障边界 | [Bad Case 案例](cw022_bad_cases.md) | CW-022：四个合成案例与两个契约负例 |
| 使用简历表述 | [中英文条目与 claim 表](cw023_resume_claims.md) | CW-023：7 条 claim、19 项来源及文本/口径负向校验 |

## 验收证据与限制

- CW-023 最终独立复验核对 7/7 claim、19 项来源 SHA、错误文本与来源漂移拒绝。历史 LightGBM F1 只对应 IEEE-CIS 匿名研究数据的单次 historical test 分区；指标复核读取冻结报告，不重新训练。
- README 演示已按 `docker compose -f compose.demo.yaml up --build -d` → `python scripts/verify_demo.py` → `docker compose -f compose.demo.yaml down --remove-orphans` 实测，6/6 HTTP 检查通过，使用合成常量特征，`storage=false`、`llm=false`，随后无运行容器。新环境须先准备 Docker Desktop Linux 引擎、授权 CW-004 bundle 和可用本地端口；未对无资产的新克隆环境或生产流量作实测。
- 最终验收提交的托管报告为静态违规 0、固定回归 12/12、pytest 1954 passed / 0 skipped；输入 bundle `verified`、规则夹具 `placed`、默认库 `not_required`。本地完整门禁为静态 0、回归 12/12、pytest 1951 passed / 3 skipped。
- 本项目是研究与工程原型。模型分数不等于校准欺诈概率，合成回归不证明业务正确性，跨来源模型对比不证明因果收益，合成 Bad Case 不代表生产事故或真实误报/漏报。

## 后续维护

README 是新用户的演示入口；`configs/retrieval_sources.json` 保存 CW-015 审查片段的整文件摘要。改 README 后须核对受审查 298–302 行，更新摘要并运行检索专项。CW-021、CW-022 与 CW-023 的输入清单还冻结了各自实施时的来源摘要；改动这些来源后，应重新生成对应安全聚合并逐级更新 claim 校验来源。历史验收结论始终以当时提交和报告为准。

本机协作记录在 `.handoff/latest.md` 和对应任务六文件；这些记录因含本机信息而不随 Git 发布。公开交付的范围、命令和限制以上述 README 与文档为准。新功能与生产化需求需另立工作包评估。
