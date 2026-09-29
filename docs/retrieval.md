# CW-015 可追溯规则知识库与离线检索

CW-015 建立的是“证据入口”，不是调查报告生成器。索引只包含 `configs/retrieval_sources.json`
白名单中的人工审查文本行范围：`configs/risk_policy.yaml` 是规则语义的唯一权威，README 和
CI 文档只能解释限制，不能覆盖政策。

每个片段都带有稳定 `citation_id`、项目相对来源路径、来源 SHA-256、来源版本、权限级别、行范围和
限制标签。构建索引时会先校验路径、来源摘要和 UTF-8，再读取精确行范围；任一来源缺失、篡改、版本
漂移或定位越界都会失败关闭。索引过程不读取原始交易行、SQLite、模型二进制或网络资源。

## 离线命令

```powershell
python scripts/build_retrieval_index.py
python scripts/retrieve_rules.py "严格大于阈值 等于不命中" --top-k 3
python scripts/evaluate_retrieval.py --top-k 5 --output artifacts/reports/cw015_retrieval.json
```

检索分数是词项重合排序分数，不是模型分数、概率或风险等级。结果按分数降序、`citation_id` 升序
确定性排序；`top_k` 最大为 10，`max_output_chars` 限制证据 JSON 数组的字符数；没有匹配时返回空证据，不根据查询编造引用。

评测查询集 `tests/fixtures/cw015/retrieval_queries.json` 是合成且人工审查的，覆盖两条规则、train-only
阈值来源、严格大于边界、C13 匿名限制、未校准分数、生产处置限制、CI 边界和无依据查询。评测脚本
实际运行检索并计算 Recall@1、Recall@k、MRR 与负例空结果率，指标不是手填的。Recall 为每条正例查询命中相关引用数占该查询全部相关引用数的比例，再对正例查询取平均；MRR 也只对正例查询取平均。无依据负例单独计入空结果率。

## 不支持的结论

本知识库不能证明某笔交易欺诈，不能把 `C13` 解释成业务字段，不能把未校准模型分数叫作欺诈概率，
也不能给出批准、拒绝或拦截建议。后续工作包如需自然语言调查报告，必须继续携带本包的引用和限制，
不得把解释性来源提升为政策权威。
