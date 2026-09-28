"""CW-009 固定模型回归集与一键质量门禁。

这个包只做一件事：用一组**已提交、纯合成、版本化**的固定输入，冻结当前已验收
bundle + policy 的模型分数与融合决策，然后在一个命令里依次完成产物预检、回归比较和
完整 pytest，任何偏差都失败关闭并写出一份安全的 JSON 报告。

它**不**训练、不调参、不解释模型，也不重新评估任何真实数据上的指标。固定回归集是
golden master：它回答"实现或产物是否偏离了已验收基线"，不回答"模型准不准"。

分层：

- :mod:`ai_detector.quality.config` —— 严格配置 schema、规范相对路径、冻结指纹；
- :mod:`ai_detector.quality.corpus` —— 固定回归语料的严格 schema 与契约校验；
- :mod:`ai_detector.quality.regression` —— 按生产顺序求值、逐值比较与确定性检查；
- :mod:`ai_detector.quality.report` —— 报告脱敏断言与原子写入;
- :mod:`ai_detector.quality.gate` —— 一键流程、pytest 子进程与冻结退出码。
"""

from __future__ import annotations

from ai_detector.quality.config import (
    CONFIG_SCHEMA_VERSION,
    DEFAULT_CONFIG_PATH,
    DEFAULT_REPORT_PATH,
    WORK_PACKAGE,
    DuplicateMappingKeyError,
    InputDigest,
    QualityGateConfig,
    collect_frozen_digest,
    collect_frozen_digests,
    first_problem,
    strict_yaml_load,
)
from ai_detector.quality.corpus import (
    CORPUS_SCHEMA_VERSION,
    RegressionCase,
    RegressionCorpus,
    RegressionExpectation,
    load_corpus,
    parse_corpus,
)
from ai_detector.quality.errors import QualityGateError, QualityGateErrorCode
from ai_detector.quality.gate import (
    EXIT_OK,
    EXIT_PREFLIGHT,
    EXIT_PYTEST,
    EXIT_PYTEST_TIMEOUT,
    EXIT_REGRESSION,
    EXIT_REPORT,
    GateOutcome,
    PytestOutcome,
    main,
    run_gate,
    run_pytest,
)
from ai_detector.quality.regression import (
    CaseActual,
    RegressionOutcome,
    run_regression,
    score_case,
)
from ai_detector.quality.report import REPORT_SCHEMA_VERSION, write_report

__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "CORPUS_SCHEMA_VERSION",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_REPORT_PATH",
    "EXIT_OK",
    "EXIT_PREFLIGHT",
    "EXIT_PYTEST",
    "EXIT_PYTEST_TIMEOUT",
    "EXIT_REGRESSION",
    "EXIT_REPORT",
    "CaseActual",
    "DuplicateMappingKeyError",
    "GateOutcome",
    "InputDigest",
    "PytestOutcome",
    "QualityGateConfig",
    "QualityGateError",
    "QualityGateErrorCode",
    "REPORT_SCHEMA_VERSION",
    "RegressionCase",
    "RegressionCorpus",
    "RegressionExpectation",
    "RegressionOutcome",
    "WORK_PACKAGE",
    "collect_frozen_digest",
    "collect_frozen_digests",
    "first_problem",
    "load_corpus",
    "main",
    "parse_corpus",
    "run_gate",
    "run_pytest",
    "run_regression",
    "score_case",
    "strict_yaml_load",
    "write_report",
]
