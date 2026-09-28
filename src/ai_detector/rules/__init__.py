"""CW-006 版本化透明规则引擎。

本包把两条**公开、透明、确定**的尾部规则与模型阈值信号组合成研究型风险等级。
它没有 I/O、没有随机性、没有全局状态，也不访问标签或任何训练/test 数据。

模块划分：

- :mod:`ai_detector.rules.config` —— 政策配置：路径安全、先验哈希后解析、严格 schema；
- :mod:`ai_detector.rules.engine` —— 纯规则求值与 0–4 点信号融合。

**它不是什么**：不是业务规则库，不是动态规则平台，也不产生任何生产处置。
"""

from __future__ import annotations

from ai_detector.rules.config import (
    FROZEN_BANDS,
    FROZEN_RULES,
    POLICY_SCHEMA_VERSION,
    POLICY_SEMANTICS,
    POLICY_VERSION,
    THRESHOLD_SOURCE,
    PolicyErrorCode,
    PolicyLoadError,
    RiskPolicy,
    RuleSpec,
    load_policy,
)
from ai_detector.rules.engine import (
    PolicyContractError,
    PolicyEngine,
    RiskAssessment,
    RiskPolicyEngine,
    RuleEvaluation,
    RuleEvaluationResult,
    fuse,
    validate_rule_result,
)

__all__ = [
    "FROZEN_BANDS",
    "FROZEN_RULES",
    "POLICY_SCHEMA_VERSION",
    "POLICY_SEMANTICS",
    "POLICY_VERSION",
    "THRESHOLD_SOURCE",
    "PolicyContractError",
    "PolicyEngine",
    "PolicyErrorCode",
    "PolicyLoadError",
    "RiskAssessment",
    "RiskPolicy",
    "RiskPolicyEngine",
    "RuleEvaluation",
    "RuleEvaluationResult",
    "RuleSpec",
    "fuse",
    "load_policy",
    "validate_rule_result",
]
