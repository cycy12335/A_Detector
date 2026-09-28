"""CW-006 测试夹具：冻结政策映射、政策文件写入器与可注入的 policy engine 替身。

设计要点：

- **规范政策映射**（:data:`CANONICAL_POLICY`）与仓库里的 ``configs/risk_policy.yaml``
  保持等价，夹具据此在临时目录写政策文件，因此"先验哈希后解析"的成功路径不依赖仓库产物；
- **fake engine** 只实现 :class:`~ai_detector.rules.engine.RiskPolicyEngine` 协议
  （元数据 + ``evaluate_rules``），因此融合真值表可以在**不构造政策文件**的情况下逐例测试；
  它同时记录调用次数，"每 app 生命周期只加载一次政策"与"批量零次规则求值"靠它证明；
- 全部数据为项目自建，不访问网络，不读取真实 CSV、标签或 test 数据。
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ai_detector.config import PROJECT_ROOT
from ai_detector.rules.engine import RuleEvaluation, RuleEvaluationResult

#: 仓库里冻结政策的路径；``configs/service.yaml`` 按它固定指纹。
REPO_POLICY_PATH = PROJECT_ROOT / "configs" / "risk_policy.yaml"

#: 与 ``configs/risk_policy.yaml`` 等价的规范政策映射。
#: 任何一处偏差都应让"仓库政策与夹具政策一致"的测试失败——那份测试会把两份政策
#: 逐字段比较，避免夹具悄悄漂移成一个更宽松的政策。
CANONICAL_POLICY: dict[str, Any] = {
    "policy": {
        "schema_version": "cw006_policy_v1",
        "policy_version": "cw006-risk-policy-v1",
        "semantics": "research_signal_band",
        "threshold_source": "cw003_train_only_q0.99",
        "rules": [
            {
                "rule_id": "transaction_amt_train_q99_tail",
                "feature": "TransactionAmt",
                "operator": "gt",
                "threshold": 1092.0,
            },
            {
                "rule_id": "c13_train_q99_tail",
                "feature": "C13",
                "operator": "gt",
                "threshold": 580.0,
            },
        ],
        "fusion": {
            "flagged_signal_points": 2,
            "matched_rule_signal_points": 1,
            "rule_score_denominator": 2,
            "bands": {"low_signal": [0, 0], "elevated_signal": [1, 2], "high_signal": [3, 4]},
        },
        "disclaimer": (
            "夹具政策：公开匿名数据训练分区统计得到的透明演示信号，"
            "不是业务规则、不是欺诈概率，也不构成生产处置依据。"
        ),
    }
}


def policy_mapping(**patches: Any) -> dict[str, Any]:
    """取一份规范政策的深拷贝并施加补丁（补丁字段为 ``None`` 表示删除该键）。"""
    raw = copy.deepcopy(CANONICAL_POLICY)
    section = raw["policy"]
    for key, value in patches.items():
        if value is None:
            section.pop(key, None)
        else:
            section[key] = value
    return raw


def write_policy(path: Path, raw: dict[str, Any] | None = None) -> Path:
    """把政策映射写成 YAML 文件（默认写规范政策）。"""
    payload = CANONICAL_POLICY if raw is None else raw
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return path


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --- 测试用政策暂存目录 -------------------------------------------------
#
# 生产 policy 加载器**只接受项目相对路径**（这是硬安全契约，不能为测试开后门）。
# 因此测试也走同一条真实链路：把临时政策写进**项目内**、且已被 Git 忽略的暂存目录，
# 再用它的项目相对路径去调用真实加载函数。
#
# ``artifacts/**`` 已在 .gitignore 中被排除，所以这些临时文件不会被提交；
# 会话结束时由 conftest 里的 autouse 夹具统一清理。

#: 项目内、被 Git 忽略的测试暂存根目录。
TEST_STAGE_ROOT = PROJECT_ROOT / "artifacts" / "tmp_tests"


def stage_dir(tmp_path: Path) -> Path:
    """返回项目内的暂存目录（以 pytest ``tmp_path`` 名为后缀，保证逐测试唯一）。"""
    path = TEST_STAGE_ROOT / tmp_path.name
    path.mkdir(parents=True, exist_ok=True)
    return path


def stage_policy(
    tmp_path: Path, raw: dict[str, Any] | None = None, *, filename: str = "risk_policy.yaml"
) -> tuple[Path, str]:
    """写出政策文件，返回 ``(绝对路径, 项目相对 POSIX 路径)``。

    第二个返回值是**生产加载器唯一接受的形状**，测试据此走真实链路。
    """
    path = write_policy(stage_dir(tmp_path) / filename, raw)
    return path, path.relative_to(PROJECT_ROOT).as_posix()


def stage_relative(tmp_path: Path, *parts: str) -> str:
    """返回项目内暂存目录下某个（可能并不存在的）路径的项目相对形式。"""
    return (Path("artifacts/tmp_tests") / tmp_path.name).joinpath(*parts).as_posix()


# --- 可注入的 policy engine 替身 ----------------------------------------


class FakeRuleInfo:
    """与 :class:`ai_detector.rules.config.RuleInfo` 字段一致的轻量替身。"""

    __slots__ = ("rule_id", "feature", "operator", "threshold")

    def __init__(self, rule_id: str, feature: str, operator: str, threshold: float) -> None:
        self.rule_id = rule_id
        self.feature = feature
        self.operator = operator
        self.threshold = threshold


class FakePolicyEngine:
    """实现 ``RiskPolicyEngine`` 协议的测试替身。

    Args:
        results: 按调用顺序返回的规则求值结果；耗尽后重复最后一项（缺省则全部成"无命中且可评估"）。
        rules: 规则元数据；缺省为两条规范规则。
        error: 调用 ``evaluate_rules`` 时抛出的异常，用于验证错误折叠。
    """

    def __init__(
        self,
        *,
        results: Sequence[RuleEvaluationResult] | None = None,
        rules: Sequence[Any] | None = None,
        policy_version: str = "cw006-risk-policy-v1",
        policy_schema_version: str = "cw006_policy_v1",
        policy_sha256: str = "b" * 64,
        policy_semantics: str = "research_signal_band",
        threshold_source: str = "cw003_train_only_q0.99",
        disclaimer: str = "fake policy disclaimer",
        error: Exception | None = None,
    ) -> None:
        self.policy_version = policy_version
        self.policy_schema_version = policy_schema_version
        self.policy_sha256 = policy_sha256
        self.policy_semantics = policy_semantics
        self.threshold_source = threshold_source
        self.disclaimer = disclaimer
        self.rules = tuple(
            rules
            if rules is not None
            else (
                FakeRuleInfo("transaction_amt_train_q99_tail", "TransactionAmt", "gt", 1092.0),
                FakeRuleInfo("c13_train_q99_tail", "C13", "gt", 580.0),
            )
        )
        self._results = list(results) if results is not None else None
        self._error = error
        #: 每次 ``evaluate_rules`` 的入参快照，用于断言"零次规则求值"。
        self.evaluate_calls: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        return len(self.evaluate_calls)

    def evaluate_rules(self, features: Mapping[str, Any]) -> RuleEvaluationResult:
        self.evaluate_calls.append(dict(features))
        if self._error is not None:
            raise self._error
        if not self._results:
            return no_match_result(self.rules)
        if len(self.evaluate_calls) <= len(self._results):
            return self._results[len(self.evaluate_calls) - 1]
        return self._results[-1]


def no_match_result(rules: Sequence[Any] = ()) -> RuleEvaluationResult:
    """两条规则都可评估、都未命中的求值结果。"""
    ids = [rule.rule_id for rule in rules] or [
        "transaction_amt_train_q99_tail",
        "c13_train_q99_tail",
    ]
    return RuleEvaluationResult(
        evaluations=tuple(RuleEvaluation(rule_id=rule_id, matched=False, evaluable=True) for rule_id in ids)
    )


def matched_result(*matched: str, unevaluable: Sequence[str] = ()) -> RuleEvaluationResult:
    """按规则 ID 构造求值结果；未列出的规范规则记为可评估且未命中。"""
    all_ids = ["transaction_amt_train_q99_tail", "c13_train_q99_tail"]
    evaluations = tuple(
        RuleEvaluation(
            rule_id=rule_id,
            matched=rule_id in matched,
            evaluable=rule_id not in set(unevaluable),
        )
        for rule_id in all_ids
    )
    return RuleEvaluationResult(evaluations=evaluations)


ALL_RULE_IDS = ("transaction_amt_train_q99_tail", "c13_train_q99_tail")


class RecordingPolicyLoader:
    """记录调用次数的政策加载器；可选地抛出预设异常。"""

    def __init__(self, engine: Any = None, error: Exception | None = None) -> None:
        self.engine = engine if engine is not None else FakePolicyEngine()
        self.error = error
        self.calls = 0

    def __call__(self, config: Any) -> Any:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.engine


# --- 特征构造 -----------------------------------------------------------


def risk_features(
    runtime: Any,
    *,
    transaction_amt: Any = 0.0,
    c13: Any = 0.0,
) -> dict[str, Any]:
    """按运行时源特征清单构造一份合法 features，并把两个规则字段设成指定取值。

    其余列给常量 0.0：它们只用于满足 199 列契约，不参与任何规则判断。
    """
    payload = {name: 0.0 for name in runtime.source_features}
    payload["TransactionAmt"] = transaction_amt
    payload["C13"] = c13
    return payload
