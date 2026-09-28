"""CW-009 固定回归语料：严格 schema + 契约校验。

语料是 **golden master**：它冻结"当前已验收 bundle + policy 在一组纯合成输入上的输出"，
只用来检测**实现或产物是否偏离已验收基线**。它不证明模型正确、公平、稳定或适合真实风控，
也不是在 test 集上重新评估效果——语料里没有任何真实交易 ID、标签或原始数据行。

## 严格到什么程度

- 顶层与每个 case 的键集合必须**完全相等**：多一个"备注"字段就拒绝；
- ``case_id`` 唯一，只含小写 ASCII 字母、数字、``_``、``-``，长度 3～64；
- 每个 case 的 ``features`` 键必须与 bundle 的源特征清单**名称与顺序都完全一致**，
  不多不少；数值列只接受有限 ``number``（显式拒绝 ``bool``），类别列只接受非空字符串；
- ``expected`` 只冻结七个字段，且必须逐字段合法（枚举、范围、有限数）。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ai_detector.quality.errors import QualityGateError, QualityGateErrorCode
from ai_detector.rules.config import ALL_RISK_LEVELS, MAX_SIGNAL_POINTS

#: 本语料的 schema 版本。未知版本即失败。
CORPUS_SCHEMA_VERSION = "cw009_model_regression_corpus_v1"

#: 固定语料类型标识。
CORPUS_KIND = "synthetic_golden_master_regression_corpus"

#: case 的角色。``boundary`` 必须是"只改一条规则特征"的阈值探针。
ROLE_BASELINE = "baseline"
ROLE_BOUNDARY = "boundary"
ROLE_STRESS = "stress"
ROLE_SHAPE = "shape"
CASE_ROLES: tuple[str, ...] = (ROLE_BASELINE, ROLE_BOUNDARY, ROLE_STRESS, ROLE_SHAPE)

#: 免责声明中必须出现的三处标注（大小写与连字符/下划线不敏感）。
REQUIRED_DISCLAIMER_MARKERS: tuple[str, ...] = (
    "synthetic",
    "golden master",
    "not business correctness",
)

_CASE_ID_PATTERN = re.compile(r"[a-z0-9_-]{3,64}")

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "disclaimer",
        "model_version",
        "policy_version",
        "bundle_sha256",
        "policy_sha256",
        "source_features",
        "score_absolute_tolerance",
        "cases",
    }
)
_CASE_KEYS = frozenset(
    {"case_id", "role", "notes", "baseline_case_id", "varied_features", "features", "expected"}
)
_EXPECTED_KEYS = frozenset(
    {
        "score",
        "model_flagged",
        "matched_rule_ids",
        "unevaluable_rule_ids",
        "rule_score",
        "signal_points",
        "risk_level",
    }
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _violation(detail: str) -> QualityGateError:
    return QualityGateError(QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION, detail)


def _exact_keys(mapping: Any, expected: frozenset[str], label: str) -> None:
    if not isinstance(mapping, dict):
        raise _violation(f"{label} 必须是映射")
    actual = {str(key) for key in mapping}
    if actual != expected:
        extra = sorted(actual - expected)
        missing = sorted(expected - actual)
        raise _violation(f"{label} 字段不匹配：多出 {extra}、缺少 {missing}")


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _violation(f"{label} 必须是数值")
    number = float(value)
    if not math.isfinite(number):
        raise _violation(f"{label} 必须是有限数")
    return number


def _string_list(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise _violation(f"{label} 必须是非空字符串列表")
    if len(set(value)) != len(value):
        raise _violation(f"{label} 不得包含重复项")
    return tuple(value)


@dataclass(frozen=True)
class RegressionExpectation:
    """一个 case 冻结的七个字段。**不存取值域以外的任何东西**。"""

    score: float
    model_flagged: bool
    matched_rule_ids: tuple[str, ...]
    unevaluable_rule_ids: tuple[str, ...]
    rule_score: float
    signal_points: int | None
    risk_level: str

    def as_mapping(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "model_flagged": self.model_flagged,
            "matched_rule_ids": list(self.matched_rule_ids),
            "unevaluable_rule_ids": list(self.unevaluable_rule_ids),
            "rule_score": self.rule_score,
            "signal_points": self.signal_points,
            "risk_level": self.risk_level,
        }


@dataclass(frozen=True)
class RegressionCase:
    """一个纯合成 case：199 个源特征 + 冻结的期望事实。"""

    case_id: str
    role: str
    notes: str
    baseline_case_id: str | None
    varied_features: tuple[str, ...]
    features: tuple[tuple[str, Any], ...]
    expected: RegressionExpectation

    def feature_mapping(self) -> dict[str, Any]:
        """按冻结顺序还原成 dict（Python dict 保序，与 bundle 顺序一致）。"""
        return dict(self.features)


@dataclass(frozen=True)
class RegressionCorpus:
    """已通过严格 schema 与契约校验的固定回归语料。"""

    schema_version: str
    kind: str
    disclaimer: str
    model_version: str
    policy_version: str
    bundle_sha256: str
    policy_sha256: str
    source_features: tuple[str, ...]
    score_absolute_tolerance: float
    cases: tuple[RegressionCase, ...]

    def case_ids(self) -> tuple[str, ...]:
        return tuple(case.case_id for case in self.cases)

    def case_by_id(self, case_id: str) -> RegressionCase:
        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise KeyError(case_id)


def _check_disclaimer(text: Any) -> str:
    if not isinstance(text, str) or not text.strip():
        raise _violation("disclaimer 必须是非空字符串")
    normalized = text.lower().replace("-", " ").replace("_", " ")
    normalized = " ".join(normalized.split())
    missing = [marker for marker in REQUIRED_DISCLAIMER_MARKERS if marker not in normalized]
    if missing:
        raise _violation(
            "disclaimer 必须明确标注 synthetic / golden-master / not-business-correctness"
        )
    return text


def _parse_features(
    raw: Any,
    label: str,
    *,
    source_features: Sequence[str],
    categorical_features: frozenset[str],
) -> tuple[tuple[str, Any], ...]:
    """校验单个 case 的 199 个源特征：名称、顺序与取值类型。"""
    if not isinstance(raw, dict):
        raise _violation(f"{label} 必须是映射")
    provided = list(raw.keys())
    if provided != list(source_features):
        if set(provided) != set(source_features):
            missing = sorted(set(source_features) - set(provided))
            extra = sorted(set(provided) - set(source_features))
            raise _violation(f"{label} 与源特征清单不一致：缺少 {len(missing)} 个、多出 {len(extra)} 个")
        raise _violation(f"{label} 的键顺序必须与 bundle 的源特征顺序完全一致")

    items: list[tuple[str, Any]] = []
    for name in source_features:
        value = raw[name]
        if name in categorical_features:
            if not isinstance(value, str) or not value:
                raise _violation(f"{label} 的类别特征必须是非空字符串")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise _violation(f"{label} 的数值特征必须是有限数值（不接受 bool/null/容器）")
            number = float(value)
            if not math.isfinite(number):
                raise _violation(f"{label} 的数值特征必须是有限数值")
        items.append((name, value))
    return tuple(items)


def _parse_expected(raw: Any, label: str) -> RegressionExpectation:
    _exact_keys(raw, _EXPECTED_KEYS, label)
    score = _finite_number(raw["score"], f"{label}.score")
    flagged = raw["model_flagged"]
    if not isinstance(flagged, bool):
        raise _violation(f"{label}.model_flagged 必须是布尔值")
    matched = _string_list(raw["matched_rule_ids"], f"{label}.matched_rule_ids")
    unevaluable = _string_list(
        raw["unevaluable_rule_ids"], f"{label}.unevaluable_rule_ids"
    )
    if set(matched) & set(unevaluable):
        raise _violation(f"{label} 的 matched 与 unevaluable 不得有交集")
    rule_score = _finite_number(raw["rule_score"], f"{label}.rule_score")
    if not 0.0 <= rule_score <= 1.0:
        raise _violation(f"{label}.rule_score 必须位于 [0, 1]")

    points = raw["signal_points"]
    if points is not None:
        if isinstance(points, bool) or not isinstance(points, int):
            raise _violation(f"{label}.signal_points 必须是整数或 null")
        if not 0 <= points <= MAX_SIGNAL_POINTS:
            raise _violation(f"{label}.signal_points 必须位于 [0, {MAX_SIGNAL_POINTS}]")

    risk_level = raw["risk_level"]
    if not isinstance(risk_level, str) or risk_level not in ALL_RISK_LEVELS:
        raise _violation(f"{label}.risk_level 必须是冻结等级之一")

    return RegressionExpectation(
        score=score,
        model_flagged=flagged,
        matched_rule_ids=matched,
        unevaluable_rule_ids=unevaluable,
        rule_score=rule_score,
        signal_points=points,
        risk_level=risk_level,
    )


def _parse_case(
    raw: Any,
    index: int,
    *,
    source_features: Sequence[str],
    categorical_features: frozenset[str],
) -> RegressionCase:
    label = f"cases[{index}]"
    _exact_keys(raw, _CASE_KEYS, label)

    case_id = raw["case_id"]
    if not isinstance(case_id, str) or not _CASE_ID_PATTERN.fullmatch(case_id):
        raise _violation(
            f"{label}.case_id 必须由小写字母、数字、'_'、'-' 组成且长度为 3～64"
        )

    role = raw["role"]
    if not isinstance(role, str) or role not in CASE_ROLES:
        raise _violation(f"{label}.role 必须是 {list(CASE_ROLES)} 之一")

    notes = raw["notes"]
    if not isinstance(notes, str) or not notes.strip():
        raise _violation(f"{label}.notes 必须是非空字符串")

    baseline_case_id = raw["baseline_case_id"]
    if baseline_case_id is not None:
        if not isinstance(baseline_case_id, str) or not _CASE_ID_PATTERN.fullmatch(baseline_case_id):
            raise _violation(f"{label}.baseline_case_id 必须是 null 或合法 case ID")
        if baseline_case_id == case_id:
            raise _violation(f"{label}.baseline_case_id 不得指向自身")

    varied = _string_list(raw["varied_features"], f"{label}.varied_features")
    unknown = sorted(set(varied) - set(source_features))
    if unknown:
        raise _violation(f"{label}.varied_features 含未知源特征")
    if role == ROLE_BASELINE:
        if varied:
            raise _violation(f"{label} 是基线 case，varied_features 必须为空")
        if baseline_case_id is not None:
            raise _violation(f"{label} 是基线 case，baseline_case_id 必须为 null")
    else:
        if not varied:
            raise _violation(f"{label}.varied_features 不得为空")
        if baseline_case_id is None:
            raise _violation(f"{label}.baseline_case_id 必须指向配对基线")

    features = _parse_features(
        raw["features"],
        f"{label}.features",
        source_features=source_features,
        categorical_features=categorical_features,
    )
    expected = _parse_expected(raw["expected"], f"{label}.expected")
    return RegressionCase(
        case_id=case_id,
        role=role,
        notes=notes.strip(),
        baseline_case_id=baseline_case_id,
        varied_features=varied,
        features=features,
        expected=expected,
    )


def _check_boundary_pairs(cases: Sequence[RegressionCase], source_features: Sequence[str]) -> None:
    """边界 case 必须**只**改动声明的那个规则特征，其余 198 项与配对基线逐值一致。

    这条约束让"低于/等于/略高于阈值"三个 case 之间只差一个特征，因此归因不含糊。

    Raises:
        QualityGateError: 配对缺失、改动集合与声明不符，或非声明特征发生变化。
    """
    by_id = {case.case_id: case for case in cases}
    for case in cases:
        if case.role != ROLE_BOUNDARY:
            continue
        baseline = by_id.get(case.baseline_case_id or "")
        if baseline is None:
            raise QualityGateError(
                QualityGateErrorCode.CORPUS_CONTRACT_VIOLATION,
                f"边界 case {case.case_id} 的配对基线不存在",
            )
        if len(case.varied_features) != 1:
            raise QualityGateError(
                QualityGateErrorCode.CORPUS_CONTRACT_VIOLATION,
                f"边界 case {case.case_id} 必须只声明一个目标特征",
            )
        base_map = baseline.feature_mapping()
        case_map = case.feature_mapping()
        changed = tuple(name for name in source_features if base_map[name] != case_map[name])
        if changed != case.varied_features:
            raise QualityGateError(
                QualityGateErrorCode.CORPUS_CONTRACT_VIOLATION,
                f"边界 case {case.case_id} 实际改动的特征与 varied_features 不一致",
            )


def parse_corpus(
    raw: Any,
    *,
    source_features: Sequence[str],
    categorical_features: Sequence[str],
) -> RegressionCorpus:
    """把已解析的 JSON 映射转成固定回归语料。

    Args:
        raw: 已由 :func:`json.loads` 解析的语料对象。
        source_features: bundle 记录的源特征清单（**顺序即契约**）。
        categorical_features: bundle 记录的类别列；其余一律按数值列校验。

    Raises:
        QualityGateError: schema 或契约违规。
    """
    _exact_keys(raw, _TOP_LEVEL_KEYS, "语料顶层")

    schema_version = raw["schema_version"]
    if not isinstance(schema_version, str) or schema_version != CORPUS_SCHEMA_VERSION:
        raise _violation(f"语料 schema_version 必须是 {CORPUS_SCHEMA_VERSION!r}")
    kind = raw["kind"]
    if not isinstance(kind, str) or kind != CORPUS_KIND:
        raise _violation(f"语料 kind 必须是 {CORPUS_KIND!r}")
    disclaimer = _check_disclaimer(raw["disclaimer"])

    for label, value in (("model_version", raw["model_version"]), ("policy_version", raw["policy_version"])):
        if not isinstance(value, str) or not value.strip():
            raise _violation(f"语料 {label} 必须是非空字符串")
    for label, value in (("bundle_sha256", raw["bundle_sha256"]), ("policy_sha256", raw["policy_sha256"])):
        if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
            raise _violation(f"语料 {label} 必须是 64 位小写十六进制 SHA-256")

    declared = raw["source_features"]
    if not isinstance(declared, list) or any(
        not isinstance(item, str) or not item for item in declared
    ):
        raise _violation("语料 source_features 必须是非空字符串列表")
    if len(set(declared)) != len(declared):
        raise _violation("语料 source_features 不得包含重复项")
    if list(declared) != list(source_features):
        raise QualityGateError(
            QualityGateErrorCode.CORPUS_CONTRACT_VIOLATION,
            "语料 source_features 必须与 bundle 的源特征清单名称与顺序完全一致",
        )

    tolerance = _finite_number(raw["score_absolute_tolerance"], "语料 score_absolute_tolerance")
    if tolerance < 0.0:
        raise _violation("语料 score_absolute_tolerance 不得为负")

    cases_raw = raw["cases"]
    if not isinstance(cases_raw, list) or not cases_raw:
        raise _violation("语料 cases 必须是非空列表")

    categories = frozenset(str(name) for name in categorical_features)
    cases = tuple(
        _parse_case(
            item,
            index,
            source_features=source_features,
            categorical_features=categories,
        )
        for index, item in enumerate(cases_raw)
    )
    ids = [case.case_id for case in cases]
    if len(set(ids)) != len(ids):
        raise _violation("语料 case_id 必须唯一")

    _check_boundary_pairs(cases, source_features)

    return RegressionCorpus(
        schema_version=schema_version,
        kind=kind,
        disclaimer=disclaimer,
        model_version=str(raw["model_version"]),
        policy_version=str(raw["policy_version"]),
        bundle_sha256=str(raw["bundle_sha256"]),
        policy_sha256=str(raw["policy_sha256"]),
        source_features=tuple(str(name) for name in declared),
        score_absolute_tolerance=tolerance,
        cases=cases,
    )


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """``json.loads`` 的 ``object_pairs_hook``：任意层级拒绝重复键。

    标准库默认让后者覆盖前者且**不报错**——相同值的重复键同样静默通过。那意味着评审
    看到的文本与解析器采用的对象可能不同，"字段集合精确"也就管不住原始文本里的重复。

    抛出的异常**不携带键名**：键名可能取自语料里的可疑内容，不进错误信息更安全。
    """
    mapping: dict[str, Any] = {}
    for key, value in pairs:
        if key in mapping:
            raise _DuplicateJsonKey
        mapping[key] = value
    return mapping


class _DuplicateJsonKey(Exception):
    """JSON 对象里出现重复键（无论取值相同还是冲突）。"""


def load_corpus(
    path: Path,
    *,
    source_features: Sequence[str],
    categorical_features: Sequence[str],
) -> RegressionCorpus:
    """从**已通过二进制指纹核对**的路径读取并解析语料。

    Raises:
        QualityGateError: 文件不可读、JSON 非法、**任意层级出现重复映射键**，
            或 schema/契约违规。
    """
    try:
        raw_bytes = path.read_bytes()
    except OSError as error:
        raise QualityGateError(
            QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION,
            f"语料不可读：{type(error).__name__}",
        ) from error
    try:
        raw = json.loads(
            raw_bytes.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys
        )
    except _DuplicateJsonKey:
        # 重复键是**语料内容违规**：映射到 schema 违规，而不是"JSON 语法错误"。
        raise QualityGateError(
            QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION, "语料 JSON 存在重复映射键"
        ) from None
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualityGateError(
            QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION,
            f"语料 JSON 解析失败：{type(error).__name__}",
        ) from error
    return parse_corpus(
        raw, source_features=source_features, categorical_features=categorical_features
    )


def forbidden_report_tokens(corpus: RegressionCorpus) -> frozenset[str]:
    """报告里**绝不允许**出现的字符串：源特征名与类别列的合成取值。

    只取类别取值（字符串），不取数值——数值与合法计数/偏差无法可靠区分，硬查会误报。
    """
    tokens: set[str] = set(corpus.source_features)
    for case in corpus.cases:
        for _name, value in case.features:
            if isinstance(value, str):
                tokens.add(value)
    return frozenset(tokens)
