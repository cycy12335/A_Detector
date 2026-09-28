"""CW-006 政策配置：路径安全 + **先验哈希后解析** + 严格 schema。

## 顺序本身就是一条安全边界

与 CW-005 的 bundle 加载同源：``yaml.safe_load`` 会被喂进一整个文件，而政策文件决定的是
"什么算风险信号"这一判断本身。因此顺序写死成：

1. 文件存在性检查；
2. **流式** SHA-256 校验（分块读取，不把内容解析成对象）；
3. 只有摘要完全等于服务配置里那个固定值，才调用 ``yaml.safe_load``；
4. 解析后逐字段对照代码里的硬不变量。

哈希不符时 ``yaml.safe_load`` **一次都不会被调用**（有测试用 monkeypatch 守住）。

## 硬不变量写在代码里，配置文件不得放宽

阈值、规则 ID/顺序、严格 ``gt`` 运算符、融合真值表、版本字符串全部是模块常量。
政策文件必须**逐字复述**它们，任何偏差都是加载失败——因此"改一行 YAML 把 ``>`` 换成 ``>=``"
或"顺手把阈值调成 100"不会静默生效，只会让服务安全降级为政策未就绪。

## 阈值从哪来

``TransactionAmt > 1092.0`` 与 ``C13 > 580.0`` 来自 CW-003 透明规则基线状态
（``artifacts/models/cw003/rule_baseline_state.json``）在 **train 分区**上算出的 0.99 分位。
本模块**不重新计算分位数**、不读取原始 CSV、不读取标签，也不在 validation/test 上挑参数。
``C13`` 是匿名特征，本项目不为它赋予任何业务含义。
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

from ai_detector.config import resolve_path

#: 流式校验的分块大小。
HASH_CHUNK_SIZE = 1 << 20

#: 政策 schema 版本（写在政策文件里，也写在服务配置的 policy 段里，两处必须一致）。
POLICY_SCHEMA_VERSION = "cw006_policy_v1"

#: 政策版本（人类可读标识）。
POLICY_VERSION = "cw006-risk-policy-v1"

#: 固定语义：研究型信号分级，不是概率，也不是处置。
POLICY_SEMANTICS = "research_signal_band"

#: 阈值来源标识（对外可见，说明它来自哪里）。
THRESHOLD_SOURCE = "cw003_train_only_q0.99"

#: 模型 ``flagged=True`` 贡献的信号点。
FLAGGED_SIGNAL_POINTS = 2

#: 每条命中规则贡献的信号点。
MATCHED_RULE_SIGNAL_POINTS = 1

#: ``rule_score = 命中规则数 / RULE_SCORE_DENOMINATOR``。
RULE_SCORE_DENOMINATOR = 2

#: 唯一允许的比较运算符：**严格大于**。恰好等于阈值不命中。
STRICT_OPERATOR = "gt"

#: 风险等级名。前三个只在规则全部可评估时出现，``indeterminate`` 表示规则输入不可评估。
RISK_LEVEL_LOW = "low_signal"
RISK_LEVEL_ELEVATED = "elevated_signal"
RISK_LEVEL_HIGH = "high_signal"
RISK_LEVEL_INDETERMINATE = "indeterminate"

#: 可评估时按信号点取值的有序等级。
RISK_LEVELS: tuple[str, ...] = (RISK_LEVEL_LOW, RISK_LEVEL_ELEVATED, RISK_LEVEL_HIGH)

#: 对外公布的完整等级集合（含 ``indeterminate``）。
ALL_RISK_LEVELS: tuple[str, ...] = (*RISK_LEVELS, RISK_LEVEL_INDETERMINATE)

#: 默认政策路径（相对项目根目录）。
DEFAULT_POLICY_PATH = "configs/risk_policy.yaml"

#: 路径中不得出现的疑似凭据片段。
_CREDENTIAL_MARKERS = ("password", "token", "cookie", "api_key", "apikey", "secret")


@dataclass(frozen=True)
class RuleSpec:
    """一条冻结规则：读哪个源特征、用什么运算符、比哪个阈值。"""

    rule_id: str
    feature: str
    operator: str
    threshold: float


@dataclass(frozen=True)
class BandSpec:
    """信号点区间到风险等级的映射（闭区间）。"""

    name: str
    low: int
    high: int


#: 两条**冻结**规则。顺序即对外声明的顺序，也是求值结果的顺序。
#: 阈值为 CW-003 train-only 0.99 分位；不是业务规则，也不为匿名 C13 编造含义。
FROZEN_RULES: tuple[RuleSpec, ...] = (
    RuleSpec(
        rule_id="transaction_amt_train_q99_tail",
        feature="TransactionAmt",
        operator=STRICT_OPERATOR,
        threshold=1092.0,
    ),
    RuleSpec(
        rule_id="c13_train_q99_tail",
        feature="C13",
        operator=STRICT_OPERATOR,
        threshold=580.0,
    ),
)

#: **冻结**融合真值表：0=low、1-2=elevated、3-4=high。
FROZEN_BANDS: tuple[BandSpec, ...] = (
    BandSpec(RISK_LEVEL_LOW, 0, 0),
    BandSpec(RISK_LEVEL_ELEVATED, 1, 2),
    BandSpec(RISK_LEVEL_HIGH, 3, 4),
)

#: 信号点上限：模型 2 点 + 两条规则各 1 点。
MAX_SIGNAL_POINTS = FLAGGED_SIGNAL_POINTS + MATCHED_RULE_SIGNAL_POINTS * len(FROZEN_RULES)


class PolicyErrorCode:
    """政策加载失败的原因分类（用于启动诊断；不含路径或配置内容）。"""

    MISSING = "policy_missing"
    HASH_MISMATCH = "policy_hash_mismatch"
    PARSE_FAILED = "policy_parse_failed"
    SCHEMA_VIOLATION = "policy_schema_violation"
    PATH_UNSAFE = "policy_path_unsafe"
    NOT_CONFIGURED = "policy_not_configured"


class PolicyLoadError(RuntimeError):
    """政策无法安全加载。

    Raises 时**不得**附带绝对路径或文件内容片段；用 :attr:`code` 表达原因类别，
    细节由调用方以安全措辞记录。
    """

    def __init__(self, code: str, detail: str) -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}: {self.detail}")


@dataclass(frozen=True)
class RiskPolicy:
    """已通过指纹与严格 schema 校验的冻结政策。"""

    schema_version: str
    policy_version: str
    semantics: str
    threshold_source: str
    rules: tuple[RuleSpec, ...]
    bands: tuple[BandSpec, ...]
    flagged_signal_points: int
    matched_rule_signal_points: int
    rule_score_denominator: int
    disclaimer: str

    def rule_ids(self) -> tuple[str, ...]:
        return tuple(rule.rule_id for rule in self.rules)


def assert_project_relative_path(value: Any, label: str = "policy.path") -> str:
    """硬校验：路径必须是项目相对路径。

    判定是**平台无关**的：无论进程跑在 Windows 还是 POSIX 上，下面这些形状都被拒绝——

    - Windows 盘符绝对路径（``C:\\x``、``D:/x``）与盘符相对路径（``C:x``）；
    - UNC 共享与扩展长度前缀（``\\\\server\\share\\x``、``\\\\?\\C:\\x``）；
    - POSIX 绝对路径（``/etc/x``）与根相对路径（``\\x``）；
    - 任何含 ``..`` 组件的逃逸写法；
    - 疑似凭据片段（``token`` / ``secret`` 等）。

    刻意同时用 :class:`PureWindowsPath` 与 :class:`PurePosixPath` 判定，而不是只看宿主平台的
    ``Path``：政策文件可能在任一平台上被部署或审查，"在 Linux 上就放过 ``C:\\x``" 不是安全属性。

    与 CW-005 ``ServiceConfig._assert_project_relative`` 是**同一套规则的两个副本**：
    ``rules`` 包刻意不依赖 ``service`` 包（依赖方向是 service → rules），
    因此这里保留一份实现，作为请求链路之外的纵深防御。

    Raises:
        PolicyLoadError: 路径不安全。**先于任何文件 I/O** 抛出。
    """
    text = str(value)
    if not text.strip():
        raise PolicyLoadError(PolicyErrorCode.PATH_UNSAFE, f"{label} 不能为空")

    windows = PureWindowsPath(text)
    posix = PurePosixPath(text)
    drive_letter = len(text) > 1 and text[1] == ":"
    if (
        windows.is_absolute()
        or posix.is_absolute()
        or bool(windows.drive)
        or drive_letter
        or text.startswith(("/", "\\"))
    ):
        raise PolicyLoadError(PolicyErrorCode.PATH_UNSAFE, f"{label} 必须是项目相对路径")

    if ".." in PurePosixPath(text.replace("\\", "/")).parts:
        raise PolicyLoadError(PolicyErrorCode.PATH_UNSAFE, f"{label} 不得逃出项目根目录")

    lowered = text.lower()
    for marker in _CREDENTIAL_MARKERS:
        if marker in lowered:
            raise PolicyLoadError(PolicyErrorCode.PATH_UNSAFE, f"{label} 含疑似凭据字段 {marker}")
    return text


def sha256_of_file(path: Path, *, chunk_size: int = HASH_CHUNK_SIZE) -> str:
    """流式计算文件 SHA-256。不解析内容。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_policy_hash(path: Path, expected_sha256: str) -> str:
    """在 ``yaml.safe_load`` 之前校验政策指纹。

    Raises:
        PolicyLoadError: 文件缺失或指纹不符。**不得**在此之后才做校验。
    """
    if not path.is_file():
        raise PolicyLoadError(PolicyErrorCode.MISSING, "政策文件不存在")
    actual = sha256_of_file(path)
    if actual != expected_sha256:
        raise PolicyLoadError(
            PolicyErrorCode.HASH_MISMATCH,
            f"政策指纹不符：实际 {actual[:12]}… 期望 {expected_sha256[:12]}…",
        )
    return actual


# --- 严格 schema 校验 ---------------------------------------------------


def _violation(detail: str) -> PolicyLoadError:
    return PolicyLoadError(PolicyErrorCode.SCHEMA_VIOLATION, detail)


def _exact_keys(mapping: Any, expected: frozenset[str], label: str) -> None:
    """键集合必须**完全相等**：多一个字段就拒绝，不允许"读进来再忽略"。"""
    if not isinstance(mapping, dict):
        raise _violation(f"{label} 必须是映射")
    actual = {str(key) for key in mapping}
    if actual != expected:
        extra = sorted(actual - expected)
        missing = sorted(expected - actual)
        raise _violation(f"{label} 字段不匹配：多出 {extra}、缺少 {missing}")


def _finite_number(value: Any, label: str) -> float:
    """只接受有限整数/浮点。``bool`` 是 ``int`` 的子类，必须显式拒绝。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _violation(f"{label} 必须是数值")
    number = float(value)
    if not math.isfinite(number):
        raise _violation(f"{label} 必须是有限数")
    return number


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _violation(f"{label} 必须是整数")
    if value < 0:
        raise _violation(f"{label} 不得为负")
    return int(value)


def _parse_rules(raw: Any) -> tuple[RuleSpec, ...]:
    if not isinstance(raw, list):
        raise _violation("policy.rules 必须是列表")
    if len(raw) != len(FROZEN_RULES):
        raise _violation(f"policy.rules 必须恰好 {len(FROZEN_RULES)} 条")
    parsed: list[RuleSpec] = []
    for index, (declared, frozen) in enumerate(zip(raw, FROZEN_RULES)):
        label = f"policy.rules[{index}]"
        _exact_keys(declared, frozenset({"rule_id", "feature", "operator", "threshold"}), label)
        rule_id = declared["rule_id"]
        feature = declared["feature"]
        operator = declared["operator"]
        if not isinstance(rule_id, str) or rule_id != frozen.rule_id:
            raise _violation(f"{label}.rule_id 必须是 {frozen.rule_id}")
        if not isinstance(feature, str) or feature != frozen.feature:
            raise _violation(f"{label}.feature 必须是 {frozen.feature}")
        if not isinstance(operator, str) or operator != frozen.operator:
            # 只允许严格 `gt`：`>=` 会让"恰好等于阈值"也命中，冻结语义不可替换。
            raise _violation(f"{label}.operator 只允许 {frozen.operator!r}")
        threshold = _finite_number(declared["threshold"], f"{label}.threshold")
        if threshold != frozen.threshold:
            # 阈值只能等于 CW-003 train-only q0.99 的结果，不得现场调参。
            raise _violation(f"{label}.threshold 与冻结阈值不一致")
        parsed.append(RuleSpec(rule_id=rule_id, feature=feature, operator=operator, threshold=threshold))

    # 唯一性：重复 rule ID 或重复 feature 都拒绝（顺序已由上面的逐位比对固定）。
    ids = [rule.rule_id for rule in parsed]
    features = [rule.feature for rule in parsed]
    if len(set(ids)) != len(ids):
        raise _violation("policy.rules 存在重复 rule_id")
    if len(set(features)) != len(features):
        raise _violation("policy.rules 存在重复 feature")
    return tuple(parsed)


def _parse_bands(raw: Any) -> tuple[BandSpec, ...]:
    expected = {band.name for band in FROZEN_BANDS}
    _exact_keys(raw, frozenset(expected), "policy.fusion.bands")
    parsed: list[BandSpec] = []
    for frozen in FROZEN_BANDS:
        span = raw[frozen.name]
        label = f"policy.fusion.bands.{frozen.name}"
        if not isinstance(span, list) or len(span) != 2:
            raise _violation(f"{label} 必须是 [下界, 上界]")
        low = _positive_int(span[0], f"{label}[0]")
        high = _positive_int(span[1], f"{label}[1]")
        if (low, high) != (frozen.low, frozen.high):
            # 融合真值表是冻结的：配置不得改点值或移动区间。
            raise _violation(f"{label} 与冻结区间不一致")
        parsed.append(BandSpec(frozen.name, low, high))
    return tuple(parsed)


def _parse_fusion(raw: Any) -> tuple[int, int, int, tuple[BandSpec, ...]]:
    _exact_keys(
        raw,
        frozenset(
            {
                "flagged_signal_points",
                "matched_rule_signal_points",
                "rule_score_denominator",
                "bands",
            }
        ),
        "policy.fusion",
    )
    flagged = _positive_int(raw["flagged_signal_points"], "policy.fusion.flagged_signal_points")
    per_rule = _positive_int(
        raw["matched_rule_signal_points"], "policy.fusion.matched_rule_signal_points"
    )
    denominator = _positive_int(raw["rule_score_denominator"], "policy.fusion.rule_score_denominator")
    if flagged != FLAGGED_SIGNAL_POINTS:
        raise _violation("policy.fusion.flagged_signal_points 与冻结点值不一致")
    if per_rule != MATCHED_RULE_SIGNAL_POINTS:
        raise _violation("policy.fusion.matched_rule_signal_points 与冻结点值不一致")
    if denominator != RULE_SCORE_DENOMINATOR:
        raise _violation("policy.fusion.rule_score_denominator 与冻结分母不一致")
    return flagged, per_rule, denominator, _parse_bands(raw["bands"])


def parse_policy(raw: Any) -> RiskPolicy:
    """把已解析的 YAML 映射转成冻结政策；任何偏差都抛 :class:`PolicyLoadError`。"""
    _exact_keys(raw, frozenset({"policy"}), "政策顶层")
    section = raw["policy"]
    _exact_keys(
        section,
        frozenset(
            {
                "schema_version",
                "policy_version",
                "semantics",
                "threshold_source",
                "rules",
                "fusion",
                "disclaimer",
            }
        ),
        "policy",
    )

    schema_version = section["schema_version"]
    if not isinstance(schema_version, str) or schema_version != POLICY_SCHEMA_VERSION:
        raise _violation(f"policy.schema_version 必须是 {POLICY_SCHEMA_VERSION!r}")
    policy_version = section["policy_version"]
    if not isinstance(policy_version, str) or policy_version != POLICY_VERSION:
        raise _violation(f"policy.policy_version 必须是 {POLICY_VERSION!r}")
    semantics = section["semantics"]
    if not isinstance(semantics, str) or semantics != POLICY_SEMANTICS:
        raise _violation(f"policy.semantics 必须是 {POLICY_SEMANTICS!r}")
    threshold_source = section["threshold_source"]
    if not isinstance(threshold_source, str) or threshold_source != THRESHOLD_SOURCE:
        raise _violation(f"policy.threshold_source 必须是 {THRESHOLD_SOURCE!r}")

    disclaimer = section["disclaimer"]
    if not isinstance(disclaimer, str) or not disclaimer.strip():
        raise _violation("policy.disclaimer 必须是非空字符串")

    rules = _parse_rules(section["rules"])
    flagged, per_rule, denominator, bands = _parse_fusion(section["fusion"])

    return RiskPolicy(
        schema_version=schema_version,
        policy_version=policy_version,
        semantics=semantics,
        threshold_source=threshold_source,
        rules=rules,
        bands=bands,
        flagged_signal_points=flagged,
        matched_rule_signal_points=per_rule,
        rule_score_denominator=denominator,
        disclaimer=disclaimer.strip(),
    )


def load_policy(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_schema_version: str,
) -> tuple[RiskPolicy, str]:
    """按"先验哈希、后解析"的顺序加载政策。

    Args:
        path: 政策路径。**必须是项目相对路径**——绝对路径、盘符路径、UNC、``..`` 逃逸与
            疑似凭据路径一律拒绝，且拒绝发生在**任何文件 I/O 之前**。这里没有"测试专用"
            的放宽开关：生产入口 ``load_policy_engine()`` 也从不放宽，因此底层加载边界
            无法被绕过。
        expected_sha256: 服务配置里固定的政策指纹。
        expected_schema_version: 服务配置里声明的 policy schema 版本。

    Returns:
        ``(冻结政策, 实际摘要)``。

    Raises:
        PolicyLoadError: 路径不安全、文件缺失、指纹不符、YAML 非法或 schema 违规。
            调用方必须把政策相关能力置为"未就绪"，**不得**回退到内置默认政策。
    """
    if not re.fullmatch(r"[0-9a-f]{64}", str(expected_sha256)):
        raise PolicyLoadError(PolicyErrorCode.SCHEMA_VIOLATION, "期望的政策指纹格式不合法")
    if str(expected_schema_version) != POLICY_SCHEMA_VERSION:
        raise PolicyLoadError(
            PolicyErrorCode.SCHEMA_VIOLATION,
            f"期望的 policy schema_version 必须是 {POLICY_SCHEMA_VERSION!r}",
        )

    # 1) 路径安全：**无条件**先做，先于存在性与指纹检查。
    relative = assert_project_relative_path(path)
    policy_path = resolve_path(relative)

    # 2) 存在性 + 流式指纹校验。**这一步通过之前不会解析任何内容。**
    digest = verify_policy_hash(policy_path, str(expected_sha256))

    # 3) 只有校验通过后才解析。
    try:
        raw = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise PolicyLoadError(
            PolicyErrorCode.PARSE_FAILED, f"政策文件不是 UTF-8 文本：{type(error).__name__}"
        ) from error
    except yaml.YAMLError as error:
        raise PolicyLoadError(
            PolicyErrorCode.PARSE_FAILED, f"政策 YAML 解析失败：{type(error).__name__}"
        ) from error

    # 4) 严格校验：任何偏差都不得静默放宽冻结语义。
    policy = parse_policy(raw)
    return policy, digest
