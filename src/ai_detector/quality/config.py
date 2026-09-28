"""CW-009 质量门禁配置：严格 schema + 规范相对路径 + 冻结指纹。

## 为什么解析顺序也是安全边界

``configs/quality_gate.yaml`` 决定"拿哪四个文件、按哪个哈希核对、跑哪条命令"。因此本模块
把它当作**外部输入**处理：

1. 先按二进制读入并算摘要（配置自身没有冻结哈希，它由版本库与评审保证）；
2. 只接受**唯一**的顶层键 ``quality_gate`` 与**精确**的字段集合——多一个键、少一个键、
   类型不对（包括用 ``true`` 冒充 ``1``）、``NaN``/``Infinity``、重复项都立即失败；
3. 所有输入/输出路径必须是**规范的项目相对路径**：禁止绝对路径、盘符、UNC、反斜杠、
   ``.``/``..`` 组件、连续斜杠；解析后再做一次"不得越出项目根"的符号链接逃逸检查；
4. 输入与输出**不得互为别名**——报告不得覆盖任何被预检的输入。

## 命令形状写死在 schema 里

``pytest.arguments`` 只允许恰好 ``["-m", "pytest"]``。这不是多余的限制：门禁的全部价值
来自"跑的就是完整测试套件"，所以配置文件**不能**把它换成 ``-k``、``--ignore``、``--deselect``
之类的选择器，也不能追加任何"跳过某些用例"的参数。容差同理，上限锁死在
:data:`MAX_SCORE_ABSOLUTE_TOLERANCE`。
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

from ai_detector.config import PROJECT_ROOT
from ai_detector.quality.errors import QualityGateError, QualityGateErrorCode

#: 本配置的 schema 版本。未知版本即失败。
CONFIG_SCHEMA_VERSION = "cw009_quality_gate_config_v1"

#: 固定工作包标识。
WORK_PACKAGE = "CW-009"

#: 默认配置路径（相对项目根目录）。
DEFAULT_CONFIG_PATH = "configs/quality_gate.yaml"

#: 默认报告路径（相对项目根目录，且继续被 ``.gitignore`` 的 ``artifacts/**`` 排除）。
DEFAULT_REPORT_PATH = "artifacts/reports/cw009_quality_gate/report.json"

#: 唯一允许的 pytest 模块调用形状。配置不得把它换成别的选择器。
FROZEN_PYTEST_ARGUMENTS: tuple[str, ...] = ("-m", "pytest")

#: 分数绝对容差的上限。golden master 回归要的是"逐值不变"，不是"大致相同"，
#: 因此宽松容差本身就是配置错误。
MAX_SCORE_ABSOLUTE_TOLERANCE = 1e-6

#: pytest 超时的上限（秒）。要求"有限超时"，同时不许设成近乎无限。
MAX_PYTEST_TIMEOUT_SECONDS = 86400.0

_TOP_LEVEL_KEYS = frozenset({"quality_gate"})
_SECTION_KEYS = frozenset(
    {"schema_version", "work_package", "inputs", "comparison", "pytest", "report"}
)
_INPUT_SECTION_KEYS = frozenset({"service_config", "bundle", "policy", "corpus"})
_SERVICE_CONFIG_KEYS = frozenset({"path", "sha256"})
_BUNDLE_KEYS = frozenset({"path", "sha256", "model_version", "expected_source_features"})
_POLICY_KEYS = frozenset({"path", "sha256", "policy_version", "expected_rule_count"})
_CORPUS_KEYS = frozenset({"path", "sha256", "case_count"})
_COMPARISON_KEYS = frozenset({"score_absolute_tolerance"})
_PYTEST_KEYS = frozenset({"arguments", "timeout_seconds"})
_REPORT_KEYS = frozenset({"path"})

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_PATH_SEGMENT_PATTERN = re.compile(r"[A-Za-z0-9._-]+")

#: 输入相对路径 -> 供"输入输出不得互为别名"使用。
_INPUT_LABELS = ("service_config", "bundle", "policy", "corpus")


def _violation(detail: str) -> QualityGateError:
    return QualityGateError(QualityGateErrorCode.CONFIG_SCHEMA_VIOLATION, detail)


def _unsafe(detail: str) -> QualityGateError:
    return QualityGateError(QualityGateErrorCode.CONFIG_PATH_UNSAFE, detail)


def _exact_keys(mapping: Any, expected: frozenset[str], label: str) -> None:
    """键集合必须**完全相等**：多一个字段就拒绝，不允许"读进来再忽略"。"""
    if not isinstance(mapping, dict):
        raise _violation(f"{label} 必须是映射")
    actual = {str(key) for key in mapping}
    if actual != expected:
        extra = sorted(actual - expected)
        missing = sorted(expected - actual)
        raise _violation(f"{label} 字段不匹配：多出 {extra}、缺少 {missing}")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _violation(f"{label} 必须是非空字符串")
    return value


def _sha256(value: Any, label: str) -> str:
    """只接受 64 位**小写**十六进制。大写、带空格、带前缀都拒绝。"""
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise _violation(f"{label} 必须是 64 位小写十六进制 SHA-256")
    return value


def _real_number(value: Any, label: str) -> float:
    """只接受有限实数。``bool`` 是 ``int`` 的子类，必须显式拒绝。"""
    if isinstance(value, bool):
        raise _violation(f"{label} 必须是数值，不能是布尔值")
    if not isinstance(value, (int, float)):
        raise _violation(f"{label} 必须是数值")
    number = float(value)
    if not math.isfinite(number):
        raise _violation(f"{label} 必须是有限数")
    return number


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _violation(f"{label} 必须是整数")
    if value <= 0:
        raise _violation(f"{label} 必须是正整数")
    return int(value)


def canonical_relative_path(value: Any, label: str) -> str:
    """校验"规范的项目相对路径"。

    拒绝：绝对路径、盘符路径、UNC、反斜杠、空路径、空/``.``/``..`` 组件、连续斜杠、
    以斜杠开头或结尾，以及任何带非法字符的组件。判定是**平台无关**的——无论进程跑在
    Windows 还是 POSIX 上，``C:\\x``、``/etc/x``、``\\\\server\\share`` 都被拒绝。

    Raises:
        QualityGateError: 路径形状不安全（``config_path_unsafe``）。
    """
    if not isinstance(value, str) or not value:
        raise _unsafe(f"{label} 必须是非空字符串")
    if "\\" in value:
        raise _unsafe(f"{label} 不得包含反斜杠，只接受规范的正斜杠相对路径")
    if value.startswith("/") or value.endswith("/"):
        raise _unsafe(f"{label} 不得以斜杠开头或结尾")
    if ":" in value:
        raise _unsafe(f"{label} 不得包含盘符或协议前缀")
    parts = value.split("/")
    for part in parts:
        if part in ("", ".", ".."):
            raise _unsafe(f"{label} 不得包含空组件、'.' 或 '..'")
        if not _PATH_SEGMENT_PATTERN.fullmatch(part):
            raise _unsafe(f"{label} 含非法字符的路径组件")
    return value


def resolve_within_project(relative: str, label: str) -> Path:
    """把已校验的相对路径解析为绝对路径，并确认**未**通过符号链接逃出项目根。

    Raises:
        QualityGateError: 解析结果落在项目根之外（``config_path_unsafe``）。
    """
    root = PROJECT_ROOT.resolve()
    resolved = (root / relative).resolve()
    if resolved != root and not resolved.is_relative_to(root):
        raise _unsafe(f"{label} 解析后越出项目根目录")
    return resolved


def _existing_file(relative: str, label: str) -> Path:
    """输入必须是**已存在的普通文件**；目录、缺失或符号链接指向别处都失败。"""
    path = resolve_within_project(relative, label)
    if not path.exists():
        raise QualityGateError(QualityGateErrorCode.INPUT_MISSING, f"{label} 不存在")
    if not path.is_file():
        raise _unsafe(f"{label} 必须是普通文件")
    return path


# ---------------------------------------------------------------------------
# 严格 YAML 加载：**任意层级**拒绝重复映射键
#
# PyYAML 的 ``safe_load`` 对重复键是"后者覆盖前者"，而且**不报错**——相同值的重复键
# 也一样静默通过。这意味着评审看到的文本与解析器实际采用的对象可能不同：有人在配置里
# 多写一行 ``schema_version``，字段集合校验（只看已经去重的 dict）根本看不见。
#
# 因此门禁的配置解析走一个自定义的 SafeLoader 子类：它只加一条检查——同一个映射里
# 出现两次的键立即失败。行号可以安全报出；**键名与取值一律不进入错误信息**。
# ---------------------------------------------------------------------------


class DuplicateMappingKeyError(yaml.YAMLError):
    """YAML 映射里出现重复键（无论取值相同还是冲突）。只携带行号。"""

    def __init__(self, line: int) -> None:
        self.line = int(line)
        super().__init__(f"重复的映射键（第 {self.line} 行）")


class _StrictSafeLoader(yaml.SafeLoader):
    """``SafeLoader`` + 全层级重复键检测。只在本模块内使用。"""


def _construct_unique_mapping(loader: Any, node: Any, deep: bool = False) -> dict[Any, Any]:
    """构造映射，遇到重复键立即抛 :class:`DuplicateMappingKeyError`。

    先 ``flatten_mapping`` 展开 ``<<`` 合并键：展开之后才出现的重复同样拒绝，
    因此"用锚点绕过去"这条路也被堵住。
    """
    if not isinstance(node, yaml.MappingNode):
        raise yaml.constructor.ConstructorError(
            None, None, f"期望映射节点，实际为 {node.id}", node.start_mark
        )
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            hash(key)
        except TypeError as error:
            raise yaml.constructor.ConstructorError(
                "构造映射时", node.start_mark, "发现不可哈希的键", key_node.start_mark
            ) from error
        if key in mapping:
            raise DuplicateMappingKeyError(key_node.start_mark.line + 1)
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def strict_yaml_load(text: str) -> Any:
    """严格加载 YAML 文本。

    Raises:
        DuplicateMappingKeyError: 任意层级出现重复映射键。
        yaml.YAMLError: 其它 YAML 语法/构造错误。
    """
    return yaml.load(text, Loader=_StrictSafeLoader)  # noqa: S506 - 用的是 SafeLoader 子类


@dataclass(frozen=True)
class FrozenInput:
    """一个冻结输入：规范相对路径 + 冻结摘要 + 只读解析结果。"""

    label: str
    relative_path: str
    sha256: str

    def resolved(self) -> Path:
        return _existing_file(self.relative_path, self.label)


@dataclass(frozen=True)
class QualityGateConfig:
    """``configs/quality_gate.yaml`` 的类型化表示。"""

    schema_version: str
    work_package: str
    service_config: FrozenInput
    bundle: FrozenInput
    model_version: str
    expected_source_features: int
    policy: FrozenInput
    policy_version: str
    expected_rule_count: int
    corpus: FrozenInput
    corpus_case_count: int
    score_absolute_tolerance: float
    pytest_arguments: tuple[str, ...]
    pytest_timeout_seconds: float
    report_path: str
    config_path: str = DEFAULT_CONFIG_PATH

    # --- 契约交叉核对 ---------------------------------------------------

    def check_corpus_agreement(self, corpus: Any) -> None:
        """配置与语料必须对同一件事说同样的话。

        Raises:
            QualityGateError: 容差、case 数、特征数、版本或指纹不一致
                （``agreement_violation``）。
        """
        if float(corpus.score_absolute_tolerance) != self.score_absolute_tolerance:
            raise QualityGateError(
                QualityGateErrorCode.AGREEMENT_VIOLATION,
                "语料与配置的 score_absolute_tolerance 不一致",
            )
        if len(corpus.cases) != self.corpus_case_count:
            raise QualityGateError(
                QualityGateErrorCode.AGREEMENT_VIOLATION,
                "语料 case 数与配置冻结的 case_count 不一致",
            )
        if len(corpus.source_features) != self.expected_source_features:
            raise QualityGateError(
                QualityGateErrorCode.AGREEMENT_VIOLATION,
                "语料源特征数与配置冻结的 expected_source_features 不一致",
            )
        if str(corpus.model_version) != self.model_version:
            raise QualityGateError(
                QualityGateErrorCode.AGREEMENT_VIOLATION, "语料 model_version 与配置不一致"
            )
        if str(corpus.policy_version) != self.policy_version:
            raise QualityGateError(
                QualityGateErrorCode.AGREEMENT_VIOLATION, "语料 policy_version 与配置不一致"
            )
        if str(corpus.bundle_sha256) != self.bundle.sha256:
            raise QualityGateError(
                QualityGateErrorCode.AGREEMENT_VIOLATION, "语料 bundle_sha256 与配置不一致"
            )
        if str(corpus.policy_sha256) != self.policy.sha256:
            raise QualityGateError(
                QualityGateErrorCode.AGREEMENT_VIOLATION, "语料 policy_sha256 与配置不一致"
            )

    def check_runtime_agreement(self, runtime: Any, engine: Any) -> None:
        """配置冻结的版本/数量必须与真实加载出来的运行时一致。

        Raises:
            QualityGateError: 模型版本、源特征数、政策版本或规则条数不一致
                （``agreement_violation``）。
        """
        if str(runtime.model_version) != self.model_version:
            raise QualityGateError(
                QualityGateErrorCode.AGREEMENT_VIOLATION, "运行时 model_version 与配置不一致"
            )
        if len(runtime.source_features) != self.expected_source_features:
            raise QualityGateError(
                QualityGateErrorCode.AGREEMENT_VIOLATION, "运行时源特征数与配置不一致"
            )
        if str(runtime.bundle_sha256) != self.bundle.sha256:
            raise QualityGateError(
                QualityGateErrorCode.AGREEMENT_VIOLATION, "运行时 bundle 摘要与配置不一致"
            )
        if str(engine.policy_version) != self.policy_version:
            raise QualityGateError(
                QualityGateErrorCode.AGREEMENT_VIOLATION, "政策版本与配置不一致"
            )
        if str(engine.policy_sha256) != self.policy.sha256:
            raise QualityGateError(
                QualityGateErrorCode.AGREEMENT_VIOLATION, "政策摘要与配置不一致"
            )
        if len(engine.rules) != self.expected_rule_count:
            raise QualityGateError(
                QualityGateErrorCode.AGREEMENT_VIOLATION, "政策规则条数与配置不一致"
            )

    def report_target(self) -> Path:
        """报告的绝对路径（已做越界与类型检查）。"""
        path = resolve_within_project(self.report_path, "report.path")
        if path.exists() and path.is_dir():
            raise _unsafe("report.path 指向一个目录，不能作为报告文件")
        parent = path.parent
        if parent.exists() and not parent.is_dir():
            raise _unsafe("report.path 的父路径不是目录")
        return path

    # --- 加载 -----------------------------------------------------------

    @classmethod
    def from_mapping(cls, raw: Any, *, config_path: str = DEFAULT_CONFIG_PATH) -> QualityGateConfig:
        """从已解析的 YAML 映射构造配置。

        Raises:
            QualityGateError: 缺段、缺字段、类型不符、取值非法、路径不安全或输入输出互为别名。
        """
        _exact_keys(raw, _TOP_LEVEL_KEYS, "配置顶层")
        section = raw["quality_gate"]
        _exact_keys(section, _SECTION_KEYS, "quality_gate")

        schema_version = section["schema_version"]
        if not isinstance(schema_version, str) or schema_version != CONFIG_SCHEMA_VERSION:
            raise _violation(f"quality_gate.schema_version 必须是 {CONFIG_SCHEMA_VERSION!r}")
        work_package = section["work_package"]
        if not isinstance(work_package, str) or work_package != WORK_PACKAGE:
            raise _violation(f"quality_gate.work_package 必须是 {WORK_PACKAGE!r}")

        inputs = section["inputs"]
        _exact_keys(inputs, _INPUT_SECTION_KEYS, "quality_gate.inputs")

        service_raw = inputs["service_config"]
        _exact_keys(service_raw, _SERVICE_CONFIG_KEYS, "quality_gate.inputs.service_config")
        service_config = FrozenInput(
            label="service_config",
            relative_path=canonical_relative_path(
                service_raw["path"], "quality_gate.inputs.service_config.path"
            ),
            sha256=_sha256(
                service_raw["sha256"], "quality_gate.inputs.service_config.sha256"
            ),
        )

        bundle_raw = inputs["bundle"]
        _exact_keys(bundle_raw, _BUNDLE_KEYS, "quality_gate.inputs.bundle")
        bundle = FrozenInput(
            label="bundle",
            relative_path=canonical_relative_path(
                bundle_raw["path"], "quality_gate.inputs.bundle.path"
            ),
            sha256=_sha256(bundle_raw["sha256"], "quality_gate.inputs.bundle.sha256"),
        )

        policy_raw = inputs["policy"]
        _exact_keys(policy_raw, _POLICY_KEYS, "quality_gate.inputs.policy")
        policy = FrozenInput(
            label="policy",
            relative_path=canonical_relative_path(
                policy_raw["path"], "quality_gate.inputs.policy.path"
            ),
            sha256=_sha256(policy_raw["sha256"], "quality_gate.inputs.policy.sha256"),
        )

        corpus_raw = inputs["corpus"]
        _exact_keys(corpus_raw, _CORPUS_KEYS, "quality_gate.inputs.corpus")
        corpus = FrozenInput(
            label="corpus",
            relative_path=canonical_relative_path(
                corpus_raw["path"], "quality_gate.inputs.corpus.path"
            ),
            sha256=_sha256(corpus_raw["sha256"], "quality_gate.inputs.corpus.sha256"),
        )

        comparison = section["comparison"]
        _exact_keys(comparison, _COMPARISON_KEYS, "quality_gate.comparison")
        tolerance = _real_number(
            comparison["score_absolute_tolerance"],
            "quality_gate.comparison.score_absolute_tolerance",
        )
        if not 0.0 <= tolerance <= MAX_SCORE_ABSOLUTE_TOLERANCE:
            raise _violation(
                "quality_gate.comparison.score_absolute_tolerance 必须位于 "
                f"[0, {MAX_SCORE_ABSOLUTE_TOLERANCE:g}]：golden master 不接受宽松容差"
            )

        pytest_section = section["pytest"]
        _exact_keys(pytest_section, _PYTEST_KEYS, "quality_gate.pytest")
        arguments = pytest_section["arguments"]
        if not isinstance(arguments, list) or any(not isinstance(item, str) for item in arguments):
            raise _violation("quality_gate.pytest.arguments 必须是字符串列表")
        if tuple(arguments) != FROZEN_PYTEST_ARGUMENTS:
            raise _violation(
                "quality_gate.pytest.arguments 只允许 "
                f"{list(FROZEN_PYTEST_ARGUMENTS)}：门禁不得改用其它测试选择器"
            )
        timeout = _real_number(pytest_section["timeout_seconds"], "quality_gate.pytest.timeout_seconds")
        if not 0.0 < timeout <= MAX_PYTEST_TIMEOUT_SECONDS:
            raise _violation(
                "quality_gate.pytest.timeout_seconds 必须是 (0, "
                f"{MAX_PYTEST_TIMEOUT_SECONDS:g}] 内的有限数"
            )

        report_raw = section["report"]
        _exact_keys(report_raw, _REPORT_KEYS, "quality_gate.report")
        report_path = canonical_relative_path(report_raw["path"], "quality_gate.report.path")

        config = cls(
            schema_version=schema_version,
            work_package=work_package,
            service_config=service_config,
            bundle=bundle,
            model_version=_text(bundle_raw["model_version"], "quality_gate.inputs.bundle.model_version"),
            expected_source_features=_positive_int(
                bundle_raw["expected_source_features"],
                "quality_gate.inputs.bundle.expected_source_features",
            ),
            policy=policy,
            policy_version=_text(policy_raw["policy_version"], "quality_gate.inputs.policy.policy_version"),
            expected_rule_count=_positive_int(
                policy_raw["expected_rule_count"], "quality_gate.inputs.policy.expected_rule_count"
            ),
            corpus=corpus,
            corpus_case_count=_positive_int(
                corpus_raw["case_count"], "quality_gate.inputs.corpus.case_count"
            ),
            score_absolute_tolerance=tolerance,
            pytest_arguments=tuple(arguments),
            pytest_timeout_seconds=timeout,
            report_path=report_path,
            config_path=config_path,
        )
        config._check_no_alias()
        return config

    def _check_no_alias(self) -> None:
        """输入与输出**不得互为别名**：报告不能覆盖任何被预检的输入。

        比较的是解析后的绝对路径，因此 ``a/../b`` 之类的形状差异无法绕开
        （``..`` 本身已被 :func:`canonical_relative_path` 拒绝，这里是第二道）。
        """
        inputs = {
            label: resolve_within_project(getattr(self, label).relative_path, label)
            for label in _INPUT_LABELS
        }
        target = resolve_within_project(self.report_path, "report.path")
        for label, path in inputs.items():
            if path == target:
                raise QualityGateError(
                    QualityGateErrorCode.CONFIG_PATH_ALIAS,
                    f"report.path 与输入 {label} 指向同一路径",
                )

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> QualityGateConfig:
        """从 YAML 文件加载配置（相对项目根目录解析）。

        Raises:
            QualityGateError: 文件不可读、YAML 非法、**任意层级出现重复映射键**、
                顶层不是映射或字段违规。
        """
        relative = canonical_relative_path(str(path), "config 路径")
        config_path = resolve_within_project(relative, "config 路径")
        if not config_path.is_file():
            raise QualityGateError(
                QualityGateErrorCode.CONFIG_UNREADABLE, f"配置文件不存在：{relative}"
            )
        try:
            raw_text = config_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise QualityGateError(
                QualityGateErrorCode.CONFIG_UNREADABLE,
                f"配置文件不可读：{type(error).__name__}",
            ) from error
        try:
            raw = strict_yaml_load(raw_text)
        except DuplicateMappingKeyError as error:
            # 重复键是**配置内容违规**（不是"文件读不了"）：映射到 schema 违规。
            raise _violation(f"配置 YAML 存在重复映射键（第 {error.line} 行）") from None
        except yaml.YAMLError as error:
            raise QualityGateError(
                QualityGateErrorCode.CONFIG_UNREADABLE,
                f"配置 YAML 解析失败：{type(error).__name__}",
            ) from error
        if not isinstance(raw, dict):
            raise _violation("配置顶层必须是映射")
        return cls.from_mapping(raw, config_path=relative)


def sha256_of_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """**流式**计算文件 SHA-256：不解析、不反序列化、不把内容读进内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class InputDigest:
    """一个冻结输入的**二进制**预检事实。

    这是"先验哈希、后反序列化"这道边界的产物：只包含存在性、实际摘要与是否匹配，
    **不含任何文件内容**。``problem`` 为 ``None`` 表示该输入通过预检。
    """

    label: str
    relative_path: str
    declared_sha256: str
    actual_sha256: str | None
    match: bool
    problem: str | None

    def as_report_entry(self) -> dict[str, Any]:
        """报告里的安全条目：相对路径、冻结摘要、实际摘要、是否匹配。"""
        return {
            "path": self.relative_path,
            "declared_sha256": self.declared_sha256,
            "actual_sha256": self.actual_sha256,
            "match": self.match,
        }


def collect_frozen_digest(frozen: FrozenInput) -> InputDigest:
    """收集**一个**输入的二进制摘要。**任何情况下都不抛错**。

    不会因文件缺失或摘要不符而中断：缺失时 ``actual_sha256`` 为 ``None``，存在时一定
    记录真实摘要与 ``match``。判定由调用方在所有输入都收集完之后**统一**做——这样失败
    报告里四个输入的实际摘要才是完整的。

    这个函数是纯二进制的：不解析 YAML/JSON、不反序列化、不做任何推理。
    """
    try:
        path = _existing_file(frozen.relative_path, frozen.label)
    except QualityGateError as error:
        return InputDigest(
            label=frozen.label,
            relative_path=frozen.relative_path,
            declared_sha256=frozen.sha256,
            actual_sha256=None,
            match=False,
            problem=error.code,
        )
    actual = sha256_of_file(path)
    return InputDigest(
        label=frozen.label,
        relative_path=frozen.relative_path,
        declared_sha256=frozen.sha256,
        actual_sha256=actual,
        match=actual == frozen.sha256,
        problem=None if actual == frozen.sha256 else QualityGateErrorCode.INPUT_HASH_MISMATCH,
    )


def collect_frozen_digests(frozens: Sequence[FrozenInput]) -> tuple[InputDigest, ...]:
    """按给定顺序收集全部输入的二进制摘要（**不短路**）。"""
    return tuple(collect_frozen_digest(frozen) for frozen in frozens)


def first_problem(digests: Sequence[InputDigest]) -> str | None:
    """返回第一个未通过输入的稳定错误码（全部通过时为 ``None``）。

    顺序即输入的声明顺序，因此判定是确定的。
    """
    for digest in digests:
        if digest.problem is not None:
            return digest.problem
    return None


def verify_frozen_input(frozen: FrozenInput) -> tuple[Path, str]:
    """二进制核对**单个**冻结输入的指纹，不符时立即抛错。

    这是"先验哈希、后反序列化"的那道哈希：它在 ``joblib.load``、YAML/JSON 语义解析
    与任何模型推理**之前**执行。门禁主流程用的是 :func:`collect_frozen_digests` +
    :func:`first_problem`（要先收齐四个再判定）；本函数是同一实现之上的单输入便捷入口。

    Returns:
        ``(解析后的绝对路径, 实际摘要)``。

    Raises:
        QualityGateError: 文件缺失、类型不符或指纹不符。
    """
    digest = collect_frozen_digest(frozen)
    if digest.problem is not None:
        raise QualityGateError(digest.problem, f"{frozen.label} 未通过二进制预检")
    return _existing_file(frozen.relative_path, frozen.label), str(digest.actual_sha256)
