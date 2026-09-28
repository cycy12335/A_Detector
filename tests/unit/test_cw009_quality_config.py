"""CW-009 门禁配置：严格 schema、规范相对路径、冻结指纹与命令行面。

这些用例守住的是"配置文件不能把门禁改弱"：未知键、缺键、布尔冒充整数、``NaN``/``Infinity``、
绝对路径、``..`` 逃逸、输入输出互为别名、宽松容差、非冻结的 pytest 参数——全部失败关闭。
"""

from __future__ import annotations

import argparse
import copy
import re

import pytest
import yaml

from ai_detector.config import PROJECT_ROOT
from ai_detector.quality.config import (
    CONFIG_SCHEMA_VERSION,
    DEFAULT_CONFIG_PATH,
    DEFAULT_REPORT_PATH,
    FROZEN_PYTEST_ARGUMENTS,
    MAX_SCORE_ABSOLUTE_TOLERANCE,
    DuplicateMappingKeyError,
    QualityGateConfig,
    _StrictSafeLoader,
    strict_yaml_load,
)
from ai_detector.quality.errors import QualityGateError, QualityGateErrorCode
from ai_detector.quality.gate import build_parser

STAGE_ROOT = PROJECT_ROOT / "artifacts" / "tmp_tests" / "cw009"

CORPUS_SHA256 = "d58166b70223af0adb443b9acff0690e6e3c571110dc11b16c2395a8d6fdc4b0"


def _raw_config() -> dict:
    """真实配置的深拷贝（测试只改副本，绝不写回仓库配置）。"""
    return copy.deepcopy(
        yaml.safe_load((PROJECT_ROOT / DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
    )


def _section(raw: dict) -> dict:
    return raw["quality_gate"]


def _build(raw: dict) -> QualityGateConfig:
    return QualityGateConfig.from_mapping(raw)


def _expect(raw: dict, code: str) -> QualityGateError:
    with pytest.raises(QualityGateError) as info:
        _build(raw)
    assert info.value.code == code
    return info.value


# --- 重复映射键（原始文本层面） -----------------------------------------


def config_text() -> str:
    return (PROJECT_ROOT / DEFAULT_CONFIG_PATH).read_text(encoding="utf-8")


def stage_yaml(name: str, text: str) -> str:
    """把一段 YAML 文本写进项目内临时目录，返回项目相对路径。"""
    directory = STAGE_ROOT / name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "quality_gate.yaml"
    path.write_text(text, encoding="utf-8")
    return path.relative_to(PROJECT_ROOT).as_posix()


def duplicate_line(text: str, needle: str) -> str:
    """把第一处与 ``needle`` 逐字相同的整行**原样复制一份**在它后面。

    原样复制保证"重复键取值相同"这一最容易被静默接受的形态也被覆盖。
    """
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.strip() == needle:
            indent = line[: len(line) - len(line.lstrip())]
            return "".join(lines[: index + 1]) + indent + needle + "\n" + "".join(lines[index + 1 :])
    raise AssertionError(f"配置里找不到整行 {needle!r}")


# 顶层、每个嵌套段、每个输入块与报告块各取一行：覆盖"任意层级"。
DUPLICATE_CONFIG_LINES = (
    "schema_version: cw009_quality_gate_config_v1",
    "work_package: CW-009",
    "inputs:",
    "service_config:",
    "path: configs/service.yaml",
    "bundle:",
    "sha256: f02bb50e61c5f70c85448430c8bb653bd6e347f400220771500ee84a70135a1c",
    "model_version: cw004-lightgbm-challenger-v1",
    "policy_version: cw006-risk-policy-v1",
    "expected_rule_count: 2",
    "comparison:",
    "score_absolute_tolerance: 1.0e-12",
    "pytest:",
    'arguments: ["-m", "pytest"]',
    "timeout_seconds: 1800.0",
    "report:",
    "path: artifacts/reports/cw009_quality_gate/report.json",
)


def test_duplicate_line_helper_finds_every_probe_line() -> None:
    """先确认探针行确实存在：否则参数化测试会因为"找不到行"而假绿。"""
    text = config_text()
    for needle in DUPLICATE_CONFIG_LINES:
        assert duplicate_line(text, needle) != text, needle



# --- 正常路径 -----------------------------------------------------------


def test_real_config_freezes_expected_values() -> None:
    config = QualityGateConfig.from_yaml(DEFAULT_CONFIG_PATH)

    assert config.schema_version == CONFIG_SCHEMA_VERSION
    assert config.work_package == "CW-009"
    assert config.service_config.relative_path == "configs/service.yaml"
    assert config.bundle.relative_path == "artifacts/models/cw004/lightgbm_bundle.joblib"
    assert config.policy.relative_path == "configs/risk_policy.yaml"
    assert config.corpus.relative_path == "tests/fixtures/cw009/model_regression_cases.json"
    assert config.corpus.sha256 == CORPUS_SHA256
    assert config.model_version == "cw004-lightgbm-challenger-v1"
    assert config.expected_source_features == 199
    assert config.policy_version == "cw006-risk-policy-v1"
    assert config.expected_rule_count == 2
    assert config.corpus_case_count == 12
    assert config.score_absolute_tolerance == 1e-12
    assert config.pytest_arguments == FROZEN_PYTEST_ARGUMENTS
    assert config.pytest_timeout_seconds == 1800.0
    assert config.report_path == DEFAULT_REPORT_PATH
    # 报告路径必须仍然落在被 gitignore 的 artifacts/ 下。
    assert config.report_path.startswith("artifacts/")


def test_config_paths_resolve_inside_project_root() -> None:
    config = QualityGateConfig.from_yaml(DEFAULT_CONFIG_PATH)
    for frozen in (config.service_config, config.bundle, config.policy, config.corpus):
        path = frozen.resolved()
        assert path.is_file()
        assert path.is_relative_to(PROJECT_ROOT)
    assert config.report_target().parent.is_dir()


# --- schema 严格性 ------------------------------------------------------


def test_unknown_top_level_key_is_rejected() -> None:
    raw = _raw_config()
    raw["extra"] = {}
    _expect(raw, QualityGateErrorCode.CONFIG_SCHEMA_VIOLATION)


def test_unknown_section_key_is_rejected() -> None:
    raw = _raw_config()
    _section(raw)["accept_current"] = True
    _expect(raw, QualityGateErrorCode.CONFIG_SCHEMA_VIOLATION)


def test_unknown_input_key_is_rejected() -> None:
    raw = _raw_config()
    _section(raw)["inputs"]["oracle"] = {"path": "x"}
    _expect(raw, QualityGateErrorCode.CONFIG_SCHEMA_VIOLATION)


def test_missing_required_key_is_rejected() -> None:
    raw = _raw_config()
    del _section(raw)["inputs"]["bundle"]["sha256"]
    _expect(raw, QualityGateErrorCode.CONFIG_SCHEMA_VIOLATION)


def test_wrong_schema_version_is_rejected() -> None:
    raw = _raw_config()
    _section(raw)["schema_version"] = "cw009_quality_gate_config_v2"
    _expect(raw, QualityGateErrorCode.CONFIG_SCHEMA_VIOLATION)


def test_boolean_masquerading_as_integer_is_rejected() -> None:
    for field in ("expected_source_features", "case_count"):
        raw = _raw_config()
        inputs = _section(raw)["inputs"]
        target = inputs["bundle"] if field == "expected_source_features" else inputs["corpus"]
        target[field] = True
        _expect(raw, QualityGateErrorCode.CONFIG_SCHEMA_VIOLATION)


def test_boolean_masquerading_as_number_is_rejected() -> None:
    raw = _raw_config()
    _section(raw)["comparison"]["score_absolute_tolerance"] = True
    _expect(raw, QualityGateErrorCode.CONFIG_SCHEMA_VIOLATION)


def test_non_finite_tolerance_is_rejected() -> None:
    for token in (float("nan"), float("inf"), float("-inf")):
        raw = _raw_config()
        _section(raw)["comparison"]["score_absolute_tolerance"] = token
        _expect(raw, QualityGateErrorCode.CONFIG_SCHEMA_VIOLATION)


def test_non_finite_timeout_is_rejected() -> None:
    raw = _raw_config()
    _section(raw)["pytest"]["timeout_seconds"] = float("inf")
    _expect(raw, QualityGateErrorCode.CONFIG_SCHEMA_VIOLATION)


def test_loose_tolerance_is_rejected() -> None:
    """宽松容差本身就是配置错误——golden master 不接受"大致相同"。"""
    for value in (0.5, 1.0, MAX_SCORE_ABSOLUTE_TOLERANCE * 10):
        raw = _raw_config()
        _section(raw)["comparison"]["score_absolute_tolerance"] = value
        _expect(raw, QualityGateErrorCode.CONFIG_SCHEMA_VIOLATION)


def test_negative_tolerance_is_rejected() -> None:
    raw = _raw_config()
    _section(raw)["comparison"]["score_absolute_tolerance"] = -1e-12
    _expect(raw, QualityGateErrorCode.CONFIG_SCHEMA_VIOLATION)


def test_zero_timeout_is_rejected() -> None:
    raw = _raw_config()
    _section(raw)["pytest"]["timeout_seconds"] = 0.0
    _expect(raw, QualityGateErrorCode.CONFIG_SCHEMA_VIOLATION)


@pytest.mark.parametrize(
    "arguments",
    [
        ["-m", "pytest", "-k", "cw005"],
        ["-m", "pytest", "--ignore=tests/integration"],
        ["-m", "pytest", "--deselect", "tests/unit/test_x.py::test_y"],
        ["-m", "pytest", "-p", "no:cacheprovider"],
        ["pytest"],
        [],
        "pytest",
    ],
)
def test_pytest_arguments_must_be_frozen_module_call(arguments: object) -> None:
    raw = _raw_config()
    _section(raw)["pytest"]["arguments"] = arguments
    _expect(raw, QualityGateErrorCode.CONFIG_SCHEMA_VIOLATION)


def test_sha256_must_be_lowercase_hex() -> None:
    for bad in ("A" * 64, CORPUS_SHA256.upper(), CORPUS_SHA256[:63], "z" * 64, 123):
        raw = _raw_config()
        _section(raw)["inputs"]["corpus"]["sha256"] = bad
        _expect(raw, QualityGateErrorCode.CONFIG_SCHEMA_VIOLATION)


# --- 重复映射键：必须失败关闭，不能被解析器静默覆盖 ---------------------


def test_plain_yaml_safe_load_would_silently_accept_a_duplicate_key() -> None:
    """先记录根因：普通 ``safe_load`` 对重复键是"后者覆盖前者"且不报错。

    这条断言同时是区分性证明——如果门禁也用 ``safe_load``，下面的用例全会"通过"。
    """
    text = duplicate_line(config_text(), "work_package: CW-009")
    plain = yaml.safe_load(text)
    assert plain["quality_gate"]["work_package"] == "CW-009"  # 静默去重，看不出重复
    assert text.count("work_package: CW-009") == 2

    with pytest.raises(DuplicateMappingKeyError):
        strict_yaml_load(text)


@pytest.mark.parametrize("needle", DUPLICATE_CONFIG_LINES, ids=lambda item: item.split(":")[0])
def test_duplicate_config_key_at_any_level_exits_two(needle: str) -> None:
    """相同值的重复键——最容易被静默接受的形态——必须在任意层级失败关闭。"""
    text = duplicate_line(config_text(), needle)
    path = stage_yaml("dup_same", text)

    with pytest.raises(QualityGateError) as info:
        QualityGateConfig.from_yaml(path)

    assert info.value.code == QualityGateErrorCode.CONFIG_SCHEMA_VIOLATION
    assert "D:" not in info.value.detail and "\\" not in info.value.detail


@pytest.mark.parametrize(
    "needle, conflicting",
    [
        (
            "schema_version: cw009_quality_gate_config_v1",
            "schema_version: cw009_quality_gate_config_v2",
        ),
        ("work_package: CW-009", "work_package: CW-010"),
        ("expected_rule_count: 2", "expected_rule_count: 3"),
        ("score_absolute_tolerance: 1.0e-12", "score_absolute_tolerance: 0.5"),
        ("timeout_seconds: 1800.0", "timeout_seconds: 1.0"),
    ],
)
def test_conflicting_duplicate_config_key_exits_two(needle: str, conflicting: str) -> None:
    """取值冲突的重复键同样拒绝：绝不能"看解析器最后拿到哪个"。"""
    text = duplicate_line(config_text(), needle).replace(needle, conflicting, 1)
    path = stage_yaml("dup_conflict", text)

    with pytest.raises(QualityGateError) as info:
        QualityGateConfig.from_yaml(path)

    assert info.value.code == QualityGateErrorCode.CONFIG_SCHEMA_VIOLATION


def test_duplicate_config_key_never_reaches_field_validation() -> None:
    """重复键必须在**解析阶段**失败，而不是靠"字段集合精确"事后补救。

    这里把重复键放在一个本来就会通过所有字段校验的位置：若解析器先去重，
    ``from_mapping`` 会完全看不出问题。
    """
    text = duplicate_line(config_text(), "expected_source_features: 199")
    path = stage_yaml("dup_contract", text)

    # 先证明"去重后的对象"本身是合法的（否则这条用例不能归因于重复键）。
    assert QualityGateConfig.from_mapping(yaml.safe_load(text)) is not None

    with pytest.raises(QualityGateError) as info:
        QualityGateConfig.from_yaml(path)
    assert info.value.code == QualityGateErrorCode.CONFIG_SCHEMA_VIOLATION


def test_duplicate_mapping_key_error_carries_only_a_line_number() -> None:
    error = DuplicateMappingKeyError(14)
    assert error.line == 14
    assert "14" in str(error)


def test_strict_yaml_load_still_accepts_a_clean_document() -> None:
    assert strict_yaml_load("a: 1\nb:\n  c: 2\n") == {"a": 1, "b": {"c": 2}}


def test_strict_loader_does_not_change_the_global_safe_loader() -> None:
    """严格加载器必须是**局部**的：其它模块的 ``yaml.safe_load`` 行为不得改变。

    否则"门禁更严"就会变成"服务配置与政策解析也跟着变了"——那是越界修改。
    """
    import yaml as yaml_module

    # PyYAML 的 add_constructor 会在子类首次改动时 copy 一份构造器表；
    # 这里逐条确认 SafeLoader 本身仍是宽松的。
    assert yaml_module.safe_load("a: 1\na: 2\n") == {"a": 2}
    assert yaml_module.safe_load("a: 1\na: 1\n") == {"a": 1}
    assert yaml_module.SafeLoader.yaml_constructors is not _StrictSafeLoader.yaml_constructors

    # service / policy 的既有加载路径（普通 safe_load）仍然可读
    for relative in ("configs/service.yaml", "configs/risk_policy.yaml"):
        raw = yaml_module.safe_load((PROJECT_ROOT / relative).read_text(encoding="utf-8"))
        assert isinstance(raw, dict)

    # 而严格加载器只影响自己
    with pytest.raises(DuplicateMappingKeyError):
        strict_yaml_load("a: 1\na: 2\n")


def test_strict_yaml_load_rejects_duplicates_introduced_by_merge_keys() -> None:
    """``<<`` 展开后产生的重复键同样拒绝：不能靠锚点绕过去。"""
    text = "base: &b\n  a: 1\nmerged:\n  <<: *b\n  a: 2\n"
    with pytest.raises(DuplicateMappingKeyError):
        strict_yaml_load(text)


# --- 路径纪律 -----------------------------------------------------------

@pytest.mark.parametrize(
    "bad_path",
    [
        "C:/ai_detector/configs/service.yaml",
        "D:\\data\\service.yaml",
        "/etc/service.yaml",
        "\\\\server\\share\\service.yaml",
        "../configs/service.yaml",
        "configs/../configs/service.yaml",
        "./configs/service.yaml",
        "configs//service.yaml",
        "configs/",
        "",
    ],
)
def test_unsafe_input_paths_are_rejected(bad_path: str) -> None:
    raw = _raw_config()
    _section(raw)["inputs"]["service_config"]["path"] = bad_path
    _expect(raw, QualityGateErrorCode.CONFIG_PATH_UNSAFE)


def test_unsafe_report_path_is_rejected() -> None:
    raw = _raw_config()
    _section(raw)["report"]["path"] = "C:/tmp/report.json"
    _expect(raw, QualityGateErrorCode.CONFIG_PATH_UNSAFE)


def test_input_output_alias_is_rejected() -> None:
    """报告不得指向任何被预检的输入，否则门禁会覆盖自己核对过的证据。"""
    for label in ("service_config", "bundle", "policy", "corpus"):
        raw = _raw_config()
        _section(raw)["report"]["path"] = _section(raw)["inputs"][label]["path"]
        _expect(raw, QualityGateErrorCode.CONFIG_PATH_ALIAS)


def test_missing_input_file_is_rejected() -> None:
    raw = _raw_config()
    _section(raw)["inputs"]["corpus"]["path"] = "configs/definitely_missing_cw009.json"
    config = _build(raw)
    from ai_detector.quality.config import verify_frozen_input

    with pytest.raises(QualityGateError) as info:
        verify_frozen_input(config.corpus)
    assert info.value.code == QualityGateErrorCode.INPUT_MISSING


def test_input_must_be_a_file_not_a_directory() -> None:
    raw = _raw_config()
    _section(raw)["inputs"]["corpus"]["path"] = "src/ai_detector/quality"
    config = _build(raw)
    from ai_detector.quality.config import verify_frozen_input

    with pytest.raises(QualityGateError) as info:
        verify_frozen_input(config.corpus)
    assert info.value.code == QualityGateErrorCode.CONFIG_PATH_UNSAFE


def test_report_path_must_not_be_an_existing_directory() -> None:
    raw = _raw_config()
    _section(raw)["report"]["path"] = "configs"
    config = _build(raw)
    with pytest.raises(QualityGateError) as info:
        config.report_target()
    assert info.value.code == QualityGateErrorCode.CONFIG_PATH_UNSAFE


def test_report_parent_must_not_be_a_file() -> None:
    raw = _raw_config()
    _section(raw)["report"]["path"] = "pyproject.toml/report.json"
    config = _build(raw)
    with pytest.raises(QualityGateError) as info:
        config.report_target()
    assert info.value.code == QualityGateErrorCode.CONFIG_PATH_UNSAFE


def test_absolute_config_path_is_rejected() -> None:
    with pytest.raises(QualityGateError) as info:
        QualityGateConfig.from_yaml(str(PROJECT_ROOT / DEFAULT_CONFIG_PATH))
    assert info.value.code == QualityGateErrorCode.CONFIG_PATH_UNSAFE


def test_missing_config_file_is_rejected() -> None:
    with pytest.raises(QualityGateError) as info:
        QualityGateConfig.from_yaml("configs/definitely_missing_cw009.yaml")
    assert info.value.code == QualityGateErrorCode.CONFIG_UNREADABLE


# --- 命令行面 -----------------------------------------------------------


def test_cli_exposes_only_config_and_regression_only() -> None:
    """命令行**没有**跳过哈希、放宽容差或更新 oracle 的开关。"""
    parser = build_parser()
    options = sorted(
        option
        for action in parser._actions
        if not isinstance(action, argparse._HelpAction)
        for option in action.option_strings
    )
    assert options == ["--config", "--regression-only"]

    forbidden = re.compile(r"hash|tolerance|oracle|bless|update|accept|skip|force", re.I)
    assert not any(forbidden.search(option) for option in options)
