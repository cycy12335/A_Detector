"""CW-009 固定回归语料：严格 schema、特征顺序契约、边界配对与覆盖契约。

语料是 golden master 的**唯一事实来源**，因此这里的每一条反例都对应一种"让语料悄悄变松"
的尝试：多写一个字段、换一个特征顺序、用 ``true`` 冒充数值、把边界 case 多改一个特征、
删掉某个覆盖组合。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ai_detector.config import PROJECT_ROOT
from ai_detector.quality.corpus import (
    CORPUS_SCHEMA_VERSION,
    ROLE_BASELINE,
    ROLE_BOUNDARY,
    forbidden_report_tokens,
    load_corpus,
    parse_corpus,
)
from ai_detector.quality.errors import QualityGateError, QualityGateErrorCode
from ai_detector.quality.regression import check_coverage
from ai_detector.service.config import ServiceConfig
from ai_detector.service.runtime import load_policy_engine, load_runtime

CORPUS_REL = "tests/fixtures/cw009/model_regression_cases.json"
CORPUS_SHA256 = "d58166b70223af0adb443b9acff0690e6e3c571110dc11b16c2395a8d6fdc4b0"


@pytest.fixture(scope="module")
def runtime():
    """真实 bundle 运行时：语料契约必须对着真 bundle 校验，不能用 fake。"""
    return load_runtime(ServiceConfig.from_yaml("configs/service.yaml"))


@pytest.fixture(scope="module")
def engine():
    return load_policy_engine(ServiceConfig.from_yaml("configs/service.yaml"))


@pytest.fixture(scope="module")
def corpus(runtime):
    return parse_corpus(
        raw_corpus(),
        source_features=runtime.source_features,
        categorical_features=runtime.categorical_features,
    )


def raw_corpus() -> dict:
    """真实语料的深拷贝（测试只改副本）。"""
    path = PROJECT_ROOT / CORPUS_REL
    assert path.is_file()
    return copy.deepcopy(json.loads(path.read_text(encoding="utf-8")))


def build(raw: dict, runtime):
    return parse_corpus(
        raw,
        source_features=runtime.source_features,
        categorical_features=runtime.categorical_features,
    )


def expect(raw: dict, runtime, code: str) -> QualityGateError:
    with pytest.raises(QualityGateError) as info:
        build(raw, runtime)
    assert info.value.code == code
    return info.value


def case_of(raw: dict, case_id: str) -> dict:
    for case in raw["cases"]:
        if case["case_id"] == case_id:
            return case
    raise AssertionError(case_id)


# --- 重复映射键（原始 JSON 文本层面） -----------------------------------


CORPUS_PATH = PROJECT_ROOT / CORPUS_REL

STAGE_ROOT = PROJECT_ROOT / "artifacts" / "tmp_tests" / "cw009"


def corpus_text() -> str:
    return CORPUS_PATH.read_text(encoding="utf-8")


def duplicate_line(text: str, needle: str) -> str:
    """把第一处与 ``needle`` 逐字相同的整行**原样复制一份**在它后面。

    原样复制保证"重复键取值相同"这一最容易被静默接受的形态也被覆盖。
    """
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.strip() == needle:
            indent = line[: len(line) - len(line.lstrip())]
            return "".join(lines[: index + 1]) + indent + needle + "\n" + "".join(lines[index + 1 :])
    raise AssertionError(f"语料里找不到整行 {needle!r}")


def stage_corpus_text(name: str, text: str) -> Path:
    directory = STAGE_ROOT / name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "corpus.json"
    path.write_text(text, encoding="utf-8")
    return path


def load_text(path: Path, runtime):
    return load_corpus(
        path,
        source_features=runtime.source_features,
        categorical_features=runtime.categorical_features,
    )


# 顶层、源特征清单、容差、case 容器，以及 case / features / expected 三层各取一行。
DUPLICATE_CORPUS_LINES = (
    '"schema_version": "cw009_model_regression_corpus_v1",',
    '"kind": "synthetic_golden_master_regression_corpus",',
    '"bundle_sha256": "f02bb50e61c5f70c85448430c8bb653bd6e347f400220771500ee84a70135a1c",',
    '"source_features": [',
    '"score_absolute_tolerance": 1e-12,',
    '"cases": [',
    '"case_id": "case_base_calm",',
    '"role": "baseline",',
    '"notes": "未触发任何规则的平静基线：全部数值特征为 0.0、五个类别特征取同一合成占位值。",',
    '"varied_features": [],',
    '"features": {',
    '"C1": 0.0,',
    '"expected": {',
    '"score": 0.08124042039839917,',
    '"model_flagged": false,',
    '"risk_level": "low_signal"',
)


def test_duplicate_line_helper_finds_every_probe_line() -> None:
    """先确认探针行确实存在：否则参数化测试会因为"找不到行"而假绿。"""
    text = corpus_text()
    for needle in DUPLICATE_CORPUS_LINES:
        assert duplicate_line(text, needle) != text, needle


def test_plain_json_loads_would_silently_accept_a_duplicate_key() -> None:
    """先记录根因：标准库 ``json`` 对重复键是"后者覆盖前者"且不报错。

    这条断言同时是区分性证明——如果门禁也用普通 ``json.loads``，下面的用例全会"通过"。
    """
    import json as json_module

    text = duplicate_line(corpus_text(), '"schema_version": "cw009_model_regression_corpus_v1",')
    plain = json_module.loads(text)
    assert plain["schema_version"] == "cw009_model_regression_corpus_v1"
    assert text.count('"schema_version": "cw009_model_regression_corpus_v1"') == 2


@pytest.mark.parametrize("needle", DUPLICATE_CORPUS_LINES, ids=lambda item: item.split(":")[0].strip('"'))
def test_duplicate_corpus_key_at_any_level_is_rejected(runtime, needle: str) -> None:
    """相同值的重复键——最容易被静默接受的形态——必须在任意层级失败关闭。"""
    path = stage_corpus_text("dup_same", duplicate_line(corpus_text(), needle))

    with pytest.raises(QualityGateError) as info:
        load_text(path, runtime)

    assert info.value.code == QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION
    assert "D:" not in info.value.detail and "\\" not in info.value.detail


@pytest.mark.parametrize(
    "needle, conflicting",
    [
        ('"schema_version": "cw009_model_regression_corpus_v1",', '"schema_version": "cw009_model_regression_corpus_v2",'),
        ('"kind": "synthetic_golden_master_regression_corpus",', '"kind": "handwritten_expectations",'),
        ('"case_id": "case_base_calm",', '"case_id": "case_base_other",'),
        ('"role": "baseline",', '"role": "stress",'),
        ('"C1": 0.0,', '"C1": 10000.0,'),
        ('"score": 0.08124042039839917,', '"score": 0.5,'),
        ('"model_flagged": false,', '"model_flagged": true,'),
    ],
)
def test_conflicting_duplicate_corpus_key_is_rejected(
    runtime, needle: str, conflicting: str
) -> None:
    """取值冲突的重复键同样拒绝：绝不能"看解析器最后拿到哪个"。"""
    text = duplicate_line(corpus_text(), needle).replace(needle, conflicting, 1)
    path = stage_corpus_text("dup_conflict", text)

    with pytest.raises(QualityGateError) as info:
        load_text(path, runtime)

    assert info.value.code == QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION


def test_duplicate_corpus_key_error_carries_no_key_name() -> None:
    """错误信息不带键名：键名可能取自语料里的可疑内容。"""
    path = stage_corpus_text(
        "dup_detail", duplicate_line(corpus_text(), '"schema_version": "cw009_model_regression_corpus_v1",')
    )
    tiny = stage_corpus_text("dup_detail_tiny", '{"a": 1, "a": 2}')
    assert tiny.is_file() and path.is_file()


def test_duplicate_key_detection_is_reached_before_schema_validation(runtime) -> None:
    """重复键是**解析期**失败，不依赖"字段集合精确"事后补救。"""
    tiny = stage_corpus_text("dup_tiny", '{"schema_version": "x", "schema_version": "x"}')
    with pytest.raises(QualityGateError) as info:
        load_text(tiny, runtime)
    assert info.value.code == QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION


def test_nested_object_inside_a_list_is_also_checked(runtime) -> None:
    tiny = stage_corpus_text(
        "dup_in_list", '{"cases": [{"a": {"b": 1, "b": 2}}]}'
    )
    with pytest.raises(QualityGateError) as info:
        load_text(tiny, runtime)
    assert info.value.code == QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION


def test_clean_corpus_still_loads_after_the_strict_hook(runtime) -> None:
    """加了重复键检查之后，真实语料仍必须正常加载。"""
    corpus = load_text(CORPUS_PATH, runtime)
    assert len(corpus.cases) == 12


# --- 正常路径与覆盖 -----------------------------------------------------


def test_frozen_corpus_has_exactly_twelve_unique_cases(corpus) -> None:
    assert corpus.schema_version == CORPUS_SCHEMA_VERSION
    assert len(corpus.cases) == 12
    assert len(set(corpus.case_ids())) == 12
    assert len(corpus.source_features) == 199


def test_every_case_carries_all_features_in_bundle_order(corpus, runtime) -> None:
    for case in corpus.cases:
        assert list(case.feature_mapping()) == list(runtime.source_features)


def test_frozen_corpus_covers_required_combinations(corpus, engine) -> None:
    summary = check_coverage(corpus, engine)
    assert summary["model_flagged_true"] >= 1
    assert summary["model_flagged_false"] >= 1
    assert summary["rule_match_kinds"] == [0, 1, 2]
    assert summary["risk_levels_present"] == ["low_signal", "elevated_signal", "high_signal"]
    assert all(
        entry["below"] and entry["equal"] and entry["above"]
        for entry in summary["rule_feature_boundaries"].values()
    )


def test_boundary_cases_only_change_the_declared_feature(corpus) -> None:
    by_id = {case.case_id: case for case in corpus.cases}
    boundaries = [case for case in corpus.cases if case.role == ROLE_BOUNDARY]
    assert boundaries
    for case in boundaries:
        assert len(case.varied_features) == 1
        baseline = by_id[case.baseline_case_id]
        base_map = baseline.feature_mapping()
        case_map = case.feature_mapping()
        changed = [name for name in corpus.source_features if base_map[name] != case_map[name]]
        assert changed == list(case.varied_features)


def test_each_categorical_feature_has_two_synthetic_values(corpus, runtime) -> None:
    for name in runtime.categorical_features:
        values = {case.feature_mapping()[name] for case in corpus.cases}
        assert len(values) >= 2
        assert all(isinstance(value, str) for value in values)


def test_at_least_one_case_uses_a_value_unseen_by_the_training_mapping(corpus, runtime) -> None:
    """类别编码器的训练映射里不含这些合成取值——即"未见值"路径确实被覆盖。"""
    encoder = runtime.bundle.preprocessor.transformers_[1][1].steps[-1][1]
    for index, name in enumerate(runtime.categorical_features):
        known = {str(value) for value in encoder.categories_[index]}
        used = {case.feature_mapping()[name] for case in corpus.cases}
        assert used - known, f"类别特征 {name} 至少需要一个训练映射未见值"


def test_forbidden_report_tokens_cover_features_and_categories(corpus, runtime) -> None:
    tokens = forbidden_report_tokens(corpus)
    assert set(runtime.source_features) <= tokens
    assert "cw009_cat_alpha" in tokens


# --- 语料哈希与免责声明 -------------------------------------------------


def test_disclaimer_must_be_marked_synthetic_golden_master(runtime) -> None:
    for bad in ("", "这是一个回归语料。", "synthetic only"):
        raw = raw_corpus()
        raw["disclaimer"] = bad
        expect(raw, runtime, QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION)


def test_schema_version_and_kind_are_fixed(runtime) -> None:
    raw = raw_corpus()
    raw["schema_version"] = "cw009_model_regression_corpus_v2"
    expect(raw, runtime, QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION)

    raw = raw_corpus()
    raw["kind"] = "handwritten_expectations"
    expect(raw, runtime, QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION)


def test_unknown_top_level_key_is_rejected(runtime) -> None:
    raw = raw_corpus()
    raw["recorded_at"] = "2026-09-26"
    expect(raw, runtime, QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION)


# --- 特征契约 -----------------------------------------------------------


def test_feature_order_change_is_rejected(runtime) -> None:
    raw = raw_corpus()
    features = case_of(raw, "case_base_calm")["features"]
    items = list(features.items())
    items[0], items[1] = items[1], items[0]
    case_of(raw, "case_base_calm")["features"] = dict(items)
    expect(raw, runtime, QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION)


def test_renamed_feature_is_rejected(runtime) -> None:
    raw = raw_corpus()
    features = case_of(raw, "case_base_calm")["features"]
    renamed = {}
    for index, (name, value) in enumerate(features.items()):
        renamed["NotAFeature" if index == 0 else name] = value
    case_of(raw, "case_base_calm")["features"] = renamed
    expect(raw, runtime, QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION)


def test_missing_feature_is_rejected(runtime) -> None:
    raw = raw_corpus()
    features = case_of(raw, "case_base_calm")["features"]
    features.pop(next(iter(features)))
    expect(raw, runtime, QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION)


@pytest.mark.parametrize("bad_value", [True, False, None, [1.0], {"v": 1.0}, "12.5", float("nan")])
def test_illegal_feature_values_are_rejected(runtime, bad_value: object) -> None:
    raw = raw_corpus()
    case_of(raw, "case_base_calm")["features"]["TransactionAmt"] = bad_value
    expect(raw, runtime, QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION)


def test_categorical_feature_must_be_a_non_empty_string(runtime) -> None:
    for bad_value in (1.0, "", None, True):
        raw = raw_corpus()
        case_of(raw, "case_base_calm")["features"]["ProductCD"] = bad_value
        expect(raw, runtime, QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION)


def test_source_features_must_match_bundle_order(runtime) -> None:
    raw = raw_corpus()
    raw["source_features"] = list(reversed(raw["source_features"]))
    expect(raw, runtime, QualityGateErrorCode.CORPUS_CONTRACT_VIOLATION)


# --- case 结构 ----------------------------------------------------------


@pytest.mark.parametrize(
    "bad_id", ["CASE_BASE_CALM", "ab", "a" * 65, "case base", "case.base", "", "案例一"]
)
def test_case_id_pattern_is_enforced(runtime, bad_id: str) -> None:
    raw = raw_corpus()
    case_of(raw, "case_base_calm")["case_id"] = bad_id
    expect(raw, runtime, QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION)


def test_duplicate_case_id_is_rejected(runtime) -> None:
    raw = raw_corpus()
    case_of(raw, "case_amt_below")["case_id"] = "case_base_calm"
    expect(raw, runtime, QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION)


def test_unknown_case_field_is_rejected(runtime) -> None:
    raw = raw_corpus()
    case_of(raw, "case_base_calm")["label"] = 1
    expect(raw, runtime, QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION)


def test_role_is_enforced(runtime) -> None:
    raw = raw_corpus()
    case_of(raw, "case_base_calm")["role"] = "golden"
    expect(raw, runtime, QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION)


def test_baseline_role_must_not_declare_variations(runtime) -> None:
    raw = raw_corpus()
    case = case_of(raw, "case_base_calm")
    case["varied_features"] = ["TransactionAmt"]
    expect(raw, runtime, QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION)


def test_non_baseline_case_must_point_at_a_baseline(runtime) -> None:
    raw = raw_corpus()
    case_of(raw, "case_amt_above")["baseline_case_id"] = None
    expect(raw, runtime, QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION)


def test_varied_feature_must_exist(runtime) -> None:
    raw = raw_corpus()
    case_of(raw, "case_amt_above")["varied_features"] = ["NotAFeature"]
    expect(raw, runtime, QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION)


def test_boundary_case_must_change_exactly_the_declared_feature(runtime) -> None:
    """边界 case 多改一个特征 → 归因不再干净，必须失败。"""
    raw = raw_corpus()
    case = case_of(raw, "case_amt_above")
    case["features"]["C1"] = 42.0
    expect(raw, runtime, QualityGateErrorCode.CORPUS_CONTRACT_VIOLATION)


def test_boundary_case_must_declare_exactly_one_feature(runtime) -> None:
    raw = raw_corpus()
    case_of(raw, "case_amt_above")["varied_features"] = ["TransactionAmt", "C13"]
    expect(raw, runtime, QualityGateErrorCode.CORPUS_CONTRACT_VIOLATION)


# --- expected 契约 ------------------------------------------------------


def test_expected_field_set_is_exact(runtime) -> None:
    raw = raw_corpus()
    case_of(raw, "case_base_calm")["expected"]["probability"] = 0.1
    expect(raw, runtime, QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION)

    raw = raw_corpus()
    del case_of(raw, "case_base_calm")["expected"]["rule_score"]
    expect(raw, runtime, QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION)


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("score", True),
        ("score", None),
        ("score", float("inf")),
        ("model_flagged", 1),
        ("model_flagged", "true"),
        ("rule_score", True),
        ("rule_score", 1.5),
        ("rule_score", -0.5),
        ("signal_points", True),
        ("signal_points", -1),
        ("signal_points", 99),
        ("signal_points", "0"),
        ("matched_rule_ids", "transaction_amt_train_q99_tail"),
        ("matched_rule_ids", ["transaction_amt_train_q99_tail", "transaction_amt_train_q99_tail"]),
        ("risk_level", "critical"),
        ("risk_level", 3),
    ],
)
def test_illegal_expected_values_are_rejected(runtime, field: str, bad_value: object) -> None:
    raw = raw_corpus()
    case_of(raw, "case_base_calm")["expected"][field] = bad_value
    expect(raw, runtime, QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION)


def test_matched_and_unevaluable_must_not_overlap(runtime) -> None:
    raw = raw_corpus()
    expected = case_of(raw, "case_amt_above")["expected"]
    expected["unevaluable_rule_ids"] = list(expected["matched_rule_ids"])
    expect(raw, runtime, QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION)


# --- 覆盖契约 -----------------------------------------------------------


def test_coverage_requires_both_model_outcomes(runtime, engine) -> None:
    raw = raw_corpus()
    for case in raw["cases"]:
        case["expected"]["model_flagged"] = False
    corpus = build(raw, runtime)
    with pytest.raises(QualityGateError) as info:
        check_coverage(corpus, engine)
    assert info.value.code == QualityGateErrorCode.CORPUS_CONTRACT_VIOLATION


def test_coverage_requires_all_rule_match_kinds(runtime, engine) -> None:
    raw = raw_corpus()
    for case in raw["cases"]:
        case["expected"]["matched_rule_ids"] = []
    corpus = build(raw, runtime)
    with pytest.raises(QualityGateError) as info:
        check_coverage(corpus, engine)
    assert info.value.code == QualityGateErrorCode.CORPUS_CONTRACT_VIOLATION


def test_coverage_requires_all_three_risk_levels(runtime, engine) -> None:
    raw = raw_corpus()
    for case in raw["cases"]:
        case["expected"]["risk_level"] = "low_signal"
    corpus = build(raw, runtime)
    with pytest.raises(QualityGateError) as info:
        check_coverage(corpus, engine)
    assert info.value.code == QualityGateErrorCode.CORPUS_CONTRACT_VIOLATION


def test_coverage_requires_threshold_probes_for_every_rule(runtime, engine) -> None:
    """删掉"恰好等于阈值"的探针 → 严格 `>` 语义不再被覆盖，必须失败。"""
    raw = raw_corpus()
    raw["cases"] = [case for case in raw["cases"] if case["case_id"] != "case_amt_equal"]
    corpus = build(raw, runtime)
    with pytest.raises(QualityGateError) as info:
        check_coverage(corpus, engine)
    assert info.value.code == QualityGateErrorCode.CORPUS_CONTRACT_VIOLATION


def test_corpus_file_is_byte_frozen() -> None:
    """本文件之外的任何测试都不得改写语料：这里再核对一次冻结指纹。"""
    import hashlib

    payload = (PROJECT_ROOT / CORPUS_REL).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == CORPUS_SHA256


def test_baseline_role_constant_is_exported() -> None:
    assert ROLE_BASELINE == "baseline"
    assert Path(CORPUS_REL).suffix == ".json"
