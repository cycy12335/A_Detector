"""CW-009 一键门禁：冻结退出码、阶段顺序、失败关闭与报告安全。

每条用例都尽量走**真实**链路（真 bundle、真政策、真语料），只把 pytest 子进程换成注入
实现——因为这里要验证的是"阶段之间的编排与退出码映射"，不是 pytest 本身。

所有临时配置与语料副本都写在项目内被 gitignore 的 ``artifacts/tmp_tests/cw009/`` 下：
门禁的路径纪律只接受项目相对路径，测试因此也走同一条真实链路，不为测试开后门。
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from ai_detector.config import PROJECT_ROOT
from ai_detector.quality import gate as gate_module
from ai_detector.quality.errors import QualityGateError, QualityGateErrorCode
from ai_detector.quality.gate import (
    EXIT_OK,
    EXIT_PREFLIGHT,
    EXIT_PYTEST,
    EXIT_PYTEST_TIMEOUT,
    EXIT_REGRESSION,
    EXIT_REPORT,
    PytestOutcome,
    run_gate,
)

CONFIG_REL = "configs/quality_gate.yaml"
CORPUS_REL = "tests/fixtures/cw009/model_regression_cases.json"
STAGE_ROOT = PROJECT_ROOT / "artifacts" / "tmp_tests" / "cw009"

BUNDLE_REL = "artifacts/models/cw004/lightgbm_bundle.joblib"
POLICY_REL = "configs/risk_policy.yaml"
SERVICE_CONFIG_REL = "configs/service.yaml"


class PytestNotCalled(Exception):
    """注入的 pytest 实现不该被调用时抛出，避免掩盖阶段顺序错误。"""


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_dir(name: str) -> Path:
    directory = STAGE_ROOT / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def stage_config(name: str, mutate=None) -> tuple[str, Path]:
    """把真实配置复制到项目内临时目录，报告路径一并改到该目录下。

    Returns:
        ``(项目相对的配置路径, 报告绝对路径)``。
    """
    raw = copy.deepcopy(yaml.safe_load((PROJECT_ROOT / CONFIG_REL).read_text(encoding="utf-8")))
    section = raw["quality_gate"]
    section["report"]["path"] = f"artifacts/tmp_tests/cw009/{name}/report.json"
    if mutate is not None:
        mutate(section)
    directory = stage_dir(name)
    config_path = directory / "quality_gate.yaml"
    config_path.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return config_path.relative_to(PROJECT_ROOT).as_posix(), directory / "report.json"


def stage_corpus(name: str, mutate) -> tuple[str, str]:
    """把真实语料复制到项目内临时目录并篡改，返回 ``(相对路径, 篡改后摘要)``。"""
    raw = copy.deepcopy(json.loads((PROJECT_ROOT / CORPUS_REL).read_text(encoding="utf-8")))
    mutate(raw)
    directory = stage_dir(name)
    corpus_path = directory / "corpus.json"
    corpus_path.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return corpus_path.relative_to(PROJECT_ROOT).as_posix(), sha256_of(corpus_path)


def corpus_case(raw: dict, case_id: str) -> dict:
    for case in raw["cases"]:
        if case["case_id"] == case_id:
            return case
    raise AssertionError(case_id)


def config_text() -> str:
    return (PROJECT_ROOT / CONFIG_REL).read_text(encoding="utf-8")


def duplicate_line(text: str, needle: str) -> str:
    """把第一处与 ``needle`` 逐字相同的整行**原样复制**一份在它后面。"""
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.strip() == needle:
            indent = line[: len(line) - len(line.lstrip())]
            return "".join(lines[: index + 1]) + indent + needle + "\n" + "".join(lines[index + 1 :])
    raise AssertionError(f"配置里找不到整行 {needle!r}")


def stage_raw_config(name: str, text: str) -> tuple[str, Path]:
    """把一段**原始 YAML 文本**写成临时配置（不能用 safe_dump，重复键会被它吃掉）。"""
    directory = stage_dir(name)
    path = directory / "quality_gate.yaml"
    path.write_text(text, encoding="utf-8")
    return path.relative_to(PROJECT_ROOT).as_posix(), directory / "report.json"


def config_text_pointing_at(name: str) -> str:
    """真实配置文本，但报告路径改到临时目录（避免污染真实报告）。"""
    return config_text().replace(
        "    path: artifacts/reports/cw009_quality_gate/report.json",
        f"    path: artifacts/tmp_tests/cw009/{name}/report.json",
        1,
    )


def staged_report_rel(name: str) -> str:
    return f"artifacts/tmp_tests/cw009/{name}/report.json"


def passing_runner(calls: list[dict]):
    def runner(command, *, cwd, timeout_seconds):
        calls.append(
            {"command": list(command), "cwd": cwd, "timeout_seconds": timeout_seconds}
        )
        return PytestOutcome(
            status="passed",
            exit_code=0,
            duration_seconds=42.0,
            passed=1343,
            skipped=3,
            failed=0,
            errors=0,
        )

    return runner


def failing_runner(command, *, cwd, timeout_seconds):
    return PytestOutcome(
        status="failed", exit_code=1, duration_seconds=7.0, passed=10, skipped=0, failed=2, errors=0
    )


def timing_out_runner(command, *, cwd, timeout_seconds):
    return PytestOutcome(
        status="timeout",
        exit_code=None,
        duration_seconds=1800.0,
        passed=0,
        skipped=0,
        failed=0,
        errors=0,
        child_reaped=True,
    )


def spawn_failing_runner(command, *, cwd, timeout_seconds):
    return PytestOutcome(
        status="spawn_failed",
        exit_code=None,
        duration_seconds=0.01,
        passed=0,
        skipped=0,
        failed=0,
        errors=0,
    )


def forbidden_runner(*args, **kwargs):
    raise PytestNotCalled("pytest 阶段不得被启动")


# --- 成功路径 -----------------------------------------------------------


def test_regression_only_exits_zero_and_writes_safe_report() -> None:
    config_path, report_path = stage_config("ok_regression_only")
    outcome = run_gate(config_path, regression_only=True)

    assert outcome.exit_code == EXIT_OK
    assert outcome.report_written is True
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["overall_status"] == "passed"
    assert report["work_package"] == "CW-009"
    assert report["regression"]["cases_total"] == 12
    assert report["regression"]["cases_failed"] == 0
    assert report["pytest"]["status"] == "skipped_by_regression_only"
    assert [stage["name"] for stage in report["stages"]] == ["preflight", "regression", "pytest"]


def test_report_records_the_four_actual_hashes() -> None:
    config_path, report_path = stage_config("ok_hashes")
    assert run_gate(config_path, regression_only=True).exit_code == EXIT_OK
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert set(report["inputs"]) == {"service_config", "bundle", "policy", "corpus"}
    for label, relative in (
        ("service_config", SERVICE_CONFIG_REL),
        ("bundle", BUNDLE_REL),
        ("policy", POLICY_REL),
        ("corpus", CORPUS_REL),
    ):
        entry = report["inputs"][label]
        assert entry["path"] == relative
        assert entry["match"] is True
        assert entry["actual_sha256"] == sha256_of(PROJECT_ROOT / relative)


def test_default_command_runs_pytest_with_frozen_shape() -> None:
    calls: list[dict] = []
    config_path, report_path = stage_config("ok_pytest")
    outcome = run_gate(config_path, pytest_runner=passing_runner(calls))

    assert outcome.exit_code == EXIT_OK
    assert len(calls) == 1, "默认命令必须**恰好**运行一次完整 pytest"
    import sys

    assert calls[0]["command"] == [sys.executable, "-m", "pytest"]
    assert calls[0]["cwd"] == PROJECT_ROOT
    assert calls[0]["timeout_seconds"] == 1800.0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["pytest"]["status"] == "passed"
    assert report["pytest"]["counts"] == {"passed": 1343, "skipped": 3, "failed": 0, "errors": 0}
    assert report["pytest"]["command"] == {
        "program": "<sys.executable>",
        "arguments": ["-m", "pytest"],
        "cwd": "<project_root>",
        "shell": False,
        "timeout_seconds": 1800.0,
    }


# --- pytest 阶段失败映射 ------------------------------------------------


def test_pytest_failure_exits_four() -> None:
    config_path, report_path = stage_config("pytest_failed")
    outcome = run_gate(config_path, pytest_runner=failing_runner)

    assert outcome.exit_code == EXIT_PYTEST
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["error_code"] == QualityGateErrorCode.PYTEST_FAILED
    assert report["pytest"]["exit_code"] == 1


def test_pytest_timeout_exits_five() -> None:
    config_path, report_path = stage_config("pytest_timeout")
    outcome = run_gate(config_path, pytest_runner=timing_out_runner)

    assert outcome.exit_code == EXIT_PYTEST_TIMEOUT
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["error_code"] == QualityGateErrorCode.PYTEST_TIMEOUT
    assert report["pytest"]["child_reaped"] is True


def test_pytest_spawn_failure_exits_four() -> None:
    config_path, report_path = stage_config("pytest_spawn")
    outcome = run_gate(config_path, pytest_runner=spawn_failing_runner)

    assert outcome.exit_code == EXIT_PYTEST
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["error_code"] == QualityGateErrorCode.PYTEST_SPAWN_FAILED


def test_injected_pytest_exception_fails_closed() -> None:
    def exploding(command, *, cwd, timeout_seconds):
        raise RuntimeError("injected")

    config_path, report_path = stage_config("pytest_exploding")
    outcome = run_gate(config_path, pytest_runner=exploding)

    assert outcome.exit_code == EXIT_PYTEST
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["error_code"] == QualityGateErrorCode.INTERNAL_ERROR
    assert "injected" not in report_path.read_text(encoding="utf-8")


# --- 预检失败：不得加载模型、不得跑 pytest ------------------------------


@pytest.mark.parametrize(
    "label, relative",
    [
        ("bundle", BUNDLE_REL),
        ("policy", POLICY_REL),
        ("corpus", CORPUS_REL),
        ("service_config", SERVICE_CONFIG_REL),
    ],
)
def test_input_hash_tamper_exits_two_without_running_pytest(label: str, relative: str) -> None:
    def mutate(section, label=label):
        section["inputs"][label]["sha256"] = "0" * 64

    config_path, report_path = stage_config(f"tamper_{label}", mutate)
    outcome = run_gate(config_path, pytest_runner=forbidden_runner)

    assert outcome.exit_code == EXIT_PREFLIGHT
    assert outcome.report_written is True
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["error_code"] == QualityGateErrorCode.INPUT_HASH_MISMATCH
    assert report["inputs"][label]["match"] is False
    assert report["regression"] == {}
    assert report["pytest"] == {"status": "not_run"}


FOUR_INPUTS = (
    ("service_config", SERVICE_CONFIG_REL),
    ("bundle", BUNDLE_REL),
    ("policy", POLICY_REL),
    ("corpus", CORPUS_REL),
)


def test_hash_mismatch_report_records_every_existing_input_actual() -> None:
    """指纹不符时，报告要写出**全部四个存在输入**的实际摘要与 match——不只是失败那个。"""

    def mutate(section):
        section["inputs"]["corpus"]["sha256"] = "0" * 63 + "1"

    config_path, report_path = stage_config("mismatch_actuals", mutate)
    outcome = run_gate(config_path, pytest_runner=forbidden_runner)

    assert outcome.exit_code == EXIT_PREFLIGHT
    assert outcome.report_written is True
    report = json.loads(report_path.read_text(encoding="utf-8"))
    inputs = report["inputs"]

    assert set(inputs) == {label for label, _ in FOUR_INPUTS}
    for label, relative in FOUR_INPUTS:
        assert inputs[label]["actual_sha256"] == sha256_of(PROJECT_ROOT / relative), label
        assert inputs[label]["path"] == relative
    assert inputs["corpus"]["match"] is False
    assert [inputs[label]["match"] for label, _ in FOUR_INPUTS] == [True, True, True, False]


def test_multiple_hash_mismatches_are_all_recorded() -> None:
    """多个输入同时不符时也不能短路：四个实际摘要都要落进报告。"""

    def mutate(section):
        section["inputs"]["bundle"]["sha256"] = "0" * 64
        section["inputs"]["policy"]["sha256"] = "1" * 64

    config_path, report_path = stage_config("mismatch_multi", mutate)
    outcome = run_gate(config_path, pytest_runner=forbidden_runner)

    assert outcome.exit_code == EXIT_PREFLIGHT
    report = json.loads(report_path.read_text(encoding="utf-8"))
    inputs = report["inputs"]
    assert inputs["bundle"]["match"] is False
    assert inputs["policy"]["match"] is False
    assert inputs["service_config"]["match"] is True
    assert inputs["corpus"]["match"] is True
    assert all(entry["actual_sha256"] for entry in inputs.values())


def test_missing_input_records_null_only_for_the_missing_file() -> None:
    """文件确实缺失时该条为 null，但其它存在输入的摘要仍要记录。"""

    def mutate(section):
        section["inputs"]["policy"]["path"] = "configs/missing_cw009_policy.yaml"

    config_path, report_path = stage_config("missing_actuals", mutate)
    outcome = run_gate(config_path, pytest_runner=forbidden_runner)

    assert outcome.exit_code == EXIT_PREFLIGHT
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["error_code"] == QualityGateErrorCode.INPUT_MISSING
    inputs = report["inputs"]
    assert inputs["policy"]["actual_sha256"] is None
    assert inputs["policy"]["match"] is False
    for label, relative in (("service_config", SERVICE_CONFIG_REL), ("bundle", BUNDLE_REL), ("corpus", CORPUS_REL)):
        assert inputs[label]["match"] is True
        assert inputs[label]["actual_sha256"] == sha256_of(PROJECT_ROOT / relative)


def test_all_four_digests_are_collected_before_any_semantic_parsing(monkeypatch) -> None:
    """预检必须先**收齐**四个二进制摘要，再做语义解析/反序列化/加载。

    正对照保证这个探针不是空转：指纹全对时四个下游步骤都会被调用。
    """
    calls: list[str] = []
    real_service_config = gate_module.ServiceConfig
    real = {
        "load_runtime": gate_module.load_runtime,
        "load_policy_engine": gate_module.load_policy_engine,
        "load_corpus": gate_module.load_corpus,
    }

    class RecordingServiceConfig:
        @staticmethod
        def from_yaml(path):
            calls.append("ServiceConfig.from_yaml")
            return real_service_config.from_yaml(path)

    def recording(attribute: str):
        def wrapper(*args, **kwargs):
            calls.append(attribute)
            return real[attribute](*args, **kwargs)

        return wrapper

    monkeypatch.setattr(gate_module, "ServiceConfig", RecordingServiceConfig)
    for attribute in real:
        monkeypatch.setattr(gate_module, attribute, recording(attribute))

    ok_config, _ = stage_config("collect_ok")
    assert run_gate(ok_config, regression_only=True).exit_code == EXIT_OK
    assert calls == [
        "ServiceConfig.from_yaml",
        "load_runtime",
        "load_policy_engine",
        "load_corpus",
    ]

    calls.clear()

    def mutate(section):
        section["inputs"]["bundle"]["sha256"] = "0" * 64

    bad_config, report_path = stage_config("collect_bad", mutate)
    outcome = run_gate(bad_config, regression_only=True)

    assert outcome.exit_code == EXIT_PREFLIGHT
    assert calls == [], "指纹不符时不得做任何语义解析、反序列化或加载"
    assert outcome.report_written is True
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert all(entry["actual_sha256"] for entry in report["inputs"].values())


def test_duplicate_config_key_exits_two() -> None:
    """相同值的重复 YAML 键必须 exit 2，且不得进入任何后续阶段。"""
    text = duplicate_line(config_text_pointing_at("dup_config_key"), "work_package: CW-009")
    config_path, report_path = stage_raw_config("dup_config_key", text)

    outcome = run_gate(
        config_path,
        pytest_runner=forbidden_runner,
        fallback_report_path=staged_report_rel("dup_config_key"),
    )

    assert outcome.exit_code == EXIT_PREFLIGHT
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["error_code"] == QualityGateErrorCode.CONFIG_SCHEMA_VIOLATION
    assert report["inputs"] == {}
    assert report["regression"] == {}
    assert report["pytest"] == {"status": "not_run"}


@pytest.mark.parametrize(
    "needle",
    [
        "  schema_version: cw009_quality_gate_config_v1",
        "  score_absolute_tolerance: 1.0e-12",
        "  timeout_seconds: 1800.0",
        '  arguments: ["-m", "pytest"]',
    ],
)
def test_duplicate_config_key_at_nested_levels_exits_two(needle: str) -> None:
    name = "dup_nested"
    text = duplicate_line(config_text_pointing_at(name), needle.strip())
    config_path, report_path = stage_raw_config(name, text)

    outcome = run_gate(
        config_path,
        pytest_runner=forbidden_runner,
        fallback_report_path=staged_report_rel(name),
    )

    assert outcome.exit_code == EXIT_PREFLIGHT
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["error_code"] == QualityGateErrorCode.CONFIG_SCHEMA_VIOLATION


def test_conflicting_duplicate_config_key_is_not_resolved_by_last_wins() -> None:
    """冲突值的重复键不能被"后者覆盖前者"悄悄定案。"""
    name = "dup_conflict"
    text = duplicate_line(config_text_pointing_at(name), "expected_source_features: 199").replace(
        "expected_source_features: 199", "expected_source_features: 1", 1
    )
    config_path, report_path = stage_raw_config(name, text)

    outcome = run_gate(
        config_path,
        pytest_runner=forbidden_runner,
        fallback_report_path=staged_report_rel(name),
    )

    assert outcome.exit_code == EXIT_PREFLIGHT
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["error_code"] == QualityGateErrorCode.CONFIG_SCHEMA_VIOLATION


def test_missing_input_exits_two() -> None:
    def mutate(section):
        section["inputs"]["corpus"]["path"] = "configs/missing_cw009_corpus.json"

    config_path, _ = stage_config("missing_input", mutate)
    outcome = run_gate(config_path, pytest_runner=forbidden_runner)

    assert outcome.exit_code == EXIT_PREFLIGHT
    assert outcome.report["error_code"] == QualityGateErrorCode.INPUT_MISSING


def test_config_schema_violation_exits_two() -> None:
    def mutate(section):
        section["comparison"]["score_absolute_tolerance"] = 0.25

    config_path, _ = stage_config("loose_tolerance", mutate)
    outcome = run_gate(config_path, pytest_runner=forbidden_runner)

    assert outcome.exit_code == EXIT_PREFLIGHT
    assert outcome.report["error_code"] == QualityGateErrorCode.CONFIG_SCHEMA_VIOLATION
    assert outcome.report["inputs"] == {}


def test_unknown_config_key_exits_two() -> None:
    def mutate(section):
        section["update_oracle"] = True

    config_path, _ = stage_config("unknown_key", mutate)
    outcome = run_gate(config_path, pytest_runner=forbidden_runner)

    assert outcome.exit_code == EXIT_PREFLIGHT
    assert outcome.report["error_code"] == QualityGateErrorCode.CONFIG_SCHEMA_VIOLATION


def test_unexpected_internal_error_fails_closed(monkeypatch) -> None:
    def exploding(config, report):
        raise RuntimeError("boom")

    monkeypatch.setattr(gate_module, "preflight", exploding)
    config_path, report_path = stage_config("internal_error")
    outcome = run_gate(config_path, pytest_runner=forbidden_runner)

    assert outcome.exit_code == EXIT_PREFLIGHT
    text = report_path.read_text(encoding="utf-8")
    report = json.loads(text)
    assert report["error_code"] == QualityGateErrorCode.INTERNAL_ERROR
    assert "Traceback" not in text
    assert "boom" not in text


def test_missing_config_uses_fallback_report_path() -> None:
    fallback = "artifacts/tmp_tests/cw009/fallback_missing_config/report.json"
    outcome = run_gate(
        "configs/definitely_missing_cw009.yaml",
        pytest_runner=forbidden_runner,
        fallback_report_path=fallback,
    )

    assert outcome.exit_code == EXIT_PREFLIGHT
    assert outcome.report["error_code"] == QualityGateErrorCode.CONFIG_UNREADABLE
    assert outcome.report_path == fallback
    assert (PROJECT_ROOT / fallback).is_file()


# --- 回归失败 -----------------------------------------------------------


def test_tampered_expected_score_exits_three() -> None:
    corpus_rel, corpus_sha = stage_corpus(
        "regression_score",
        lambda raw: corpus_case(raw, "case_stress_flagged").__setitem__(
            "expected",
            {**corpus_case(raw, "case_stress_flagged")["expected"], "score": 0.5},
        ),
    )

    def mutate(section):
        section["inputs"]["corpus"]["path"] = corpus_rel
        section["inputs"]["corpus"]["sha256"] = corpus_sha

    config_path, report_path = stage_config("regression_score", mutate)
    outcome = run_gate(config_path, pytest_runner=forbidden_runner)

    assert outcome.exit_code == EXIT_REGRESSION
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["error_code"] == QualityGateErrorCode.REGRESSION_MISMATCH
    assert report["regression"]["failed_case_ids"] == ["case_stress_flagged"]
    assert report["regression"]["failed_fields"] == {"case_stress_flagged": ["score"]}
    assert report["pytest"] == {"status": "not_run"}


def test_tampered_feature_value_exits_three() -> None:
    """只改一个特征取值（语料哈希同步更新）→ 分数/派生事实漂移 → 退出 3。"""

    def tamper(raw):
        # case_stress_flagged 是 stress 角色，改它的施压特征不会破坏边界配对。
        corpus_case(raw, "case_stress_flagged")["features"]["C1"] = 0.0

    corpus_rel, corpus_sha = stage_corpus("regression_feature", tamper)

    def mutate(section):
        section["inputs"]["corpus"]["path"] = corpus_rel
        section["inputs"]["corpus"]["sha256"] = corpus_sha

    config_path, report_path = stage_config("regression_feature", mutate)
    outcome = run_gate(config_path, pytest_runner=forbidden_runner)

    assert outcome.exit_code == EXIT_REGRESSION
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["regression"]["failed_case_ids"] == ["case_stress_flagged"]
    assert report["regression"]["failed_fields"]["case_stress_flagged"] == [
        "score",
        "model_flagged",
        "signal_points",
        "risk_level",
    ]
    # 报告里不得出现任何源特征名、类别取值或修改后的特征取值。
    text = report_path.read_text(encoding="utf-8")
    for token in ("TransactionAmt", "C13", "C1\"", "cw009_cat_alpha", "cw009_cat_beta"):
        assert token not in text


def test_corpus_feature_reorder_exits_two() -> None:
    def tamper(raw):
        for case in raw["cases"]:
            items = list(case["features"].items())
            items[0], items[1] = items[1], items[0]
            case["features"] = dict(items)

    corpus_rel, corpus_sha = stage_corpus("corpus_reorder", tamper)

    def mutate(section):
        section["inputs"]["corpus"]["path"] = corpus_rel
        section["inputs"]["corpus"]["sha256"] = corpus_sha

    config_path, _ = stage_config("corpus_reorder", mutate)
    outcome = run_gate(config_path, pytest_runner=forbidden_runner)

    assert outcome.exit_code == EXIT_PREFLIGHT
    assert outcome.report["error_code"] == QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION


def test_tolerance_mismatch_between_config_and_corpus_exits_two() -> None:
    corpus_rel, corpus_sha = stage_corpus(
        "tolerance_mismatch", lambda raw: raw.__setitem__("score_absolute_tolerance", 1e-9)
    )

    def mutate(section):
        section["inputs"]["corpus"]["path"] = corpus_rel
        section["inputs"]["corpus"]["sha256"] = corpus_sha

    config_path, _ = stage_config("tolerance_mismatch", mutate)
    outcome = run_gate(config_path, pytest_runner=forbidden_runner)

    assert outcome.exit_code == EXIT_PREFLIGHT
    assert outcome.report["error_code"] == QualityGateErrorCode.AGREEMENT_VIOLATION


# --- 报告失败 -----------------------------------------------------------


def test_report_write_failure_exits_six(monkeypatch) -> None:
    def exploding(*args, **kwargs):
        raise QualityGateError(QualityGateErrorCode.REPORT_WRITE_FAILED, "injected")

    monkeypatch.setattr(gate_module, "write_report", exploding)
    config_path, _ = stage_config("report_write_failure")
    outcome = run_gate(config_path, regression_only=True)

    assert outcome.exit_code == EXIT_REPORT
    assert outcome.report is None
    assert outcome.report_written is False


def test_unsafe_report_exits_six(monkeypatch) -> None:
    def exploding(*args, **kwargs):
        raise QualityGateError(QualityGateErrorCode.REPORT_UNSAFE, "injected")

    monkeypatch.setattr(gate_module, "write_report", exploding)
    config_path, _ = stage_config("report_unsafe")
    outcome = run_gate(config_path, regression_only=True)

    assert outcome.exit_code == EXIT_REPORT
    assert outcome.report_written is False


# --- 只读性 -------------------------------------------------------------


def test_gate_runs_do_not_modify_frozen_inputs() -> None:
    """门禁是只读的：跑完之后四个**已提交**输入的摘要必须一字不变。

    默认决策数据库不在此列：它是运行时产物、被 ``.gitignore`` 排除，干净检出里本来
    就不存在（CW-014 已把"默认库不被改写、也不被凭空创建"交给根 ``conftest.py`` 的
    会话级守卫 ``guard_default_database``，在有无该文件的两种环境下都成立）。
    """
    before = {
        relative: sha256_of(PROJECT_ROOT / relative)
        for relative in (
            SERVICE_CONFIG_REL,
            BUNDLE_REL,
            POLICY_REL,
            CORPUS_REL,
        )
    }
    config_path, _ = stage_config("readonly")
    assert run_gate(config_path, regression_only=True).exit_code == EXIT_OK
    after = {relative: sha256_of(PROJECT_ROOT / relative) for relative in before}
    assert after == before
