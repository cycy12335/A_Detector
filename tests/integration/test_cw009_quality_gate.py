"""CW-009 端到端：真实默认配置 + 全部篡改反例 + 冻结产物一字未变。

这些用例走**完整真实链路**：真 bundle、真政策、真语料、真服务配置。篡改只发生在
``artifacts/tmp_tests/cw009/`` 下的临时副本上，仓库里的冻结文件只读不改——最后一条用例
会逐一核对它们的指纹。

刻意**不**在测试里启动默认门禁（那会在 pytest 内部再跑一次完整 pytest）。默认命令的
真实执行属于人工/验收验证，不在这里递归。
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest
import yaml

from ai_detector.config import PROJECT_ROOT
from ai_detector.quality.errors import QualityGateErrorCode
from ai_detector.quality.gate import (
    EXIT_OK,
    EXIT_PREFLIGHT,
    EXIT_REGRESSION,
    run_gate,
)

CONFIG_REL = "configs/quality_gate.yaml"
CORPUS_REL = "tests/fixtures/cw009/model_regression_cases.json"
SERVICE_CONFIG_REL = "configs/service.yaml"
BUNDLE_REL = "artifacts/models/cw004/lightgbm_bundle.joblib"
POLICY_REL = "configs/risk_policy.yaml"
DEFAULT_REPORT_REL = "artifacts/reports/cw009_quality_gate/report.json"
STAGE_ROOT = PROJECT_ROOT / "artifacts" / "tmp_tests" / "cw009_integration"


def sha256_of(relative: str) -> str:
    return hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest()


def stage_dir(name: str) -> Path:
    directory = STAGE_ROOT / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def raw_config() -> dict:
    return copy.deepcopy(yaml.safe_load((PROJECT_ROOT / CONFIG_REL).read_text(encoding="utf-8")))


def raw_corpus() -> dict:
    return copy.deepcopy(json.loads((PROJECT_ROOT / CORPUS_REL).read_text(encoding="utf-8")))


def corpus_case(raw: dict, case_id: str) -> dict:
    for case in raw["cases"]:
        if case["case_id"] == case_id:
            return case
    raise AssertionError(case_id)


def write_config(name: str, mutate: Callable[[dict], None] | None = None) -> tuple[str, Path]:
    raw = raw_config()
    section = raw["quality_gate"]
    section["report"]["path"] = f"artifacts/tmp_tests/cw009_integration/{name}/report.json"
    if mutate is not None:
        mutate(section)
    path = stage_dir(name) / "quality_gate.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path.relative_to(PROJECT_ROOT).as_posix(), stage_dir(name) / "report.json"


def write_corpus(name: str, mutate: Callable[[dict], None]) -> tuple[str, str]:
    raw = raw_corpus()
    mutate(raw)
    path = stage_dir(name) / "corpus.json"
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path.relative_to(PROJECT_ROOT).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest()


def point_at_staged_corpus(corpus_rel: str, corpus_sha: str):
    def mutate(section: dict) -> None:
        section["inputs"]["corpus"]["path"] = corpus_rel
        section["inputs"]["corpus"]["sha256"] = corpus_sha

    return mutate


def config_text() -> str:
    return (PROJECT_ROOT / CONFIG_REL).read_text(encoding="utf-8")


def corpus_text() -> str:
    return (PROJECT_ROOT / CORPUS_REL).read_text(encoding="utf-8")


def duplicate_line(text: str, needle: str) -> str:
    """把第一处与 ``needle`` 逐字相同的整行**原样复制**一份在它后面。

    原样复制的是"重复键取值相同"这一最容易被解析器静默接受的形态。
    """
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.strip() == needle:
            indent = line[: len(line) - len(line.lstrip())]
            return "".join(lines[: index + 1]) + indent + needle + "\n" + "".join(lines[index + 1 :])
    raise AssertionError(f"找不到整行 {needle!r}")


def config_text_pointing_at(name: str) -> str:
    return config_text().replace(
        "    path: artifacts/reports/cw009_quality_gate/report.json",
        f"    path: artifacts/tmp_tests/cw009_integration/{name}/report.json",
        1,
    )


def write_raw_config(name: str, text: str) -> tuple[str, Path]:
    path = stage_dir(name) / "quality_gate.yaml"
    path.write_text(text, encoding="utf-8")
    return path.relative_to(PROJECT_ROOT).as_posix(), stage_dir(name) / "report.json"


def write_corpus_text(name: str, text: str) -> tuple[str, str]:
    path = stage_dir(name) / "corpus.json"
    path.write_text(text, encoding="utf-8")
    return path.relative_to(PROJECT_ROOT).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest()


def report_rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def staged_report_rel(name: str) -> str:
    return f"artifacts/tmp_tests/cw009_integration/{name}/report.json"


# --- 默认配置的真实链路 -------------------------------------------------


def test_default_config_passes_preflight_and_regression() -> None:
    """默认配置本身必须通过预检与回归（只跳过 pytest，避免递归启动门禁）。"""
    outcome = run_gate(regression_only=True)

    assert outcome.exit_code == EXIT_OK
    assert outcome.report_path == DEFAULT_REPORT_REL
    report = outcome.report
    assert report["overall_status"] == "passed"
    assert report["regression"]["cases_passed"] == 12
    assert report["regression"]["deterministic"] is True
    assert report["regression"]["max_score_absolute_deviation"] == 0.0
    assert report["coverage"]["model_flagged_true"] >= 1
    assert report["coverage"]["model_flagged_false"] >= 1
    assert report["coverage"]["risk_levels_present"] == [
        "low_signal",
        "elevated_signal",
        "high_signal",
    ]
    assert report["inputs"]["corpus"]["actual_sha256"] == sha256_of(CORPUS_REL)


def test_written_report_is_safe() -> None:
    assert run_gate(regression_only=True).exit_code == EXIT_OK
    text = (PROJECT_ROOT / DEFAULT_REPORT_REL).read_text(encoding="utf-8")
    report = json.loads(text)

    for token in (
        "TransactionAmt",
        "ProductCD",
        "cw009_cat_alpha",
        "cw009_cat_beta",
        "D:\\",
        "D:/",
        "C:\\",
        "/home/",
        "site-packages",
    ):
        assert token not in text, f"报告不得包含 {token!r}"
    assert "traceback" not in text.lower()
    assert report["pytest"]["status"] in {"not_run", "skipped_by_regression_only", "passed"}


def test_cli_entrypoint_runs_and_exits_zero() -> None:
    """脚本入口（真实子进程）能够跑通并返回 0，且控制台不打印绝对路径。"""
    config_path, _ = write_config("cli_entry")
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_quality_gate.py",
            "--config",
            config_path,
            "--regression-only",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=300,
        check=False,
    )

    assert completed.returncode == EXIT_OK, completed.stderr
    assert "preflight" in completed.stdout
    assert "regression" in completed.stdout
    for token in ("D:\\", "D:/", "artifacts\\tmp_tests"):
        assert token not in completed.stdout
    assert "D:\\" not in completed.stderr


def test_cli_rejects_unknown_flags() -> None:
    """命令行不存在"跳过哈希 / 放宽容差 / 更新 oracle"之类的开关。"""
    for flag in ("--update-oracle", "--bless", "--skip-hash", "--tolerance=1"):
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_quality_gate.py",
                "--config",
                CONFIG_REL,
                "--regression-only",
                flag,
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            timeout=300,
            check=False,
        )
        assert completed.returncode != EXIT_OK
        assert "unrecognized arguments" in completed.stderr or "invalid" in completed.stderr


# --- 篡改反例 -----------------------------------------------------------


@dataclass(frozen=True)
class Tamper:
    name: str
    corpus_mutate: Callable[[dict], None] | None
    config_mutate: Callable[[dict], None] | None
    exit_code: int
    error_code: str


def _swap_expected(case_id: str, **changes):
    def mutate(raw: dict) -> None:
        case = corpus_case(raw, case_id)
        case["expected"] = {**case["expected"], **changes}

    return mutate


def _rename_feature(raw: dict) -> None:
    case = corpus_case(raw, "case_base_calm")
    renamed = {}
    for index, (name, value) in enumerate(case["features"].items()):
        renamed[f"Renamed{index}" if index == 0 else name] = value
    case["features"] = renamed


def _reorder_features(raw: dict) -> None:
    for case in raw["cases"]:
        items = list(case["features"].items())
        items[0], items[1] = items[1], items[0]
        case["features"] = dict(items)


def _bad_case_id(raw: dict) -> None:
    corpus_case(raw, "case_base_calm")["case_id"] = "Case-Base-Calm"


def _drop_a_case(raw: dict) -> None:
    raw["cases"] = [case for case in raw["cases"] if case["case_id"] != "case_amt_equal"]


TAMPERS: tuple[Tamper, ...] = (
    Tamper(
        "expected_score",
        _swap_expected("case_stress_flagged", score=0.5),
        None,
        EXIT_REGRESSION,
        QualityGateErrorCode.REGRESSION_MISMATCH,
    ),
    Tamper(
        "expected_flag",
        _swap_expected("case_stress_flagged", model_flagged=False),
        None,
        EXIT_REGRESSION,
        QualityGateErrorCode.REGRESSION_MISMATCH,
    ),
    Tamper(
        "expected_risk_level",
        _swap_expected("case_stress_amt_above", risk_level="low_signal"),
        None,
        EXIT_REGRESSION,
        QualityGateErrorCode.REGRESSION_MISMATCH,
    ),
    Tamper(
        "expected_rule_ids",
        _swap_expected("case_amt_above", matched_rule_ids=[]),
        None,
        EXIT_REGRESSION,
        QualityGateErrorCode.REGRESSION_MISMATCH,
    ),
    Tamper(
        "expected_signal_points",
        _swap_expected("case_stress_amt_above", signal_points=4),
        None,
        EXIT_REGRESSION,
        QualityGateErrorCode.REGRESSION_MISMATCH,
    ),
    Tamper(
        "expected_rule_score",
        _swap_expected("case_c13_above", rule_score=1.0),
        None,
        EXIT_REGRESSION,
        QualityGateErrorCode.REGRESSION_MISMATCH,
    ),
    Tamper(
        "expected_unevaluable",
        _swap_expected("case_base_calm", unevaluable_rule_ids=["c13_train_q99_tail"]),
        None,
        EXIT_REGRESSION,
        QualityGateErrorCode.REGRESSION_MISMATCH,
    ),
    Tamper(
        "feature_value",
        lambda raw: corpus_case(raw, "case_stress_flagged")["features"].__setitem__("C1", 0.0),
        None,
        EXIT_REGRESSION,
        QualityGateErrorCode.REGRESSION_MISMATCH,
    ),
    Tamper(
        "feature_rename",
        _rename_feature,
        None,
        EXIT_PREFLIGHT,
        QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION,
    ),
    Tamper(
        "feature_order",
        _reorder_features,
        None,
        EXIT_PREFLIGHT,
        QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION,
    ),
    Tamper(
        "case_id",
        _bad_case_id,
        None,
        EXIT_PREFLIGHT,
        QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION,
    ),
    Tamper(
        "dropped_boundary_probe",
        _drop_a_case,
        None,
        EXIT_PREFLIGHT,
        QualityGateErrorCode.AGREEMENT_VIOLATION,
    ),
    Tamper("corpus_hash", None, lambda s: s["inputs"]["corpus"].__setitem__("sha256", "0" * 64),
           EXIT_PREFLIGHT, QualityGateErrorCode.INPUT_HASH_MISMATCH),
    Tamper("bundle_hash", None, lambda s: s["inputs"]["bundle"].__setitem__("sha256", "0" * 64),
           EXIT_PREFLIGHT, QualityGateErrorCode.INPUT_HASH_MISMATCH),
    Tamper("policy_hash", None, lambda s: s["inputs"]["policy"].__setitem__("sha256", "0" * 64),
           EXIT_PREFLIGHT, QualityGateErrorCode.INPUT_HASH_MISMATCH),
    Tamper(
        "service_config_hash",
        None,
        lambda s: s["inputs"]["service_config"].__setitem__("sha256", "0" * 64),
        EXIT_PREFLIGHT,
        QualityGateErrorCode.INPUT_HASH_MISMATCH,
    ),
    Tamper(
        "loose_tolerance",
        None,
        lambda s: s["comparison"].__setitem__("score_absolute_tolerance", 0.5),
        EXIT_PREFLIGHT,
        QualityGateErrorCode.CONFIG_SCHEMA_VIOLATION,
    ),
    Tamper(
        "pytest_selector",
        None,
        lambda s: s["pytest"].__setitem__("arguments", ["-m", "pytest", "-k", "cw005"]),
        EXIT_PREFLIGHT,
        QualityGateErrorCode.CONFIG_SCHEMA_VIOLATION,
    ),
)


@pytest.mark.parametrize("tamper", TAMPERS, ids=[item.name for item in TAMPERS])
def test_tamper_counterexample_fails_closed(tamper: Tamper) -> None:
    config_mutate = tamper.config_mutate
    if tamper.corpus_mutate is not None:
        corpus_rel, corpus_sha = write_corpus(f"tamper_{tamper.name}", tamper.corpus_mutate)
        staged = point_at_staged_corpus(corpus_rel, corpus_sha)
        if config_mutate is None:
            config_mutate = staged
        else:
            base = config_mutate

            def config_mutate(section: dict, base=base, staged=staged) -> None:
                base(section)
                staged(section)

    config_path, report_path = write_config(f"tamper_{tamper.name}", config_mutate)
    # 配置本身解析失败时门禁不知道报告该写哪里，只会退回默认路径；测试显式把回退路径
    # 也指到临时目录，避免污染真实报告。
    outcome = run_gate(
        config_path,
        regression_only=True,
        fallback_report_path=report_path.relative_to(PROJECT_ROOT).as_posix(),
    )

    assert outcome.exit_code == tamper.exit_code, (tamper.name, outcome.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["error_code"] == tamper.error_code
    assert report["overall_status"] == "failed"
    # 除了"报告写入失败"，任何失败都必须留下一份可定位的报告。
    assert report_path.is_file()


# --- 重复映射键：端到端 -------------------------------------------------


def test_duplicate_config_key_exits_two_end_to_end() -> None:
    """相同值的重复 YAML 键：exit 2 `config_schema_violation`，且后续阶段一步都没跑。"""
    name = "dup_config_key"
    text = duplicate_line(config_text_pointing_at(name), "work_package: CW-009")
    config_path, report_path = write_raw_config(name, text)

    outcome = run_gate(
        config_path, regression_only=True, fallback_report_path=staged_report_rel(name)
    )

    assert outcome.exit_code == EXIT_PREFLIGHT
    assert outcome.report_written is True
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["error_code"] == QualityGateErrorCode.CONFIG_SCHEMA_VIOLATION
    assert report["overall_status"] == "failed"
    assert report["inputs"] == {}
    assert report["regression"] == {}
    assert report["pytest"] == {"status": "not_run"}


@pytest.mark.parametrize(
    "needle",
    [
        "schema_version: cw009_quality_gate_config_v1",
        "service_config:",
        "model_version: cw004-lightgbm-challenger-v1",
        "score_absolute_tolerance: 1.0e-12",
        "expected_source_features: 199",
        "timeout_seconds: 1800.0",
        'arguments: ["-m", "pytest"]',
    ],
)
def test_duplicate_config_key_at_any_level_exits_two_end_to_end(needle: str) -> None:
    name = "dup_config_nested"
    text = duplicate_line(config_text_pointing_at(name), needle)
    config_path, report_path = write_raw_config(name, text)

    outcome = run_gate(
        config_path, regression_only=True, fallback_report_path=staged_report_rel(name)
    )

    assert outcome.exit_code == EXIT_PREFLIGHT
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["error_code"] == QualityGateErrorCode.CONFIG_SCHEMA_VIOLATION


def test_duplicate_corpus_key_exits_two_after_byte_hash_preflight_passes() -> None:
    """语料重复键：**先过字节哈希预检**，再在解析阶段以 `corpus_schema_violation` 失败。"""
    name = "dup_corpus_key"
    text = duplicate_line(
        corpus_text(), '"schema_version": "cw009_model_regression_corpus_v1",'
    )
    corpus_rel, corpus_sha = write_corpus_text(name, text)
    config_path, report_path = write_config(name, point_at_staged_corpus(corpus_rel, corpus_sha))

    outcome = run_gate(config_path, regression_only=True)

    assert outcome.exit_code == EXIT_PREFLIGHT
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["error_code"] == QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION
    # 字节哈希预检**通过**（match 为 true）：证明是"先验哈希、后解析"，
    # 而不是因为指纹不符才退出。
    assert report["inputs"]["corpus"]["match"] is True
    assert report["inputs"]["corpus"]["actual_sha256"] == corpus_sha
    assert report["regression"] == {}
    assert report["pytest"] == {"status": "not_run"}


@pytest.mark.parametrize(
    "needle",
    [
        '"kind": "synthetic_golden_master_regression_corpus",',
        '"source_features": [',
        '"case_id": "case_base_calm",',
        '"varied_features": [],',
        '"C1": 0.0,',
        '"expected": {',
        '"score": 0.08124042039839917,',
        '"risk_level": "low_signal"',
    ],
)
def test_duplicate_corpus_key_at_nested_levels_exits_two(needle: str) -> None:
    name = "dup_corpus_nested"
    text = duplicate_line(corpus_text(), needle)
    corpus_rel, corpus_sha = write_corpus_text(name, text)
    config_path, _ = write_config(name, point_at_staged_corpus(corpus_rel, corpus_sha))

    outcome = run_gate(config_path, regression_only=True)

    assert outcome.exit_code == EXIT_PREFLIGHT
    assert outcome.report["error_code"] == QualityGateErrorCode.CORPUS_SCHEMA_VIOLATION


# --- 哈希失败报告：四个实际摘要 -----------------------------------------


def test_hash_mismatch_report_records_all_four_actuals_end_to_end() -> None:
    """指纹不符时退出 2（不是 6），且报告里四个存在输入的 actual/match 必须完整。"""

    def mutate(section: dict) -> None:
        section["inputs"]["corpus"]["sha256"] = "0" * 64

    config_path, report_path = write_config("mismatch_actuals", mutate)
    outcome = run_gate(config_path, regression_only=True)

    assert outcome.exit_code == EXIT_PREFLIGHT
    assert outcome.report_written is True
    report = json.loads(report_path.read_text(encoding="utf-8"))
    inputs = report["inputs"]

    assert set(inputs) == {"service_config", "bundle", "policy", "corpus"}
    for label, relative in (
        ("service_config", SERVICE_CONFIG_REL),
        ("bundle", BUNDLE_REL),
        ("policy", POLICY_REL),
        ("corpus", CORPUS_REL),
    ):
        assert inputs[label]["path"] == relative
        assert inputs[label]["actual_sha256"] == sha256_of(relative)
    assert [inputs[label]["match"] for label in ("service_config", "bundle", "policy", "corpus")] == [
        True,
        True,
        True,
        False,
    ]
    assert report["error_code"] == QualityGateErrorCode.INPUT_HASH_MISMATCH
    assert report["regression"] == {}
    assert report["pytest"] == {"status": "not_run"}


def test_two_hash_mismatches_are_both_recorded_end_to_end() -> None:
    def mutate(section: dict) -> None:
        section["inputs"]["bundle"]["sha256"] = "0" * 64
        section["inputs"]["corpus"]["sha256"] = "1" * 64

    config_path, report_path = write_config("mismatch_two", mutate)
    outcome = run_gate(config_path, regression_only=True)

    assert outcome.exit_code == EXIT_PREFLIGHT
    inputs = json.loads(report_path.read_text(encoding="utf-8"))["inputs"]
    assert inputs["bundle"]["match"] is False
    assert inputs["corpus"]["match"] is False
    assert inputs["service_config"]["match"] is True
    assert inputs["policy"]["match"] is True
    assert all(entry["actual_sha256"] for entry in inputs.values())


def test_staged_tamper_never_touches_the_frozen_corpus() -> None:
    """篡改只发生在临时副本上：跑完所有反例后，仓库语料仍然逐字节未变。"""
    before = {
        relative: sha256_of(relative)
        for relative in (CORPUS_REL, CONFIG_REL, SERVICE_CONFIG_REL, BUNDLE_REL, POLICY_REL)
    }
    corpus_rel, corpus_sha = write_corpus(
        "isolation", lambda raw: corpus_case(raw, "case_base_calm").__setitem__("notes", "tampered")
    )
    config_path, _ = write_config("isolation", point_at_staged_corpus(corpus_rel, corpus_sha))
    run_gate(config_path, regression_only=True)

    assert {relative: sha256_of(relative) for relative in before} == before
    assert sha256_of(CORPUS_REL) == "d58166b70223af0adb443b9acff0690e6e3c571110dc11b16c2395a8d6fdc4b0"


def test_frozen_fingerprints_are_unchanged_by_the_whole_suite() -> None:
    """门禁与专项测试都是只读的：四个**已提交**输入不得被改写。

    默认决策数据库不在这里断言固定摘要：它是运行时产物、被 ``.gitignore`` 排除，
    干净检出里本来就不存在，断言固定摘要等于要求 CI 提供开发机数据库。
    它的"不被改写、也不被凭空创建"由根 ``conftest.py`` 的会话级守卫
    ``guard_default_database`` 负责（CW-014），在有无该文件的两种环境下都成立。
    """
    expected = {
        SERVICE_CONFIG_REL: "4dd211557214cee85abf1f427f72bfcec3510f6a4d01a928becb141fffb1c865",
        BUNDLE_REL: "f02bb50e61c5f70c85448430c8bb653bd6e347f400220771500ee84a70135a1c",
        POLICY_REL: "7ed07f0b784715c01b444035bb34ea5166d1b376ff07e10428561d898d6824f0",
        CORPUS_REL: "d58166b70223af0adb443b9acff0690e6e3c571110dc11b16c2395a8d6fdc4b0",
    }
    for relative, digest in expected.items():
        assert sha256_of(relative) == digest, relative
