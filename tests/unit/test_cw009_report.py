"""CW-009 报告：脱敏断言 + 临时文件 + ``os.replace`` 原子写入。

报告是唯一产物，也是最容易泄漏的产物。这里的用例逐条守住"什么绝不允许写进报告"：
绝对路径、UUID、计划外的 64 位摘要、源特征名与类别合成取值。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_detector.quality.errors import QualityGateError, QualityGateErrorCode
from ai_detector.quality.report import (
    REPORT_SCHEMA_VERSION,
    assert_report_safe,
    dumps,
    write_report,
)

ALLOWED_DIGEST = "d58166b70223af0adb443b9acff0690e6e3c571110dc11b16c2395a8d6fdc4b0"
OTHER_DIGEST = "0" * 64
FORBIDDEN = frozenset({"TransactionAmt", "cw009_cat_alpha"})


def safe_report() -> dict:
    return {
        "work_package": "CW-009",
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": "2026-09-26T02:35:21Z",
        "overall_status": "passed",
        "exit_code": 0,
        "inputs": {
            "corpus": {
                "path": "tests/fixtures/cw009/model_regression_cases.json",
                "declared_sha256": ALLOWED_DIGEST,
                "actual_sha256": ALLOWED_DIGEST,
            }
        },
        "regression": {
            "failed_case_ids": ["case_base_calm"],
            "failed_fields": {"case_base_calm": ["score"]},
            "max_score_absolute_deviation": 0.25,
        },
    }


# --- 正常路径 -----------------------------------------------------------


def test_safe_report_passes_assertion() -> None:
    assert_report_safe(
        safe_report(), allowed_digests={ALLOWED_DIGEST}, forbidden_tokens=FORBIDDEN
    )


def test_write_report_roundtrips_and_leaves_no_temp_file(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "report.json"
    write_report(
        target,
        safe_report(),
        allowed_digests={ALLOWED_DIGEST},
        forbidden_tokens=FORBIDDEN,
    )

    assert json.loads(target.read_text(encoding="utf-8"))["work_package"] == "CW-009"
    leftovers = [item.name for item in target.parent.iterdir() if item.name != "report.json"]
    assert leftovers == []


def test_write_report_overwrites_existing_file_atomically(tmp_path: Path) -> None:
    target = tmp_path / "report.json"
    target.write_text('{"stale": true}', encoding="utf-8")
    write_report(
        target,
        safe_report(),
        allowed_digests={ALLOWED_DIGEST},
        forbidden_tokens=FORBIDDEN,
    )
    assert "stale" not in target.read_text(encoding="utf-8")
    assert [item.name for item in tmp_path.iterdir()] == ["report.json"]


def test_dumps_is_utf8_and_stable() -> None:
    text = dumps({"a": "中文", "b": 1})
    assert "中文" in text
    assert text.endswith("\n")
    assert json.loads(text) == {"a": "中文", "b": 1}


# --- 脱敏反例 -----------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "C:/ai_detector/artifacts/report.json",
        "C:\\ai_detector\\artifacts\\report.json",
        "\\\\server\\share\\report.json",
        "/var/tmp/report.json",
    ],
)
def test_absolute_paths_are_rejected(value: str) -> None:
    report = safe_report()
    report["note"] = value
    with pytest.raises(QualityGateError) as info:
        assert_report_safe(report, allowed_digests={ALLOWED_DIGEST}, forbidden_tokens=FORBIDDEN)
    assert info.value.code == QualityGateErrorCode.REPORT_UNSAFE


def test_absolute_path_in_a_nested_key_is_rejected() -> None:
    report = safe_report()
    report["inputs"]["corpus"]["path"] = "D:/ai_detector/tests/fixtures/cw009/cases.json"
    with pytest.raises(QualityGateError) as info:
        assert_report_safe(report, allowed_digests={ALLOWED_DIGEST}, forbidden_tokens=FORBIDDEN)
    assert info.value.code == QualityGateErrorCode.REPORT_UNSAFE


def test_uuid_is_rejected() -> None:
    report = safe_report()
    report["request_id"] = "3f2a9c1e-8b44-4d0e-9a11-77c2b0d5e6f7"
    with pytest.raises(QualityGateError) as info:
        assert_report_safe(report, allowed_digests={ALLOWED_DIGEST}, forbidden_tokens=FORBIDDEN)
    assert info.value.code == QualityGateErrorCode.REPORT_UNSAFE


def test_unexpected_digest_is_rejected() -> None:
    report = safe_report()
    report["idempotency_key_digest"] = OTHER_DIGEST
    with pytest.raises(QualityGateError) as info:
        assert_report_safe(report, allowed_digests={ALLOWED_DIGEST}, forbidden_tokens=FORBIDDEN)
    assert info.value.code == QualityGateErrorCode.REPORT_UNSAFE


def test_allowed_digests_are_accepted() -> None:
    report = safe_report()
    report["extra_digest"] = OTHER_DIGEST
    assert_report_safe(
        report, allowed_digests={ALLOWED_DIGEST, OTHER_DIGEST}, forbidden_tokens=FORBIDDEN
    )


@pytest.mark.parametrize("token", ["TransactionAmt", "cw009_cat_alpha"])
def test_feature_names_and_category_values_are_rejected(token: str) -> None:
    report = safe_report()
    report["regression"]["failed_fields"] = {"case_base_calm": [token]}
    with pytest.raises(QualityGateError) as info:
        assert_report_safe(report, allowed_digests={ALLOWED_DIGEST}, forbidden_tokens=FORBIDDEN)
    assert info.value.code == QualityGateErrorCode.REPORT_UNSAFE


def test_key_names_are_scanned_too() -> None:
    report = safe_report()
    report["features"] = {"TransactionAmt": 1.0}
    with pytest.raises(QualityGateError) as info:
        assert_report_safe(report, allowed_digests={ALLOWED_DIGEST}, forbidden_tokens=FORBIDDEN)
    assert info.value.code == QualityGateErrorCode.REPORT_UNSAFE


def test_unsafe_report_is_never_written(tmp_path: Path) -> None:
    target = tmp_path / "report.json"
    report = safe_report()
    report["note"] = "C:/leak.json"
    with pytest.raises(QualityGateError) as info:
        write_report(
            target, report, allowed_digests={ALLOWED_DIGEST}, forbidden_tokens=FORBIDDEN
        )
    assert info.value.code == QualityGateErrorCode.REPORT_UNSAFE
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


# --- 写入失败 -----------------------------------------------------------


def test_write_failure_maps_to_report_write_failed(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    with pytest.raises(QualityGateError) as info:
        write_report(
            blocker / "report.json",
            safe_report(),
            allowed_digests={ALLOWED_DIGEST},
            forbidden_tokens=FORBIDDEN,
        )
    assert info.value.code == QualityGateErrorCode.REPORT_WRITE_FAILED
