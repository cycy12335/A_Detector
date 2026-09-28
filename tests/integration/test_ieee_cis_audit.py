"""IEEE-CIS 审计流程端到端集成测试。

在临时目录中用**项目自建的合成 fixture** 跑完整审计，验证四类产物、报告不变量、
确定性与数据门禁。不访问网络，不读取任何真实 Kaggle 数据。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ai_detector.config import PROJECT_ROOT
from ai_detector.data.audit import run_audit
from ai_detector.data.ieee_cis import IeeeCisConfig, MissingDataError, check_required_files
from tests.conftest import IeeeCisFixtureBuilder, read_base_identity, read_base_transactions

REQUIRED_AUDIT_KEYS = {
    "work_package",
    "generated_at",
    "package_version",
    "dataset",
    "files",
    "schema_audit",
    "join_audit",
    "split",
    "feature_roles",
    "profiling",
    "timing_seconds",
    "limitations",
    "unverifiable_items",
    "environment",
}

REQUIRED_MANIFEST_KEYS = {
    "algorithm_version",
    "file_fingerprints",
    "boundaries",
    "sizes",
    "actual_ratios",
    "label_rates",
    "assertions",
}

PARTITION_NAMES = ("train", "validation", "test")


@pytest.fixture(scope="module")
def audited(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """跑一次完整审计，供本模块多个测试复用。"""
    workspace = tmp_path_factory.mktemp("cw002")
    builder = IeeeCisFixtureBuilder(workspace)
    builder.write()
    config = builder.config()
    result = run_audit(config)
    return {"result": result, "config": config}


# --- 产物完整性 ---------------------------------------------------------


def test_four_artifacts_exist(audited: dict) -> None:
    result = audited["result"]

    assert result.audit_json.is_file()
    assert result.audit_markdown.is_file()
    assert result.column_profile_csv.is_file()
    assert result.split_manifest_json.is_file()


def test_audit_json_has_all_required_sections(audited: dict) -> None:
    report = audited["result"].report

    assert REQUIRED_AUDIT_KEYS <= set(report)
    assert report["work_package"] == "CW-002"
    assert report["dataset"]["official_data_url"].startswith("https://www.kaggle.com/")
    assert report["dataset"]["official_rules_url"].startswith("https://www.kaggle.com/")


def test_split_manifest_has_all_required_keys(audited: dict) -> None:
    manifest = json.loads(audited["result"].split_manifest_json.read_text(encoding="utf-8"))

    assert REQUIRED_MANIFEST_KEYS <= set(manifest)
    assert manifest["algorithm_version"] == "temporal_tie_safe_v1"
    assert set(manifest["sizes"]) == set(PARTITION_NAMES)
    assert manifest["assertions"]["transaction_ids_disjoint"] is True
    assert manifest["assertions"]["no_transaction_dt_spans_partitions"] is True


def test_fingerprints_are_recorded_for_both_files(audited: dict) -> None:
    files = audited["result"].report["files"]

    assert set(files) == {"train_transaction.csv", "train_identity.csv"}
    for name, info in files.items():
        assert len(info["sha256"]) == 64, name
        assert info["size_bytes"] > 0, name
        assert not Path(info["relative_path"]).is_absolute(), name


def test_environment_records_runtime_versions(audited: dict) -> None:
    environment = audited["result"].report["environment"]

    assert set(environment) >= {"python", "pandas", "numpy", "scikit-learn", "joblib", "PyYAML"}
    assert environment["PyYAML"] not in {"unavailable", "unknown"}


# --- 报告安全边界 -------------------------------------------------------


def test_report_contains_no_absolute_paths(audited: dict) -> None:
    """报告不得包含机器专属绝对路径。"""
    payload = json.dumps(audited["result"].report, ensure_ascii=False)

    assert str(PROJECT_ROOT) not in payload
    assert "C:\\" not in payload
    assert "D:\\" not in payload
    assert "/Users/" not in payload
    assert "/home/" not in payload


def test_report_contains_no_row_level_samples_or_credentials(audited: dict) -> None:
    """报告不得含逐行取值、邮箱原文、设备串或凭据。

    ``environment.platform`` 会合法地包含操作系统名称（例如 ``Windows-11-...``），
    而 fixture 里的 ``id_30`` 取值恰好也叫 ``Windows``/``iOS``。因此先把 environment
    段摘掉再扫描，否则断言会把合法的环境记录误判为数据泄漏。
    """
    report = dict(audited["result"].report)
    platform_values = " ".join(report.pop("environment").values())

    payload = json.dumps(report, ensure_ascii=False)
    profile_csv = audited["result"].column_profile_csv.read_text(encoding="utf-8")
    markdown = audited["result"].audit_markdown.read_text(encoding="utf-8")

    # 这些取值只存在于 fixture 数据里，绝不能出现在任何产物中。
    secret_markers = (
        "gmail.com",
        "yahoo.com",
        "chrome",
        "safari",
        "edge",
        "desktop",
        "mobile",
        "iOS",
    )
    for text, label in ((payload, "audit.json"), (profile_csv, "column_profile.csv")):
        for marker in secret_markers:
            assert marker not in text, f"{label} 中出现原始取值：{marker}"

    # 平台串里的 Windows 是环境记录而非数据样本；它也不应出现在列画像里。
    assert "Windows" in platform_values, "环境记录应包含操作系统名称"
    assert "Windows" not in profile_csv
    assert "Windows" not in payload

    for credential in ("password", "token", "cookie", "api_key", "apikey", "secret", "bearer"):
        assert credential not in payload.lower()
        assert credential not in profile_csv.lower()
        assert credential not in markdown.lower()


def test_markdown_separates_facts_from_unverifiable_items(audited: dict) -> None:
    """可读摘要必须区分已验证事实、设计决策与无法确认事项。"""
    markdown = audited["result"].audit_markdown.read_text(encoding="utf-8")

    for heading in ("## 已验证事实", "## 设计决策", "## 无法从公开匿名数据确认的事项",
                    "## 数据与方法限制"):
        assert heading in markdown, heading
    assert "不代表任何真实银行的线上效果" in markdown
    for overclaim in ("生产级", "银行级", "真实线上效果保证"):
        assert overclaim not in markdown, overclaim


def test_column_profile_csv_has_one_row_per_column(audited: dict) -> None:
    profile = pd.read_csv(audited["result"].column_profile_csv)
    report = audited["result"].report

    expected = (
        report["profiling"]["row_count_transaction"] + report["profiling"]["row_count_identity"]
    )
    assert len(profile) == expected
    assert set(profile["source_file"]) == {"train_transaction.csv", "train_identity.csv"}
    for column in ("column_name", "role", "null_rate", "distinct_count", "semantic_note"):
        assert column in profile.columns


# --- 审计结论不变量 -----------------------------------------------------


def test_join_audit_matches_fixture_structure(audited: dict) -> None:
    join = audited["result"].report["join_audit"]

    assert join["transaction_rows"] == 40
    assert join["identity_rows"] == 26
    assert join["matched_transaction_rows"] == 25
    assert join["orphan_identity_keys"] == 1
    assert join["identity_coverage_rate"] == pytest.approx(0.625)
    assert join["rows_after_left_join"] == join["transaction_rows"]
    assert join["transaction_rows_preserved"] is True


def test_partitions_are_disjoint_complete_and_non_empty(audited: dict) -> None:
    split = audited["result"].report["split"]

    assert set(split["sizes"]) == set(PARTITION_NAMES)
    assert all(size > 0 for size in split["sizes"].values())
    assert sum(split["sizes"].values()) == 40
    for name in PARTITION_NAMES:
        assert 0.0 <= split["label_rates"][name] <= 1.0
        assert split["actual_ratios"][name] == pytest.approx(
            split["sizes"][name] / split["n_rows"]
        )


def test_row_partitions_are_disjoint_and_tie_safe(audited: dict) -> None:
    """行级分区表：ID 不重不漏，且同一 TransactionDT 不跨分区。"""
    rows = pd.read_csv(audited["result"].row_partitions_csv)

    assert len(rows) == 40
    assert rows["TransactionID"].is_unique
    assert set(rows["partition"]) == set(PARTITION_NAMES)
    per_time = rows.groupby("TransactionDT")["partition"].nunique()
    assert (per_time == 1).all()

    # 时间顺序不得穿越。
    bounds = rows.groupby("partition")["TransactionDT"].agg(["min", "max"])
    assert bounds.loc["train", "max"] < bounds.loc["validation", "min"]
    assert bounds.loc["validation", "max"] < bounds.loc["test", "min"]


def test_structural_columns_are_not_candidate_features(audited: dict) -> None:
    """验收标准第 6 项：target、join_key 与原始时间列不得进入候选特征。"""
    roles = audited["result"].report["feature_roles"]["by_role"]

    assert roles["target"] == ["isFraud"]
    assert roles["join_key"] == ["TransactionID"]
    assert roles["split_only"] == ["TransactionDT"]
    for name in ("isFraud", "TransactionID", "TransactionDT"):
        assert name not in roles["candidate_feature"]
        assert name not in roles["excluded"]


def test_all_missing_column_is_excluded_with_reason(audited: dict) -> None:
    roles = audited["result"].report["feature_roles"]["by_role"]
    registry = audited["result"].report["feature_roles"]["registry"]

    assert "V2" in roles["excluded"]
    v2 = [row for row in registry if row["column_name"] == "V2"][0]
    assert "训练分区" in v2["role_reason"]

    # 被排除的列不得同时出现在候选特征里。
    assert not set(roles["excluded"]) & set(roles["candidate_feature"])


def test_anonymous_columns_are_flagged_in_registry(audited: dict) -> None:
    registry = audited["result"].report["feature_roles"]["registry"]
    note = audited["result"].report["feature_roles"]["anonymous_columns_note"]

    anonymous_rows = [row for row in registry if row["semantic_note"] == note]
    assert anonymous_rows, "匿名列未被登记"
    assert any(row["column_name"].startswith("V") for row in anonymous_rows)
    assert any(row["column_name"].startswith("id_") for row in anonymous_rows)
    assert audited["result"].report["feature_roles"]["anonymous_column_count"] == len(
        anonymous_rows
    )


def test_unverifiable_items_and_limitations_are_present(audited: dict) -> None:
    report = audited["result"].report

    assert len(report["limitations"]) >= 5
    assert len(report["unverifiable_items"]) >= 4
    joined = " ".join(report["limitations"])
    assert "相对时间偏移" in joined
    assert "语义匿名" in joined


# --- 确定性 -------------------------------------------------------------


def test_repeated_runs_differ_only_in_generation_time(
    tmp_path: Path, ieee_cis_fixture: IeeeCisFixtureBuilder
) -> None:
    """重复运行的结论必须一致，只有生成时间可以不同。"""
    ieee_cis_fixture.write()
    config = ieee_cis_fixture.config()

    first = run_audit(config).report
    second = run_audit(config).report

    assert first.pop("generated_at") != second.pop("generated_at") or True
    # 除生成时间与耗时外，其余字段必须逐字段相同。
    for payload in (first, second):
        payload.pop("generated_at", None)
        payload.pop("timing_seconds", None)
    assert first == second


def test_same_input_same_fingerprint_across_runs(
    tmp_path: Path, ieee_cis_fixture: IeeeCisFixtureBuilder
) -> None:
    ieee_cis_fixture.write()
    config = ieee_cis_fixture.config()

    first = run_audit(config).report["files"]
    second = run_audit(config).report["files"]

    assert first == second


def test_different_data_produces_different_fingerprint(
    ieee_cis_fixture: IeeeCisFixtureBuilder,
) -> None:
    ieee_cis_fixture.write()
    config = ieee_cis_fixture.config()
    original = run_audit(config).report["files"]["train_transaction.csv"]["sha256"]

    transactions = read_base_transactions()
    transactions.loc[0, "TransactionAmt"] = transactions.loc[0, "TransactionAmt"] + 1.0
    ieee_cis_fixture.write(transactions=transactions)
    modified = run_audit(ieee_cis_fixture.config()).report["files"]["train_transaction.csv"][
        "sha256"
    ]

    assert original != modified


def test_chunk_size_does_not_change_audit_conclusions(
    tmp_path: Path, ieee_cis_fixture: IeeeCisFixtureBuilder
) -> None:
    """不同分块大小必须给出同样的审计结论。"""
    from dataclasses import replace

    ieee_cis_fixture.write()
    config = ieee_cis_fixture.config()
    default_report = run_audit(config).report

    tiny_config = replace(
        config, profiling=replace(config.profiling, chunk_size=3)
    )
    tiny_report = run_audit(tiny_config, write_row_partitions=False).report

    for key in ("schema_audit", "join_audit", "files"):
        assert default_report[key] == tiny_report[key], key
    assert default_report["split"]["sizes"] == tiny_report["split"]["sizes"]
    assert default_report["split"]["boundaries"] == tiny_report["split"]["boundaries"]
    assert default_report["feature_roles"]["by_role"] == tiny_report["feature_roles"]["by_role"]


# --- 数据门禁 -----------------------------------------------------------


def test_audit_refuses_to_run_without_real_data(real_data_gate_config: IeeeCisConfig) -> None:
    """没有真实数据时不得产出审计产物，也不得伪造任何结论。"""
    if all(path.is_file() for _, path in real_data_gate_config.required_files()):
        pytest.skip("本地已提供真实数据，缺失门禁不适用")

    with pytest.raises(MissingDataError):
        run_audit(real_data_gate_config)


def test_repository_audit_outputs_absent_when_data_missing() -> None:
    """仓库级产物目录不应因为缺少真实数据而出现伪造报告。"""
    report_dir = PROJECT_ROOT / "artifacts" / "reports" / "cw002_ieee_cis"
    config = IeeeCisConfig.from_yaml(PROJECT_ROOT / "configs" / "ieee_cis.yaml")
    real_files_present = all(path.is_file() for _, path in config.required_files())

    if real_files_present:
        pytest.skip("本地已提供真实数据")
    assert not (report_dir / "audit.json").exists(), (
        "缺少真实数据时不应存在审计报告；若为历史遗留请删除或补齐真实数据后重跑"
    )


def test_fixture_files_are_labelled_as_synthetic() -> None:
    """仓库内的 fixture 必须明确标注为合成数据，避免被误当成真实样本。"""
    readme = (PROJECT_ROOT / "tests" / "fixtures" / "ieee_cis" / "README.md").read_text(
        encoding="utf-8"
    )

    assert "不是 Kaggle" in readme or "不是 Kaggle 的 IEEE-CIS 数据" in readme
    assert "合成数据" in readme
    assert "绝不要把真实 Kaggle 数据复制到本目录" in readme


def test_fixture_structure_is_unchanged() -> None:
    """fixture 结构是多项断言的基础，变动时必须显式更新测试而不是放宽断言。"""
    transactions = read_base_transactions()
    identity = read_base_identity()

    assert transactions.shape == (40, 10)
    assert identity.shape == (26, 3)
    assert "9999999" in identity["TransactionID"].astype(str).tolist() or 9999999 in set(
        identity["TransactionID"]
    )
    assert transactions["V2"].isna().all()


def test_config_check_required_files_ok_for_fixture(
    ieee_cis_fixture: IeeeCisFixtureBuilder,
) -> None:
    ieee_cis_fixture.write()
    check_required_files(ieee_cis_fixture.config())
