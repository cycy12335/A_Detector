"""IEEE-CIS 配置、Schema 与连接契约测试。

全部使用项目自建的合成 fixture（``tests/fixtures/ieee_cis/``），不访问网络，
不读取任何真实 Kaggle 数据。
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

from ai_detector.config import ConfigError, PROJECT_ROOT
from ai_detector.data.ieee_cis import (
    DataContractError,
    IeeeCisConfig,
    MissingDataError,
    assert_binary_target,
    audit_join,
    check_required_files,
    read_header_columns,
    scan_transactions,
)
from tests.conftest import IeeeCisFixtureBuilder, read_base_identity


# --- 配置 ---------------------------------------------------------------


def test_repository_config_declares_all_required_keys() -> None:
    """仓库配置必须能加载，且禁止特征清单覆盖三个结构性列。"""
    config = IeeeCisConfig.from_yaml(PROJECT_ROOT / "configs" / "ieee_cis.yaml")

    assert config.join_key == "TransactionID"
    assert config.target == "isFraud"
    assert config.split_time == "TransactionDT"
    assert config.train_ratio == 0.6
    assert config.validation_ratio == 0.2
    assert config.raw_dir.name == "ieee_cis"
    for key in (config.join_key, config.target, config.split_time):
        assert key in config.forbidden_features
    assert config.meta.official_data_url.startswith("https://www.kaggle.com/")
    assert config.meta.official_rules_url.startswith("https://www.kaggle.com/")


def test_config_rejects_incomplete_forbidden_features() -> None:
    """结构性列若未列入禁止特征清单，配置必须被拒绝，否则角色判定会出现漏洞。"""
    import yaml

    raw = yaml.safe_load((PROJECT_ROOT / "configs" / "ieee_cis.yaml").read_text(encoding="utf-8"))
    raw["forbidden_features"] = [
        item for item in raw["forbidden_features"] if item["column"] != "TransactionDT"
    ]

    with pytest.raises(ConfigError, match="forbidden_features"):
        IeeeCisConfig.from_mapping(raw)


def test_config_contains_no_absolute_paths() -> None:
    """配置不得包含个人绝对路径或凭据。"""
    text = (PROJECT_ROOT / "configs" / "ieee_cis.yaml").read_text(encoding="utf-8")

    assert "C:\\" not in text
    assert "D:\\" not in text
    assert "/Users/" not in text
    assert "/home/" not in text
    for secret in ("password", "token", "cookie", "api_key", "apikey", "secret"):
        assert secret not in text.lower()


def test_config_paths_resolve_under_project_root() -> None:
    config = IeeeCisConfig.from_yaml(PROJECT_ROOT / "configs" / "ieee_cis.yaml")

    for path in (config.raw_dir, config.transaction_path, *config.outputs.values()):
        assert PROJECT_ROOT.resolve() in path.resolve().parents or path.resolve() == PROJECT_ROOT.resolve()


# --- 文件缺失门禁 -------------------------------------------------------


def test_missing_files_raise_clear_error(ieee_cis_fixture: IeeeCisFixtureBuilder) -> None:
    """两个文件都缺失时，错误必须列出缺失项、放置路径与人工步骤。"""
    config = ieee_cis_fixture.config()
    ieee_cis_fixture.raw_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(MissingDataError) as error:
        check_required_files(config)

    message = str(error.value)
    assert "train_transaction.csv" in message
    assert "train_identity.csv" in message
    assert "https://www.kaggle.com/competitions/ieee-fraud-detection/data" in message
    assert "不代替使用者同意条款" in message
    assert str(PROJECT_ROOT) not in message, "错误信息不得包含机器专属绝对路径"


def test_missing_identity_only(ieee_cis_fixture: IeeeCisFixtureBuilder) -> None:
    """只缺 identity 时也要报错：连接审计无法在缺一个文件时完成。"""
    ieee_cis_fixture.write(write_identity=False)
    config = ieee_cis_fixture.config()

    with pytest.raises(MissingDataError, match="train_identity.csv"):
        check_required_files(config)


def test_real_data_gate_is_reported_without_faking(config: IeeeCisConfig | None = None) -> None:
    """仓库真实数据目录当前为空时，必须报缺失而不是伪造审计结果。

    这是数据门禁的核心断言：没有真实数据就不得声称完成。
    """
    repository_config = IeeeCisConfig.from_yaml(PROJECT_ROOT / "configs" / "ieee_cis.yaml")
    real_files = [path for _, path in repository_config.required_files() if path.is_file()]

    if real_files:
        pytest.skip("本地已提供真实数据，缺失门禁不适用")
    with pytest.raises(MissingDataError):
        check_required_files(repository_config)


# --- Schema 审计 --------------------------------------------------------


def test_scan_reports_structural_facts(ieee_cis_fixture: IeeeCisFixtureBuilder) -> None:
    ieee_cis_fixture.write()
    config = ieee_cis_fixture.config()

    scan, ids, times, fraud = scan_transactions(config)

    assert scan.n_rows == len(ids) == len(times) == len(fraud)
    assert scan.n_null_keys == 0
    assert scan.n_duplicate_key_rows == 0
    # 报告面向的取值形式是排序后的列表。
    assert scan.to_dict()["observed_target_values"] == [0.0, 1.0]
    assert scan.to_dict()["n_columns"] == 10
    assert len(scan.columns) == 10
    assert scan.min_time <= scan.max_time
    assert len(np.unique(ids)) == len(ids)


def test_missing_required_column_fails(ieee_cis_fixture: IeeeCisFixtureBuilder) -> None:
    from tests.conftest import read_base_transactions

    transactions = read_base_transactions().drop(columns=["TransactionDT"])
    ieee_cis_fixture.write(transactions=transactions)
    config = ieee_cis_fixture.config()

    with pytest.raises(DataContractError, match="缺少必需列"):
        scan_transactions(config)


def test_duplicate_transaction_id_fails(ieee_cis_fixture: IeeeCisFixtureBuilder) -> None:
    from tests.conftest import read_base_transactions

    transactions = read_base_transactions()
    transactions.loc[1, "TransactionID"] = transactions.loc[0, "TransactionID"]
    ieee_cis_fixture.write(transactions=transactions)
    config = ieee_cis_fixture.config()

    with pytest.raises(DataContractError, match="重复"):
        scan_transactions(config)


def test_null_transaction_id_fails(ieee_cis_fixture: IeeeCisFixtureBuilder) -> None:
    from tests.conftest import read_base_transactions

    transactions = read_base_transactions()
    transactions.loc[0, "TransactionID"] = np.nan
    ieee_cis_fixture.write(transactions=transactions)
    config = ieee_cis_fixture.config()

    with pytest.raises(DataContractError, match="TransactionID 存在空值"):
        scan_transactions(config)


def test_non_finite_time_fails(ieee_cis_fixture: IeeeCisFixtureBuilder) -> None:
    from tests.conftest import read_base_transactions

    transactions = read_base_transactions()
    # int64 列不能直接写入 inf，先转 float 才能构造这个非法输入。
    transactions["TransactionDT"] = transactions["TransactionDT"].astype(float)
    transactions.loc[0, "TransactionDT"] = np.inf
    ieee_cis_fixture.write(transactions=transactions)
    config = ieee_cis_fixture.config()

    with pytest.raises(DataContractError, match="非有限值"):
        scan_transactions(config)


def test_illegal_label_value_fails(ieee_cis_fixture: IeeeCisFixtureBuilder) -> None:
    """标签必须严格属于 {0, 1}，不得静默截断 0.3 这类浮点值。"""
    from tests.conftest import read_base_transactions

    transactions = read_base_transactions()
    transactions["isFraud"] = transactions["isFraud"].astype(float)
    transactions.loc[0, "isFraud"] = 0.3
    ieee_cis_fixture.write(transactions=transactions)
    config = ieee_cis_fixture.config()

    with pytest.raises(DataContractError, match=r"必须严格属于 \{0, 1\}"):
        scan_transactions(config)


def test_null_label_fails(ieee_cis_fixture: IeeeCisFixtureBuilder) -> None:
    from tests.conftest import read_base_transactions

    transactions = read_base_transactions()
    transactions.loc[0, "isFraud"] = np.nan
    ieee_cis_fixture.write(transactions=transactions)
    config = ieee_cis_fixture.config()

    with pytest.raises(DataContractError, match="存在空值"):
        scan_transactions(config)


@pytest.mark.parametrize(
    "values",
    [
        pd.Series([0.0, 1.0, 0.3]),
        pd.Series([0, 2, 1]),
        pd.Series([0.0, np.nan, 1.0]),
        pd.Series([0.0, np.inf, 1.0]),
    ],
)
def test_assert_binary_target_rejects_invalid(values: pd.Series) -> None:
    with pytest.raises(DataContractError):
        assert_binary_target(values, target="isFraud", file_label="fixture")


# --- 连接审计 -----------------------------------------------------------


def test_join_audit_reports_coverage_and_orphans(
    ieee_cis_fixture: IeeeCisFixtureBuilder,
) -> None:
    """覆盖率与孤立键统计必须与 fixture 的已知结构一致。"""
    ieee_cis_fixture.write()
    config = ieee_cis_fixture.config()
    _, ids, _, _ = scan_transactions(config)

    join = audit_join(config, ids)

    # fixture：26 行 identity，其中 25 行匹配交易、1 行是孤立键 9999999。
    assert join.identity_rows == 26
    assert join.identity_unique_keys == 26
    assert join.matched_transaction_rows == 25
    assert join.orphan_identity_keys == 1
    assert join.identity_coverage_rate == pytest.approx(25 / 40)


def test_left_join_preserves_transaction_rows(ieee_cis_fixture: IeeeCisFixtureBuilder) -> None:
    """左连接不得删除交易（identity 缺失）也不得放大交易（identity 重复）。"""
    ieee_cis_fixture.write()
    config = ieee_cis_fixture.config()
    _, ids, _, _ = scan_transactions(config)

    join = audit_join(config, ids)

    assert join.rows_after_left_join == join.transaction_rows
    assert join.transaction_rows_preserved is True


def test_identity_duplicate_key_fails(ieee_cis_fixture: IeeeCisFixtureBuilder) -> None:
    """identity 重复键会在连接时放大交易行数，必须在审计阶段直接失败。"""
    identity = read_base_identity()
    identity.loc[1, "TransactionID"] = identity.loc[0, "TransactionID"]
    ieee_cis_fixture.write(identity=identity)
    config = ieee_cis_fixture.config()
    _, ids, _, _ = scan_transactions(config)

    with pytest.raises(DataContractError, match="重复"):
        audit_join(config, ids)


def test_identity_missing_join_key_column_fails(ieee_cis_fixture: IeeeCisFixtureBuilder) -> None:
    identity = read_base_identity().rename(columns={"TransactionID": "OtherID"})
    ieee_cis_fixture.write(identity=identity)
    config = ieee_cis_fixture.config()
    _, ids, _, _ = scan_transactions(config)

    with pytest.raises(DataContractError, match="缺少必需列"):
        audit_join(config, ids)


def test_identity_null_key_fails(ieee_cis_fixture: IeeeCisFixtureBuilder) -> None:
    identity = read_base_identity()
    identity.loc[0, "TransactionID"] = np.nan
    ieee_cis_fixture.write(identity=identity)
    config = ieee_cis_fixture.config()
    _, ids, _, _ = scan_transactions(config)

    with pytest.raises(DataContractError, match="存在空值"):
        audit_join(config, ids)


def test_empty_identity_file_is_not_an_error_but_reports_zero_coverage(
    ieee_cis_fixture: IeeeCisFixtureBuilder,
) -> None:
    """identity 为空不构成数据错误，但必须如实报告 0 覆盖率，且不得删交易。"""
    empty = read_base_identity().iloc[:0]
    ieee_cis_fixture.write(identity=empty)
    config = ieee_cis_fixture.config()
    _, ids, _, _ = scan_transactions(config)

    join = audit_join(config, ids)

    assert join.identity_rows == 0
    assert join.identity_coverage_rate == 0.0
    assert join.orphan_identity_keys == 0
    assert join.transaction_rows_preserved is True


def test_read_header_columns_does_not_load_rows(ieee_cis_fixture: IeeeCisFixtureBuilder) -> None:
    ieee_cis_fixture.write()
    config = ieee_cis_fixture.config()

    columns = read_header_columns(config.transaction_path)

    assert columns == list(pd.read_csv(config.transaction_path, nrows=0).columns)
    assert "TransactionID" in columns


def test_scan_is_chunk_size_invariant(ieee_cis_fixture: IeeeCisFixtureBuilder) -> None:
    """分块大小不得影响审计结论。"""
    ieee_cis_fixture.write()
    config = ieee_cis_fixture.config()
    scan_default, ids_a, times_a, fraud_a = scan_transactions(config)

    from dataclasses import replace

    tiny = replace(config, profiling=replace(config.profiling, chunk_size=3))
    scan_tiny, ids_b, times_b, fraud_b = scan_transactions(tiny)

    assert scan_default.n_rows == scan_tiny.n_rows
    assert scan_default.to_dict() == scan_tiny.to_dict()
    np.testing.assert_array_equal(ids_a, ids_b)
    np.testing.assert_array_equal(times_a, times_b)
    np.testing.assert_array_equal(fraud_a, fraud_b)
