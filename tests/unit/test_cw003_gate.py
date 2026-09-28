"""CW-003 开工数据门禁与配置契约的测试。

门禁是"没有可信数据就不许开工"的那道闸门。测试必须证明它在**每一种**不一致下都会快速
失败，而不是只在文件缺失时才失败。
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import yaml

from ai_detector.config import ConfigError
from ai_detector.data.modeling import (
    DataGateError,
    ModelingConfig,
    ModelingContractError,
    align_chunk_with_lookup,
    load_partition_lookup,
    verify_data_gate,
)


@pytest.fixture
def ready(cw003_fixture):
    """构造夹具并返回 ``(builder工作区, 配置路径, 配置)``。"""
    config_path, config = cw003_fixture.build()
    verify_data_gate(config)
    return cw003_fixture, config_path, config


# --- 门禁通过 -----------------------------------------------------------


def test_gate_passes_on_consistent_fixture(ready) -> None:
    _, _, config = ready

    gate = verify_data_gate(config)

    assert set(gate["fingerprints"]) == {"train_transaction.csv", "train_identity.csv"}
    for info in gate["fingerprints"].values():
        assert info["match"] is True
        assert len(info["sha256"]) == 64
        assert not Path(info["relative_path"]).is_absolute()
    assert gate["split_manifest"]["unique_join_keys"] is True
    assert gate["split_manifest"]["sizes"] == config.expected_split_sizes


def test_gate_evidence_records_no_absolute_path(ready) -> None:
    """门禁证据只记录路径本身；路径是否在项目内由 relative_path 表达。"""
    _, _, config = ready
    gate = verify_data_gate(config)

    blob = json.dumps(gate, ensure_ascii=False)
    assert str(config.paths["transaction"].parent).replace("\\", "/") not in blob.replace("\\", "/")
    for info in gate["fingerprints"].values():
        assert ":" not in info["relative_path"][:3], "报告中的路径不应包含盘符"
        assert not Path(info["relative_path"]).is_absolute()


# --- 门禁失败路径 -------------------------------------------------------


def test_missing_file_blocks(ready) -> None:
    _, _, config = ready
    config.paths["transaction"].unlink()

    with pytest.raises(DataGateError, match="缺少原始文件"):
        verify_data_gate(config)


def test_fingerprint_mismatch_blocks(ready) -> None:
    """内容被改动（指纹变化）必须阻塞，而不是继续用新数据实验。"""
    _, _, config = ready
    frame = pd.read_csv(config.paths["transaction"])
    frame.loc[0, "AmtA"] = frame.loc[0, "AmtA"] + 1.0
    frame.to_csv(config.paths["transaction"], index=False)

    with pytest.raises(DataGateError, match="SHA-256 与 CW-002 记录不符"):
        verify_data_gate(config)


def test_missing_manifest_blocks(ready) -> None:
    _, _, config = ready
    config.paths["split_manifest_rows"].unlink()

    with pytest.raises(DataGateError, match="缺少 CW-002 行级切分清单"):
        verify_data_gate(config)


def test_manifest_size_mismatch_blocks(ready) -> None:
    _, _, config = ready
    broken = replace(
        config, expected_split_sizes={**config.expected_split_sizes, "train": 999}
    )

    with pytest.raises(DataGateError, match="规模"):
        verify_data_gate(broken)


def test_boundary_mismatch_blocks(ready) -> None:
    _, _, config = ready
    broken = replace(
        config, expected_boundaries={**config.expected_boundaries, "test_start": 123456.0}
    )

    with pytest.raises(DataGateError, match="边界"):
        verify_data_gate(broken)


def test_missing_cw002_audit_blocks(ready) -> None:
    _, _, config = ready
    config.paths["cw002_audit_json"].unlink()

    with pytest.raises(DataGateError, match="缺少 CW-002 audit.json"):
        verify_data_gate(config)


def test_gate_error_message_lists_every_problem(ready) -> None:
    """一次列全所有问题，避免用户反复试错。"""
    _, _, config = ready
    config.paths["transaction"].unlink()
    config.paths["cw002_audit_json"].unlink()

    with pytest.raises(DataGateError) as error:
        verify_data_gate(config)

    message = str(error.value)
    assert "缺少原始文件" in message
    assert "缺少 CW-002 audit.json" in message
    assert "blocked" in message
    assert "不得替换文件或重算切分" in message


# --- 分区来源 -----------------------------------------------------------


def test_unknown_join_key_blocks(ready) -> None:
    """清单之外的连接键必须报错：分区归属只能来自 CW-002 清单。"""
    _, _, config = ready
    lookup = load_partition_lookup(config)
    chunk = pd.DataFrame({config.join_key: [10_000_000], "AmtA": [1.0]})

    with pytest.raises(ModelingContractError, match="不在 CW-002 清单中"):
        align_chunk_with_lookup(config, lookup, chunk)


def test_partition_lookup_matches_manifest(ready) -> None:
    _, _, config = ready
    lookup = load_partition_lookup(config)
    manifest = pd.read_csv(config.paths["split_manifest_rows"])

    assert len(lookup) == config.expected_total_rows
    assert lookup.index.is_unique
    assert set(lookup.unique()) == {"train", "validation", "test"}
    merged = manifest.set_index(config.join_key)["partition"]
    pd.testing.assert_series_equal(lookup.sort_index(), merged.sort_index())


# --- 配置契约 -----------------------------------------------------------


def _load_config_mapping(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def test_config_rejects_forbidden_feature_in_allowlist(ready) -> None:
    _, config_path, _ = ready
    raw = _load_config_mapping(config_path)
    raw["feature_allowlist"]["sources"]["train_transaction.csv"]["columns"].append(
        {"name": "isFraud", "kind": "numeric", "reason_code": "anonymous"}
    )
    raw["feature_allowlist"]["sources"]["train_transaction.csv"]["declared_numeric"] += 1
    raw["feature_allowlist"]["sources"]["train_transaction.csv"]["declared_total"] += 1

    with pytest.raises(ConfigError, match="禁止字段"):
        ModelingConfig.from_mapping(raw, config_path=config_path)


def test_config_rejects_declared_count_mismatch(ready) -> None:
    _, config_path, _ = ready
    raw = _load_config_mapping(config_path)
    raw["feature_allowlist"]["sources"]["train_identity.csv"]["declared_total"] = 999

    with pytest.raises(ConfigError, match="declared_total"):
        ModelingConfig.from_mapping(raw, config_path=config_path)


def test_config_requires_fixed_column_contract(ready) -> None:
    """目标列 / 连接键 / 时间列的名字不得被改写。"""
    _, config_path, _ = ready
    raw = _load_config_mapping(config_path)
    raw["columns"]["target"] = "label"

    with pytest.raises(ConfigError, match="必须为 isFraud"):
        ModelingConfig.from_mapping(raw, config_path=config_path)


def test_config_forbidden_features_must_cover_structural_columns(ready) -> None:
    _, config_path, _ = ready
    raw = _load_config_mapping(config_path)
    raw["forbidden_features"] = ["isFraud"]

    with pytest.raises(ConfigError, match="forbidden_features"):
        ModelingConfig.from_mapping(raw, config_path=config_path)


def test_threshold_grid_expands_deterministically(ready) -> None:
    _, _, config = ready
    grid = config.threshold_grid()

    assert grid[0] == pytest.approx(0.05)
    assert grid[-1] == pytest.approx(0.95)
    assert len(grid) == 19
    assert list(grid) == sorted(grid)
    assert len(set(grid)) == len(grid)


def test_repository_config_declares_required_sections() -> None:
    """仓库发布的真实配置必须能被加载，且关键声明齐备。"""
    from ai_detector.config import PROJECT_ROOT

    config = ModelingConfig.from_yaml(PROJECT_ROOT / "configs" / "ieee_cis_baseline.yaml")

    assert config.work_package == "CW-003"
    assert config.target == "isFraud"
    assert config.join_key == "TransactionID"
    assert config.split_time == "TransactionDT"
    assert set(config.forbidden_features) == {"isFraud", "TransactionID", "TransactionDT"}
    assert len(config.allowlist_names()) == 431
    assert config.rule_fields() == ["TransactionAmt", "C13"]
    assert len(config.cost_scenarios()) == 3
    assert {item["name"] for item in config.cost_scenarios()} == {
        "fp_fn_1_1",
        "fp_fn_1_5",
        "fp_fn_1_20",
    }
    assert config.config_sha256 and len(config.config_sha256) == 64
    # 声明式阈值网格展开为 99 个固定阈值。
    assert len(config.threshold_grid()) == 99


def test_repository_config_has_no_absolute_paths_or_credentials() -> None:
    from ai_detector.config import PROJECT_ROOT

    text = (PROJECT_ROOT / "configs" / "ieee_cis_baseline.yaml").read_text(encoding="utf-8")

    for marker in ("C:\\", "D:\\", "/Users/", "/home/"):
        assert marker not in text
    for credential in ("password", "token", "cookie", "api_key", "secret"):
        assert credential not in text.lower()


def test_rule_baseline_requires_two_fields(ready) -> None:
    _, config_path, _ = ready
    raw = _load_config_mapping(config_path)
    raw["rule_baseline"]["fields"] = [raw["rule_baseline"]["fields"][0]]

    config = ModelingConfig.from_mapping(raw, config_path=config_path)
    from ai_detector.models.rule_baseline import RuleBaselineError, fit_rule_baseline

    with pytest.raises(RuleBaselineError, match="至少需要两个数值字段"):
        fit_rule_baseline(config.rule_baseline, pd.DataFrame({"AmtA": [1.0, 2.0]}))
