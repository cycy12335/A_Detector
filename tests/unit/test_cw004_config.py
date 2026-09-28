"""CW-004 版本化配置与冻结参数的测试。

断言对象是**仓库里的正式配置文件**，而不是夹具：这样"参数被悄悄改动"或
"配置里混进了绝对路径 / 凭据 / 搜索相关字段"都会在测试里失败。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from ai_detector.config import PROJECT_ROOT, ConfigError
from ai_detector.lightgbm_training import LightGbmConfig, REAL_SOURCE_FEATURES
from ai_detector.models.lightgbm_challenger import (
    FORBIDDEN_PARAM_KEYS,
    FROZEN_PARAM_KEYS,
    LightGbmSpecError,
)

CONFIG_PATH = PROJECT_ROOT / "configs" / "ieee_cis_lightgbm.yaml"

#: 首次真实训练前冻结的规格（与 CW-004 任务验收记录逐项一致）。
EXPECTED_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "random_state": 42,
    "n_jobs": 1,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}

#: CW-003 已验收配置与特征契约的指纹（CW-004 开工门禁 3、4）。
CW003_CONFIG_SHA256 = "2883c5bbf45f969fb84fe790d7b24f7a00db0ca9c7831e955a9419c0881a8368"
CW003_MANIFEST_SHA256 = "afb73c212c8838b1d940c8d594b1a1f730f27039137741ed65d4b2bd80e8d589"


@pytest.fixture(scope="module")
def raw_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def config() -> LightGbmConfig:
    return LightGbmConfig.from_yaml(CONFIG_PATH)


# --- 仓库配置 -----------------------------------------------------------


def test_repository_config_declares_required_sections(raw_config) -> None:
    for section in ("work_package", "seed", "cw003", "model", "determinism", "importance", "paths", "limits"):
        assert section in raw_config, f"配置缺少 {section}"
    assert raw_config["work_package"] == "CW-004"
    assert raw_config["seed"] == 42


def test_repository_config_freezes_the_exact_model_params(raw_config) -> None:
    params = raw_config["model"]["params"]
    assert params == EXPECTED_PARAMS
    assert set(params) == set(FROZEN_PARAM_KEYS)


def test_repository_config_has_no_search_or_gpu_parameters(raw_config) -> None:
    params = set(raw_config["model"]["params"])
    assert not (params & FORBIDDEN_PARAM_KEYS)
    text = CONFIG_PATH.read_text(encoding="utf-8").lower()
    for banned in ("optuna", "gridsearch", "randomizedsearch", "hyperparameter_search: true"):
        assert banned not in text, f"配置出现搜索相关字样：{banned}"
    assert raw_config["model"]["params"]["n_jobs"] == 1


def test_repository_config_declares_imbalance_and_early_stopping(raw_config) -> None:
    model = raw_config["model"]
    assert model["imbalance"]["strategy"] == "scale_pos_weight"
    assert model["imbalance"]["formula"] == "train_negative_count / train_positive_count"
    early = model["early_stopping"]
    assert early["validation_partition"] == "validation"
    assert early["metric"] == "binary_logloss"
    assert early["stopping_rounds"] == 50
    assert early["first_metric_only"] is True


def test_repository_config_points_at_accepted_cw003_artifacts(raw_config) -> None:
    cw003 = raw_config["cw003"]
    assert cw003["expected_config_sha256"] == CW003_CONFIG_SHA256
    assert cw003["expected_feature_manifest_sha256"] == CW003_MANIFEST_SHA256
    assert cw003["expected_kept_numeric"] == 194
    assert cw003["expected_kept_categorical"] == 5
    assert cw003["expected_transformed_dimension"] == 348
    assert cw003["expected_output_format"] == "sparse_csr"
    assert cw003["expected_kept_numeric"] + cw003["expected_kept_categorical"] == REAL_SOURCE_FEATURES
    assert cw003["prerequisite_required_status"] == "passed"
    assert cw003["baseline_keys"] == ["rule_tail_anomaly", "dummy", "logistic_regression"]


def test_repository_config_declares_determinism_tolerance_and_importance(raw_config) -> None:
    assert raw_config["determinism"]["absolute_tolerance"] == 1e-12
    assert raw_config["importance"]["types"] == ["gain", "split"]
    assert raw_config["importance"]["aggregation"] == "source_feature"


def test_repository_config_lists_forbidden_analyses_without_enabling_them(raw_config) -> None:
    """禁止的分析只能以「不做」清单的形式出现，模型段里不得有任何相关开关。"""
    forbidden = raw_config["importance"]["forbidden_analyses"]
    for banned in ("SHAP", "Permutation Importance", "Partial Dependence (PDP)", "概率校准"):
        assert banned in forbidden, banned

    model_section = yaml.safe_dump(raw_config["model"], allow_unicode=True).lower()
    for banned in ("shap", "permutation", "pdp", "calibrat"):
        assert banned not in model_section, f"模型参数段出现 {banned}"
    # 配置里不得出现任何"做校准 / 做解释"的开关键。
    assert "calibration" not in CONFIG_PATH.read_text(encoding="utf-8").lower()


def test_repository_config_has_no_absolute_paths_or_credentials() -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    assert "C:\\" not in text
    assert "D:\\" not in text
    assert "/Users/" not in text
    assert "/home/" not in text
    for credential in ("password", "token", "cookie", "api_key", "secret"):
        assert credential not in text.lower()

    declared: list[str] = []
    raw = yaml.safe_load(text)
    declared.extend(raw["cw003"]["required_artifacts"])
    declared.extend(
        [raw["cw003"][key] for key in ("config_path", "feature_manifest_path", "report_path", "prerequisite_task_status")]
    )
    declared.extend(raw["paths"]["outputs"].values())
    for value in declared:
        assert not Path(value).is_absolute(), value
        assert not str(value).startswith(("/", "\\")), value


def test_repository_config_loads_and_resolves_relative_to_root(config: LightGbmConfig) -> None:
    assert config.work_package == "CW-004"
    assert config.params == EXPECTED_PARAMS
    assert config.expected_source_features == REAL_SOURCE_FEATURES
    assert config.expected_transformed_dimension == 348
    assert config.absolute_tolerance == 1e-12
    for path in config.outputs.values():
        assert path.is_absolute()
        assert PROJECT_ROOT in path.parents
    assert config.config_sha256 == hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()


def test_repository_config_reuses_cw003_threshold_grid(config: LightGbmConfig) -> None:
    """99 个阈值网格、tie-break 与三个成本场景必须来自 CW-003，不得在 CW-004 重新定义。"""
    grid = config.modeling.threshold_grid()
    assert len(grid) == 99
    assert grid[0] == pytest.approx(0.01)
    assert grid[-1] == pytest.approx(0.99)
    assert config.modeling.thresholds["f1_tie_break"] == ["f1_desc", "recall_desc", "threshold_desc"]
    assert config.modeling.thresholds["cost_tie_break"] == ["cost_asc", "threshold_desc"]
    assert [scenario["name"] for scenario in config.modeling.cost_scenarios()] == [
        "fp_fn_1_1",
        "fp_fn_1_5",
        "fp_fn_1_20",
    ]
    assert config.modeling.forbidden_features == ("isFraud", "TransactionID", "TransactionDT")


# --- 配置校验本身 -------------------------------------------------------


def _mapping(**overrides) -> dict:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw.update(overrides)
    return raw


def _from_mapping(raw: dict) -> LightGbmConfig:
    return LightGbmConfig.from_mapping(raw, config_path=Path("probe.yaml"))


def test_absolute_path_in_config_is_rejected() -> None:
    raw = _mapping()
    raw["cw003"] = {**raw["cw003"], "config_path": "C:\\elsewhere\\config.yaml"}
    with pytest.raises(ConfigError, match="项目相对路径"):
        _from_mapping(raw)


def test_path_escaping_project_root_is_rejected() -> None:
    raw = _mapping()
    raw["paths"] = {"outputs": {**raw["paths"]["outputs"], "report_json": "../outside/report.json"}}
    with pytest.raises(ConfigError, match="逃出项目根目录"):
        _from_mapping(raw)


def test_credential_like_path_is_rejected() -> None:
    raw = _mapping()
    raw["paths"] = {"outputs": {**raw["paths"]["outputs"], "report_json": "artifacts/token.json"}}
    with pytest.raises(ConfigError, match="凭据"):
        _from_mapping(raw)


def test_extra_model_param_is_rejected() -> None:
    raw = _mapping()
    raw["model"] = {
        **raw["model"],
        "params": {**raw["model"]["params"], "scale_pos_weight": 5.0},
    }
    with pytest.raises(LightGbmSpecError, match="与冻结集合不一致"):
        _from_mapping(raw)


def test_wrong_validation_partition_is_rejected() -> None:
    raw = _mapping()
    raw["model"] = {
        **raw["model"],
        "early_stopping": {**raw["model"]["early_stopping"], "validation_partition": "test"},
    }
    with pytest.raises(ConfigError, match="validation_partition"):
        _from_mapping(raw)


def test_seed_must_match_random_state() -> None:
    raw = _mapping(seed=99)
    with pytest.raises(ConfigError, match="random_state"):
        _from_mapping(raw)


@pytest.mark.parametrize(
    "override",
    [
        {"learning_rate": 0.1},
        {"num_leaves": 63},
        {"n_estimators": 2000},
        {"min_child_samples": 5},
        {"subsample": 0.8},
        {"colsample_bytree": 0.8},
        {"reg_lambda": 1.0},
        {"max_depth": 5},
    ],
)
def test_deviating_frozen_values_are_rejected_by_the_loader(override: dict) -> None:
    """改一个超参数值就要同时改配置和代码；忘掉任何一处都会被拦下。"""
    raw = _mapping()
    raw["model"] = {**raw["model"], "params": {**raw["model"]["params"], **override}}
    with pytest.raises(LightGbmSpecError, match="冻结值"):
        _from_mapping(raw)


def test_importance_types_are_restricted() -> None:
    raw = _mapping()
    raw["importance"] = {**raw["importance"], "types": ["gain", "shap"]}
    with pytest.raises(ConfigError, match="gain, split"):
        _from_mapping(raw)


def test_missing_section_is_rejected() -> None:
    raw = _mapping()
    raw.pop("determinism")
    with pytest.raises(ConfigError, match="缺少字段"):
        _from_mapping(raw)


def test_wrong_work_package_is_rejected() -> None:
    with pytest.raises(ConfigError, match="work_package"):
        _from_mapping(_mapping(work_package="CW-005"))


def test_non_positive_tolerance_is_rejected() -> None:
    raw = _mapping()
    raw["determinism"] = {**raw["determinism"], "absolute_tolerance": 0.0}
    with pytest.raises(ConfigError, match="absolute_tolerance"):
        _from_mapping(raw)
