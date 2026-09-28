"""CW-004 LightGBM 规格、依赖缺失路径与推理 bundle 的单元测试。

覆盖 CW-004 任务验收要求的前三组边界：

- LightGBM 缺失时给出**清晰可操作**的错误，而不是在包导入阶段崩溃；
- 冻结参数被**逐项原样**传递，任何增删（含 GPU / 采样 / 搜索类参数）都直接失败；
- ``scale_pos_weight`` 只由传入的（即 train 的）标签计算；
- 推理 bundle 的列校验不会静默接受缺列或多列。
"""

from __future__ import annotations

import sys
from unittest import mock

import numpy as np
import pytest

from ai_detector.models.lightgbm_challenger import (
    FROZEN_PARAM_KEYS,
    FROZEN_PARAM_VALUES,
    INSTALL_HINT,
    LGBM_KEY,
    LightGbmBundle,
    LightGbmSpecError,
    LightGbmUnavailableError,
    booster_importance,
    build_lgbm_estimator,
    compute_scale_pos_weight,
    fit_lgbm_with_early_stopping,
    lightgbm_available,
    load_bundle,
    require_lightgbm,
    save_bundle,
    validate_frozen_params,
)

#: 与 `configs/ieee_cis_lightgbm.yaml` 完全一致的冻结参数（测试内独立写死一份，
#: 因此配置被改动时这里会失败，而不是跟着一起漂移）。
FROZEN_PARAMS = {
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


# --- 依赖缺失路径 -------------------------------------------------------


def test_missing_lightgbm_gives_actionable_error() -> None:
    """未安装 tree extra 时必须给出安装命令，而不是 ImportError 原始堆栈。"""
    with mock.patch.dict(sys.modules, {"lightgbm": None}):
        with pytest.raises(LightGbmUnavailableError) as error:
            require_lightgbm()
    message = str(error.value)
    assert "--only-binary=:all:" in message
    assert "lightgbm>=4.5,<5" in message
    assert '.[test,tree]' in message
    assert message.startswith("无法导入 lightgbm")
    assert INSTALL_HINT in message


def test_availability_probe_does_not_raise_when_missing() -> None:
    with mock.patch.dict(sys.modules, {"lightgbm": None}):
        assert lightgbm_available() is False
    assert lightgbm_available() is True, "tree extra 已安装：CW-004 测试不得因依赖缺失而跳过"


def test_cw003_modules_still_import_without_lightgbm() -> None:
    """延迟导入：未安装 lightgbm 时，CW-001 至 CW-003 的入口仍可导入。"""
    with mock.patch.dict(sys.modules, {"lightgbm": None}):
        import importlib

        for module_name in (
            "ai_detector.ieee_training",
            "ai_detector.features.ieee_cis",
            "ai_detector.lightgbm_training",
        ):
            importlib.import_module(module_name)
        from ai_detector.lightgbm_training import verify_prerequisite_gates  # noqa: F401


# --- 冻结参数 -----------------------------------------------------------


def test_frozen_param_key_set_is_exactly_the_declared_sixteen() -> None:
    assert set(FROZEN_PARAMS) == set(FROZEN_PARAM_KEYS)


def test_validate_frozen_params_returns_copy_and_accepts_spec() -> None:
    validated = validate_frozen_params(FROZEN_PARAMS)
    assert validated == FROZEN_PARAMS
    assert validated is not FROZEN_PARAMS


@pytest.mark.parametrize("removed", sorted(FROZEN_PARAMS))
def test_missing_param_is_rejected(removed: str) -> None:
    candidate = {key: value for key, value in FROZEN_PARAMS.items() if key != removed}
    with pytest.raises(LightGbmSpecError, match="与冻结集合不一致"):
        validate_frozen_params(candidate)


def test_extra_param_is_rejected() -> None:
    candidate = {**FROZEN_PARAMS, "scale_pos_weight": 12.0}
    with pytest.raises(LightGbmSpecError, match="与冻结集合不一致"):
        validate_frozen_params(candidate)


@pytest.mark.parametrize(
    "key",
    sorted(FROZEN_PARAM_VALUES),
)
def test_every_frozen_value_is_pinned(key: str) -> None:
    """除 random_state（由配置 seed 绑定）外，每个取值都被代码写死。"""
    expected = FROZEN_PARAM_VALUES[key]
    deviating = (not expected) if isinstance(expected, bool) else (
        expected + 1 if isinstance(expected, (int, float)) else f"{expected}_changed"
    )
    with pytest.raises(LightGbmSpecError, match="冻结值"):
        validate_frozen_params({**FROZEN_PARAMS, key: deviating})


@pytest.mark.parametrize(
    "override",
    [
        {"boosting_type": "dart"},
        {"objective": "multiclass"},
        {"n_jobs": 4},
        {"deterministic": False},
        {"force_col_wise": False},
    ],
)
def test_deviating_values_are_rejected(override: dict) -> None:
    with pytest.raises(LightGbmSpecError):
        validate_frozen_params({**FROZEN_PARAMS, **override})


def test_random_state_is_bound_to_the_config_seed_not_pinned_here() -> None:
    """random_state 由配置的 seed 校验，因此这里允许不同种子（夹具用 7）。"""
    assert "random_state" not in FROZEN_PARAM_VALUES
    validate_frozen_params({**FROZEN_PARAMS, "random_state": 7})


def test_frozen_params_are_passed_through_verbatim() -> None:
    """固定参数必须被逐项原样传给 LGBMClassifier；唯一额外参数是 scale_pos_weight。"""
    estimator = build_lgbm_estimator(FROZEN_PARAMS, scale_pos_weight=12.5)
    actual = estimator.get_params()
    for key, value in FROZEN_PARAMS.items():
        assert actual[key] == value, f"{key} 未被原样传递"
    assert actual["scale_pos_weight"] == 12.5


def test_estimator_has_no_early_stopping_constructor_argument() -> None:
    """早停不得走已废弃的构造参数，只能走 callback API。"""
    estimator = build_lgbm_estimator(FROZEN_PARAMS, scale_pos_weight=1.0)
    assert estimator.get_params().get("early_stopping_rounds") is None


def test_early_stopping_implementation_uses_callbacks() -> None:
    import inspect

    source = inspect.getsource(fit_lgbm_with_early_stopping)
    assert "callbacks=" in source
    assert "early_stopping(" in source
    assert "early_stopping_rounds=" not in source
    assert "eval_set=" not in source, "4.7.0 已弃用 eval_set，必须用 eval_X / eval_y"


# --- 早停诊断口径 -------------------------------------------------------


def _separable_problem(rows: int = 600, seed: int = 11):
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    labels = (rng.random(rows) < 0.25).astype(int)
    labels[0], labels[1] = 1, 0
    matrix = sp.random(rows, 12, density=0.5, format="csr", random_state=seed, dtype=np.float32)
    # 让前两列携带信号，保证真的会分裂并触发早停。
    matrix = matrix.tolil()
    matrix[:, 0] = labels * 2.0 + rng.normal(scale=0.1, size=rows)
    matrix[:, 1] = (1 - labels) * 2.0
    return matrix.tocsr().astype(np.float32), labels


def test_early_stopping_diagnostics_match_lightgbm_records() -> None:
    """诊断口径必须与 LightGBM 自己记录的值一致。

    这里守住一个真实缺陷：``evals_result_`` 的第 i 项是"第 i+1 棵树之后"的分数，
    而 ``best_iteration_`` 是 1 基的树序号，按 ``history[best_iteration]`` 取最优分数
    会整体偏后一位。同时 ``n_estimators_`` 是**保留**的树数，不是跑过的轮数。
    """
    matrix, labels = _separable_problem()
    split = 450
    # 用**真正的冻结参数**拟合一个很小的问题：早停会在几十轮内触发，
    # 既不需要改任何参数，也不会跑满 1000 棵树上限。
    estimator = build_lgbm_estimator(FROZEN_PARAMS, scale_pos_weight=3.0)
    diagnostics = fit_lgbm_with_early_stopping(
        estimator,
        matrix[:split],
        labels[:split],
        matrix[split:],
        labels[split:],
        metric="binary_logloss",
        stopping_rounds=10,
        first_metric_only=True,
    )

    history = list(estimator.evals_result_["valid_0"]["binary_logloss"])
    recorded = float(estimator.best_score_["valid_0"]["binary_logloss"])

    assert diagnostics.rounds_run == len(history)
    assert diagnostics.trees_kept == int(estimator.n_estimators_)
    assert diagnostics.best_iteration == int(estimator.best_iteration_)
    assert diagnostics.best_score == pytest.approx(recorded)
    assert diagnostics.best_score == pytest.approx(history[diagnostics.best_iteration - 1])
    assert diagnostics.rounds_run < diagnostics.n_estimators
    assert diagnostics.early_stopping_fired is True


def test_diagnostics_distinguish_rounds_run_from_trees_kept() -> None:
    """报告必须能区分"跑了几轮"与"保留几棵树"，否则读者会误读早停强度。"""
    matrix, labels = _separable_problem()
    payload = fit_lgbm_with_early_stopping(
        build_lgbm_estimator(FROZEN_PARAMS, scale_pos_weight=3.0),
        matrix[:450],
        labels[:450],
        matrix[450:],
        labels[450:],
        metric="binary_logloss",
        stopping_rounds=10,
        first_metric_only=True,
    ).to_dict()

    assert payload["n_estimators_cap"] == FROZEN_PARAMS["n_estimators"]
    assert "trees_trained" not in payload, "含糊的字段名不得再出现在报告里"
    assert payload["rounds_run"] >= payload["trees_kept"]


# --- scale_pos_weight ---------------------------------------------------


def test_scale_pos_weight_is_negative_over_positive() -> None:
    labels = np.array([0] * 7 + [1] * 3)
    assert compute_scale_pos_weight(labels) == pytest.approx(7 / 3)


def test_scale_pos_weight_accepts_a_single_vector_only() -> None:
    """接口只接受一个标签向量：结构上无法把 validation/test 混进来。"""
    import inspect

    parameters = list(inspect.signature(compute_scale_pos_weight).parameters)
    assert parameters == ["train_labels"]


@pytest.mark.parametrize("labels", [[0, 1, 2], [0.0, 1.5], [-1, 1]])
def test_scale_pos_weight_rejects_non_binary_labels(labels: list) -> None:
    with pytest.raises(LightGbmSpecError, match="必须严格属于"):
        compute_scale_pos_weight(np.asarray(labels))


def test_scale_pos_weight_rejects_all_negative_train() -> None:
    with pytest.raises(LightGbmSpecError, match="没有正例"):
        compute_scale_pos_weight(np.zeros(10, dtype=int))


# --- 重要性类型白名单 ---------------------------------------------------


@pytest.mark.parametrize("importance_type", ["shap", "permutation", "pdp", "gain_impurity"])
def test_only_gain_and_split_importance_allowed(importance_type: str) -> None:
    with pytest.raises(LightGbmSpecError, match="只允许 gain / split"):
        booster_importance(object(), importance_type)


def test_importance_requires_a_fitted_estimator() -> None:
    with pytest.raises(LightGbmSpecError, match="尚未拟合"):
        booster_importance(object(), "gain")


# --- 推理 bundle --------------------------------------------------------


def _bundle(**overrides) -> LightGbmBundle:
    payload = {
        "preprocessor": object(),
        "model": object(),
        "best_iteration": 42,
        "source_features": ["A", "B", "C"],
        "numeric_features": ["A", "B"],
        "categorical_features": ["C"],
        "transformed_feature_names": ["numeric__A", "numeric__B", "categorical__C_x"],
        "scale_pos_weight": 12.0,
        "frozen_thresholds": {
            "selected_f1_threshold": 0.35,
            "cost_scenarios": {"fp_fn_1_1": {"selected_threshold": 0.4}},
        },
        "fingerprints": {"cw003_config_sha256": "abc"},
        "model_params": dict(FROZEN_PARAMS),
        "config_sha256": "def",
    }
    payload.update(overrides)
    return LightGbmBundle(**payload)


def test_bundle_validate_accepts_a_well_formed_bundle() -> None:
    bundle = _bundle()
    bundle.validate()
    assert bundle.schema_version == LightGbmBundle.SCHEMA_VERSION


def test_bundle_requires_source_features() -> None:
    with pytest.raises(LightGbmSpecError, match="源特征清单"):
        _bundle(source_features=[]).validate()


def test_bundle_rejects_duplicate_source_features() -> None:
    with pytest.raises(LightGbmSpecError, match="重复"):
        _bundle(source_features=["A", "A"]).validate()


def test_bundle_requires_preprocessor_and_model() -> None:
    with pytest.raises(LightGbmSpecError, match="预处理器或模型"):
        _bundle(model=None).validate()


def test_bundle_column_check_returns_declared_order() -> None:
    bundle = _bundle()
    assert bundle.check_columns(["C", "A", "B"]) == ["A", "B", "C"]


def test_bundle_rejects_missing_columns() -> None:
    with pytest.raises(LightGbmSpecError, match="缺少"):
        _bundle().check_columns(["A", "B"])


def test_bundle_rejects_extra_columns() -> None:
    with pytest.raises(LightGbmSpecError, match="多出"):
        _bundle().check_columns(["A", "B", "C", "D"])


def test_bundle_rejects_duplicate_input_columns() -> None:
    with pytest.raises(LightGbmSpecError, match="重复列"):
        _bundle().check_columns(["A", "A", "B", "C"])


def test_bundle_roundtrip_preserves_metadata(tmp_path) -> None:
    bundle = _bundle()
    path = save_bundle(bundle, tmp_path / "nested" / "bundle.joblib")
    assert path.is_file()
    restored = load_bundle(path)
    assert restored.describe() == bundle.describe()
    assert restored.check_columns(["A", "B", "C"]) == ["A", "B", "C"]
    assert restored.frozen_thresholds == bundle.frozen_thresholds


def test_bundle_describe_declares_no_raw_data() -> None:
    described = _bundle().describe()
    assert described["contains_raw_data"] is False
    assert described["n_source_features"] == 3
    assert described["n_transformed_features"] == 3
    assert described["frozen_thresholds"]["selected_f1_threshold"] == 0.35
    assert described["frozen_thresholds"]["cost_scenario_thresholds"] == {"fp_fn_1_1": 0.4}


def test_load_bundle_rejects_foreign_payload(tmp_path) -> None:
    import joblib

    path = tmp_path / "foreign.joblib"
    joblib.dump({"not": "a bundle"}, path)
    with pytest.raises(LightGbmSpecError, match="bundle 类型不符"):
        load_bundle(path)


def test_bundle_key_is_stable() -> None:
    assert LGBM_KEY == "lightgbm"
