"""特征契约与预处理管线测试。

重点是两件事：

1. 目标列、ID 列和时间列绝不能进入模型输入。
2. 缺失值填充与缩放参数只能来自训练集——用验证集或测试集拟合统计量属于数据泄漏。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ai_detector.data.schema import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    ID_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    SchemaError,
    assert_feature_contract,
    find_forbidden_features,
)
from ai_detector.data.simulate import generate_transactions
from ai_detector.features.pipeline import (
    build_model_pipeline,
    build_preprocessor,
    split_features_target,
)

SMALL_ROWS = 300


def test_declared_features_exclude_target_ids_and_time() -> None:
    """静态声明层面就不允许禁用列出现在特征列表中。"""
    assert find_forbidden_features(FEATURE_COLUMNS) == []
    assert TARGET_COLUMN not in FEATURE_COLUMNS
    assert TIMESTAMP_COLUMN not in FEATURE_COLUMNS
    for column in ID_COLUMNS:
        assert column not in FEATURE_COLUMNS
    assert set(FEATURE_COLUMNS) == set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)


def test_feature_contract_rejects_forbidden_columns() -> None:
    for column in (TARGET_COLUMN, TIMESTAMP_COLUMN, "user_id", "device_id", "transaction_id"):
        with pytest.raises(SchemaError, match="禁止使用"):
            assert_feature_contract([*FEATURE_COLUMNS, column])


def test_extracted_features_and_target_are_correct() -> None:
    frame = generate_transactions(seed=42, rows=SMALL_ROWS)
    features, target = split_features_target(frame)

    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert target.name == TARGET_COLUMN
    assert len(features) == len(target) == len(frame)
    assert set(target.unique()) <= {0, 1}
    assert find_forbidden_features(features.columns) == []


def test_categorical_feature_is_cast_to_string() -> None:
    """类别特征要交给 OneHotEncoder，因此必须是字符串 dtype。"""
    frame = generate_transactions(seed=42, rows=SMALL_ROWS)
    features, _ = split_features_target(frame)

    assert all(
        pd.api.types.is_string_dtype(features[column]) for column in CATEGORICAL_FEATURES
    )


def test_unparseable_numeric_becomes_missing_not_crash() -> None:
    """无法解析的数值应变为缺失交给填充器，而不是抛错或变成脏值。"""
    frame = generate_transactions(seed=42, rows=SMALL_ROWS)
    # pandas 3 不允许把字符串直接写入 float64 列，因此先转为 object 模拟脏数据来源。
    frame["amount"] = frame["amount"].astype(object)
    frame.loc[0, "amount"] = "not-a-number"

    features, _ = split_features_target(frame)

    assert pd.isna(features.loc[0, "amount"])


def test_missing_required_column_raises() -> None:
    frame = generate_transactions(seed=42, rows=SMALL_ROWS).drop(columns=["amount"])
    with pytest.raises(KeyError, match="amount"):
        split_features_target(frame)


def test_preprocessor_output_shape_and_no_forbidden_names() -> None:
    """变换后的特征名必须仍然不含禁用列。"""
    frame = generate_transactions(seed=42, rows=SMALL_ROWS)
    features, _ = split_features_target(frame)
    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(features)
    names = list(preprocessor.get_feature_names_out())

    assert transformed.shape[0] == len(features)
    # 2 个数值特征 + is_new_device 的 2 个独热列
    assert transformed.shape[1] == len(NUMERIC_FEATURES) + 2
    assert find_forbidden_features(names) == []


def test_imputer_and_scaler_are_fitted_on_train_only() -> None:
    """核心泄漏检查：填充统计量与缩放参数必须等于训练集自身的统计量。

    做法是在训练集上 fit，然后直接读取管线内部的统计量，与训练集独立计算的结果
    比对。若中途混入了验证集/测试集数据，这两个数值不会相等。
    """
    train = generate_transactions(seed=42, rows=600).iloc[:400]
    held_out = generate_transactions(seed=99, rows=400)

    features_train, _ = split_features_target(train)
    features_held_out, _ = split_features_target(held_out)

    preprocessor = build_preprocessor()
    preprocessor.fit(features_train)
    preprocessor.transform(features_held_out)  # 只允许 transform

    numeric_branch = preprocessor.named_transformers_["numeric"]
    imputer = numeric_branch.named_steps["imputer"]
    scaler = numeric_branch.named_steps["scaler"]

    # 独立复算一遍“训练集 -> 中位数填充 -> 标准化”，不使用 sklearn 的中间结果。
    raw_train = features_train[list(NUMERIC_FEATURES)].to_numpy(dtype=float)
    expected_median = np.nanmedian(raw_train, axis=0)
    imputed_train = np.where(np.isnan(raw_train), expected_median, raw_train)
    expected_mean = imputed_train.mean(axis=0)
    expected_scale = imputed_train.std(axis=0, ddof=0)

    np.testing.assert_allclose(imputer.statistics_, expected_median, rtol=1e-9)
    np.testing.assert_allclose(scaler.mean_, expected_mean, rtol=1e-9)
    np.testing.assert_allclose(scaler.scale_, expected_scale, rtol=1e-9)

    # 反向确认：held-out 的统计量与训练集统计量确实不同，否则上面的断言无意义。
    raw_held_out = features_held_out[list(NUMERIC_FEATURES)].to_numpy(dtype=float)
    assert not np.allclose(expected_median, np.nanmedian(raw_held_out, axis=0))


def test_model_pipeline_is_a_single_fittable_object() -> None:
    frame = generate_transactions(seed=42, rows=SMALL_ROWS)
    features, target = split_features_target(frame)

    pipeline = build_model_pipeline(LogisticRegression(max_iter=200, random_state=0))
    pipeline.fit(features, target)

    probabilities = pipeline.predict_proba(features)[:, 1]
    assert probabilities.shape == (len(features),)
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
    assert set(pipeline.named_steps) == {"preprocessor", "model"}
