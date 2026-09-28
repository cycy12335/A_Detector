"""特征预处理管线。

所有可学习的预处理（缺失值填充、类别编码、数值缩放）都放在
``ColumnTransformer`` 内，因此只会 ``fit`` 在训练集上；对验证集和测试集只调用
``transform``，避免把验证/测试分布泄漏进训练过程。
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ai_detector.data.schema import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    assert_feature_contract,
)

#: 数值特征的填充策略：中位数对偏态金额分布比均值稳健。
NUMERIC_IMPUTE_STRATEGY = "median"


def build_preprocessor() -> ColumnTransformer:
    """构造数值分支（中位数填充 + 标准化）与类别分支（众数填充 + 独热编码）。

    先调用 :func:`assert_feature_contract`，使 ID 列、时间列或目标列被误写入
    特征列表时立即失败，而不是先产出可疑的高分结果。
    """
    assert_feature_contract()

    numeric_branch = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=NUMERIC_IMPUTE_STRATEGY)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_branch = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_branch, list(NUMERIC_FEATURES)),
            ("categorical", categorical_branch, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_features_target(
    frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """取出特征矩阵与标签向量。

    标识列、时间列和目标列都不会出现在 ``X`` 中；类别特征统一转为字符串，
    数值特征用 ``to_numeric`` 强转，无法解析的值变为缺失并交给填充器处理。
    """
    assert_feature_contract(feature_columns)

    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise KeyError("特征列缺失：" + ", ".join(missing))
    if target_column not in frame.columns:
        raise KeyError(f"目标列缺失：{target_column}")

    features = frame.loc[:, list(feature_columns)].copy()
    for column in NUMERIC_FEATURES:
        if column in features.columns:
            features[column] = pd.to_numeric(features[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        if column in features.columns:
            features[column] = features[column].astype("string")

    target = pd.to_numeric(frame[target_column], errors="raise").astype("int64")
    return features, target


def build_model_pipeline(model: Any) -> Pipeline:
    """把预处理与模型组合成一个可整体保存的 Pipeline。

    保存整条管线（而不是只保存模型）是必要的：推理时必须复用训练集拟合出的
    填充统计量与缩放参数。
    """
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", model)])
