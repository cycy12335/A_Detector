"""CW-004 源特征重要性聚合的单元测试。

覆盖 CW-004 任务验收要求：

- 数值缺失指示维度并入对应数值源列；类别 one-hot 维度并入对应类别源列；
- 聚合前后 gain / split 总量在声明容差内守恒，超差即报错；
- 输出表只含源列名与数值，**不含任何 one-hot 类别取值**；
- 排序键完全确定，重复运行顺序一致。

映射不是靠解析特征名字符串得来的，因此这里额外用"转换后特征名"做一次**独立交叉验证**：
名字里出现的源列必须与结构推算出的源列一致。两条路径都同意才说明映射可信。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ai_detector.evaluation.feature_importance import (
    CONSERVATION_ATOL,
    ImportanceAggregationError,
    aggregate_by_source,
    assert_conserved,
    build_importance_frame,
    transformed_dim_sources,
)

NUMERIC = ["N1", "N2", "N3"]
CATEGORICAL = ["C1", "C2"]
#: 只在这些列里注入缺失，从而产生缺失指示维度。
MISSING_COLUMN = "N2"
CATEGORY_VALUES = {"C1": ["x", "y"], "C2": ["p", "q", "r"]}


def _fixture_frame(rows: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(
        {
            "N1": rng.normal(size=rows),
            "N2": rng.normal(size=rows),
            "N3": rng.normal(size=rows),
            "C1": rng.choice(CATEGORY_VALUES["C1"], rows),
            "C2": rng.choice(CATEGORY_VALUES["C2"], rows),
        }
    )
    frame.loc[frame.index % 4 == 0, MISSING_COLUMN] = np.nan
    return frame


def _fitted_preprocessor() -> ColumnTransformer:
    """构造与 CW-003 ``build_preprocessor`` 同语义的预处理器（稀疏、带缺失指示器）。"""
    numeric_branch = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_branch = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_branch, NUMERIC),
            ("categorical", categorical_branch, CATEGORICAL),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )
    preprocessor.fit(_fixture_frame())
    return preprocessor


@pytest.fixture(scope="module")
def mapping() -> tuple[ColumnTransformer, list[str]]:
    preprocessor = _fitted_preprocessor()
    return preprocessor, transformed_dim_sources(preprocessor, NUMERIC, CATEGORICAL)


# --- 映射 ---------------------------------------------------------------


def test_mapping_length_matches_transformed_dimension(mapping) -> None:
    preprocessor, sources = mapping
    names = list(preprocessor.get_feature_names_out())
    assert len(sources) == len(names)
    # 3 数值 + 1 缺失指示 + 2 + 3 个 one-hot = 9
    assert len(sources) == 9


def test_missing_indicator_maps_to_its_numeric_source(mapping) -> None:
    preprocessor, sources = mapping
    names = list(preprocessor.get_feature_names_out())
    indicator_positions = [
        index for index, name in enumerate(names) if "missingindicator_" in name
    ]
    assert len(indicator_positions) == 1, "夹具只在一列注入缺失，应恰好产生一个指示维度"
    for index in indicator_positions:
        assert sources[index] == MISSING_COLUMN


def test_one_hot_dimensions_map_to_their_categorical_source(mapping) -> None:
    preprocessor, sources = mapping
    names = list(preprocessor.get_feature_names_out())
    for index, name in enumerate(names):
        if not name.startswith("categorical__"):
            continue
        expected = next(
            column for column in CATEGORICAL if name.split("__", 1)[1].startswith(f"{column}_")
        )
        assert sources[index] == expected


def test_mapping_agrees_with_transformed_feature_names(mapping) -> None:
    """独立交叉验证：结构推算出的源列必须出现在转换后特征名里。"""
    preprocessor, sources = mapping
    names = list(preprocessor.get_feature_names_out())
    for name, source in zip(names, sources):
        tail = name.split("__", 1)[-1]
        assert source in tail, f"{name} 的源列推算为 {source}，与特征名不符"


def test_mapping_covers_only_source_columns(mapping) -> None:
    _, sources = mapping
    assert set(sources) == set(NUMERIC) | set(CATEGORICAL)


def test_unfitted_preprocessor_is_rejected() -> None:
    with pytest.raises(ImportanceAggregationError, match="尚未拟合"):
        transformed_dim_sources(ColumnTransformer([]), NUMERIC, CATEGORICAL)


def test_mismatched_category_count_is_rejected() -> None:
    preprocessor = _fitted_preprocessor()
    with pytest.raises(ImportanceAggregationError, match="不符"):
        transformed_dim_sources(preprocessor, NUMERIC, CATEGORICAL[:1])


# --- 聚合与守恒 ---------------------------------------------------------


def test_aggregation_sums_each_dimension_into_its_source(mapping) -> None:
    _, sources = mapping
    values = np.arange(1.0, len(sources) + 1)
    aggregated = aggregate_by_source(sources, values)
    assert set(aggregated) == set(NUMERIC) | set(CATEGORICAL)
    # N2 是"被插补列 + 缺失指示"两维之和。
    indicator_index = [index for index, source in enumerate(sources) if source == MISSING_COLUMN]
    assert len(indicator_index) == 2
    assert aggregated[MISSING_COLUMN] == pytest.approx(sum(values[index] for index in indicator_index))


def test_aggregation_conserves_total(mapping) -> None:
    _, sources = mapping
    values = np.array([2.5, 0.0, 1e-3, 7.25, 0.5, 3.0, 1.0, 4.0, 9.0])
    result = assert_conserved(float(values.sum()), aggregate_by_source(sources, values), label="gain")
    assert result["conserved"] is True
    assert abs(result["absolute_difference"]) <= CONSERVATION_ATOL


def test_aggregation_length_mismatch_is_rejected() -> None:
    with pytest.raises(ImportanceAggregationError, match="长度不一致"):
        aggregate_by_source(["A", "B"], [1.0])


def test_conservation_violation_is_rejected() -> None:
    with pytest.raises(ImportanceAggregationError, match="不守恒"):
        assert_conserved(10.0, {"A": 3.0}, label="gain")


def test_aggregation_does_not_mutate_inputs(mapping) -> None:
    _, sources = mapping
    values = np.ones(len(sources))
    aggregate_by_source(sources, values)
    assert np.array_equal(values, np.ones(len(sources)))


# --- 输出表 -------------------------------------------------------------


def _frame(top_n: int | None = None) -> pd.DataFrame:
    preprocessor = _fitted_preprocessor()
    sources = transformed_dim_sources(preprocessor, NUMERIC, CATEGORICAL)
    gain = np.linspace(5.0, 1.0, len(sources))
    split = np.arange(len(sources), dtype=float)
    kinds = {**{name: "numeric" for name in NUMERIC}, **{name: "categorical" for name in CATEGORICAL}}
    return build_importance_frame(
        source_names=[*NUMERIC, *CATEGORICAL],
        per_dimension_sources=sources,
        gain_values=gain,
        split_values=split,
        kinds=kinds,
        top_n=top_n,
    )


def test_frame_covers_every_source_feature_once() -> None:
    frame = _frame()
    assert len(frame) == len(NUMERIC) + len(CATEGORICAL)
    assert set(frame["source_feature"]) == set(NUMERIC) | set(CATEGORICAL)
    assert list(frame["rank"]) == list(range(1, len(frame) + 1))


def test_frame_is_sorted_by_gain_descending() -> None:
    frame = _frame()
    gains = frame["aggregated_gain"].tolist()
    assert gains == sorted(gains, reverse=True)


def test_frame_ordering_is_deterministic() -> None:
    assert _frame().equals(_frame())


def test_frame_contains_no_category_values() -> None:
    """输出表不得出现任何 one-hot 类别取值或完整类别集合。"""
    frame = _frame()
    cells = set()
    for column in frame.columns:
        if frame[column].dtype == object or frame[column].dtype == bool:
            cells.update(str(value) for value in frame[column].tolist())
        else:
            cells.update(str(value) for value in frame[column].tolist())
    leaked = {
        value
        for values in CATEGORY_VALUES.values()
        for value in values
        if value in cells
    }
    assert not leaked, f"重要性表泄漏了类别取值：{sorted(leaked)}"
    # 类别**列名**是允许出现的（聚合单位就是源列）。
    assert set(frame["source_feature"]) >= set(CATEGORICAL)


def test_frame_top_n_truncates_without_changing_order() -> None:
    full = _frame()
    truncated = _frame(top_n=2)
    assert len(truncated) == 2
    assert truncated["source_feature"].tolist() == full["source_feature"].tolist()[:2]


def test_frame_rejects_unknown_mapped_source() -> None:
    with pytest.raises(ImportanceAggregationError, match="未知源列"):
        build_importance_frame(
            source_names=["A"],
            per_dimension_sources=["B"],
            gain_values=[1.0],
            split_values=[1.0],
            kinds={"A": "numeric"},
        )


def test_frame_gain_share_sums_to_one() -> None:
    frame = _frame()
    assert frame["gain_share"].sum() == pytest.approx(1.0)
