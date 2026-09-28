"""CW-003 特征契约、训练期过滤与不泄漏边界的测试。

核心断言：**训练期决策只由 train 决定**。做法是扰动 validation/test 数据，然后确认
保留列清单、训练期统计与已拟合预处理器**完全不发生变化**。若实现里任何一处误用了
held-out 数据，这些断言会失败。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from ai_detector.data.modeling import (
    assemble_partitions,
    load_partition_lookup,
    ModelingContractError,
)
from ai_detector.features.ieee_cis import (
    EXCLUSION_REASONS,
    _track_distinct,
    accumulate_train_stats,
    assert_no_forbidden_features,
    assert_no_forbidden_transformed_names,
    build_feature_pipeline,
    build_preprocessor,
    decide_columns,
)
from tests.cw003_fixtures import CW003_CATEGORICAL, CW003_NUMERIC


@pytest.fixture
def prepared(cw003_fixture):
    """构造夹具并完成门禁 + 训练期列决策。"""
    _, config = cw003_fixture.build()
    lookup = load_partition_lookup(config)
    stats = accumulate_train_stats(config, lookup, chunk_size=64)
    decision = decide_columns(config, stats)
    return config, lookup, stats, decision


# --- 训练期过滤规则 -----------------------------------------------------


def test_every_filter_rule_fires_on_its_fixture_column(prepared) -> None:
    """夹具里每一列都为某条规则而设：规则失效时会立刻暴露。"""
    _, _, _, decision = prepared
    by_name = {item["name"]: item["reason_code"] for item in decision.excluded}

    assert by_name["NumE"] == "constant_in_train", "常量数值列必须被排除"
    assert by_name["CatConst"] == "constant_in_train", "常量类别列必须被排除"
    assert by_name["NumF"] == "all_null_in_train", "全缺失列必须被排除"
    assert by_name["NumD"] == "missing_rate_above_threshold", "高缺失列必须被排除"
    assert by_name["CatMissing"] == "missing_rate_above_threshold"
    assert by_name["CatHigh"] == "cardinality_above_threshold", "高基数类别列必须被排除"

    assert "AmtA" in decision.numeric
    assert "NumC" in decision.numeric, "10% 缺失的数值列应保留"
    assert "CatLow" in decision.categorical
    assert "id_cat" in decision.categorical
    assert "identity_present" not in decision.kept, "派生字段由装配阶段添加，不在允许清单内"


def test_excluded_entries_carry_reason_text(prepared) -> None:
    _, _, _, decision = prepared

    for item in decision.excluded:
        assert item["reason_code"] in EXCLUSION_REASONS
        assert item["reason"] == EXCLUSION_REASONS[item["reason_code"]]
        assert item["source_file"] in ("train_transaction.csv", "train_identity.csv")


# --- held-out 扰动不影响训练期决策 --------------------------------------


def _perturb_partition(config, lookup: pd.Series, partition: str) -> None:
    """破坏指定分区中的特征值（含非法/极端值），train 分区保持不变。"""
    path = config.paths["transaction"]
    frame = pd.read_csv(path)
    ids = lookup[lookup == partition].index
    mask = frame["TransactionID"].isin(ids)
    for column in CW003_NUMERIC:
        frame.loc[mask, column] = 1e9
    for column in CW003_CATEGORICAL:
        frame.loc[mask, column] = "NEVER_SEEN_CATEGORY"
    frame.to_csv(path, index=False)


def test_held_out_perturbation_does_not_change_decisions(cw003_fixture) -> None:
    """扰动 validation/test 后，保留列清单与训练期统计必须逐字段不变。"""
    config_path, config = cw003_fixture.build()
    lookup = load_partition_lookup(config)
    before_stats = accumulate_train_stats(config, lookup, chunk_size=64)
    before_decision = decide_columns(config, before_stats)

    for partition in ("validation", "test"):
        _perturb_partition(config, lookup, partition)

    after_stats = accumulate_train_stats(config, lookup, chunk_size=64)
    after_decision = decide_columns(config, after_stats)

    assert after_decision.numeric == before_decision.numeric
    assert after_decision.categorical == before_decision.categorical
    assert [item["name"] for item in after_decision.excluded] == [
        item["name"] for item in before_decision.excluded
    ]
    for name, stats in before_stats.items():
        assert after_stats[name].to_dict() == stats.to_dict(), name


def test_held_out_perturbation_does_not_change_fitted_preprocessor(cw003_fixture) -> None:
    """已拟合预处理器（中位数、众数、均值/方差、类别集合）不得受 held-out 影响。"""
    config_path, config = cw003_fixture.build()
    lookup = load_partition_lookup(config)
    decision = decide_columns(config, accumulate_train_stats(config, lookup, chunk_size=64))

    frames = _load_train_features(config, lookup, decision)
    pipeline = build_preprocessor(config, decision)
    pipeline.fit(frames, None)
    baseline = _preprocessor_fingerprint(pipeline)

    for partition in ("validation", "test"):
        _perturb_partition(config, lookup, partition)

    pipeline_after = build_preprocessor(config, decision)
    pipeline_after.fit(_load_train_features(config, lookup, decision), None)

    assert _preprocessor_fingerprint(pipeline_after) == baseline


def _load_train_features(config, lookup: pd.Series, decision) -> pd.DataFrame:
    """取 train 分区的保留列。

    刻意复用真实的装配函数（含 identity 左连接），而不是自己读一次 CSV：
    保留列里含 identity 列，手写读取会漏掉它们，测试就会在错误的前提上通过。
    """
    assembled = assemble_partitions(config, lookup, decision, chunk_size=64)
    return assembled.features("train", list(decision.kept)).reset_index(drop=True)


def _preprocessor_fingerprint(pipeline) -> dict:
    """提取预处理器的可学习状态，用于比较两次 fit 是否完全一致。"""
    numeric = pipeline.named_transformers_.get("numeric")
    categorical = pipeline.named_transformers_.get("categorical")
    fingerprint: dict = {}
    if numeric is not None:
        fingerprint["median"] = np.asarray(numeric.named_steps["imputer"].statistics_).tolist()
        fingerprint["mean"] = np.asarray(numeric.named_steps["scaler"].mean_).tolist()
        fingerprint["scale"] = np.asarray(numeric.named_steps["scaler"].scale_).tolist()
    if categorical is not None:
        encoder = categorical.named_steps["encoder"]
        fingerprint["categories"] = [list(map(str, values)) for values in encoder.categories_]
    return fingerprint


# --- 禁止字段硬校验 -----------------------------------------------------


def test_forbidden_features_rejected_in_source_columns() -> None:
    with pytest.raises(ModelingContractError, match="禁止字段"):
        assert_no_forbidden_features(
            ["AmtA", "isFraud"], ("isFraud", "TransactionID", "TransactionDT")
        )


@pytest.mark.parametrize(
    "name",
    [
        "numeric__TransactionDT",
        "categorical__isFraud_0",
        "numeric__TransactionID",
    ],
)
def test_forbidden_features_rejected_in_transformed_names(name: str) -> None:
    with pytest.raises(ModelingContractError, match="转换后的特征名"):
        assert_no_forbidden_transformed_names(
            ["numeric__AmtA", name], ("isFraud", "TransactionID", "TransactionDT")
        )


def test_allowlist_never_contains_forbidden_features(cw003_fixture) -> None:
    """配置层面的允许清单必须与禁止字段零交集。"""
    _, config = cw003_fixture.build()

    names = set(config.allowlist_names())
    assert not names & set(config.forbidden_features)
    assert "isFraud" not in names
    assert "TransactionID" not in names
    assert "TransactionDT" not in names


# --- 稀疏性与未知类别 ---------------------------------------------------


def test_one_hot_output_stays_sparse(prepared) -> None:
    """one-hot 必须保持稀疏：不得被稠密化。"""
    config, _, _, decision = prepared
    features = _load_train_features(config, load_partition_lookup(config), decision)

    transformed = build_preprocessor(config, decision).fit_transform(features)

    assert sparse.issparse(transformed), "设计矩阵必须保持稀疏（sparse_threshold=0.0）"
    assert transformed.format == "csr"


def test_unknown_categories_transform_safely(prepared) -> None:
    """未见过的类别在 transform 时必须安全处理，而不是报错或静默错位。"""
    config, _, _, decision = prepared
    pipeline = build_preprocessor(config, decision)
    pipeline.fit(_load_train_features(config, load_partition_lookup(config), decision), None)

    unseen = pd.DataFrame(
        {name: ["BRAND_NEW_CATEGORY"] for name in decision.categorical}
        | {name: [0.0] for name in decision.numeric}
    )
    transformed = pipeline.transform(unseen)

    assert transformed.shape[0] == 1
    assert np.isfinite(transformed.toarray()).all()


def test_missing_indicator_added_for_column_with_train_missing(prepared) -> None:
    """训练期有缺失的数值列应产生缺失指示器。"""
    config, _, _, decision = prepared
    features = _load_train_features(config, load_partition_lookup(config), decision)
    pipeline = build_preprocessor(config, decision)
    pipeline.fit(features, None)

    names = list(pipeline.get_feature_names_out())
    indicators = [name for name in names if "missingindicator" in name]
    assert indicators, "NumC 训练期有缺失，应产生缺失指示器"
    assert any("NumC" in name for name in indicators)


def test_missing_indicator_absent_for_columns_without_train_missing(prepared) -> None:
    """训练期无缺失的列不应产生缺失指示器。"""
    config, _, _, decision = prepared
    features = _load_train_features(config, load_partition_lookup(config), decision)
    pipeline = build_preprocessor(config, decision)
    pipeline.fit(features, None)

    names = [name for name in pipeline.get_feature_names_out() if "missingindicator" in name]
    for name in names:
        assert "NumC" in name or "id_" in name, f"意外产生指示器：{name}"


def test_preprocessing_only_fits_on_train(prepared) -> None:
    """填补与缩放参数必须等于训练分区自身的统计量。"""
    config, lookup, _, decision = prepared
    features = _load_train_features(config, lookup, decision)
    pipeline = build_preprocessor(config, decision)
    pipeline.fit(features, None)

    imputer = pipeline.named_transformers_["numeric"].named_steps["imputer"]
    expected = features[decision.numeric].median().to_numpy(dtype=float)
    np.testing.assert_allclose(imputer.statistics_, expected, rtol=1e-6)


def test_model_pipeline_is_single_persistable_object(prepared) -> None:
    """预处理与模型必须组成单一对象，保证推理复用训练期统计量。"""
    config, lookup, _, decision = prepared
    features = _load_train_features(config, lookup, decision)
    target = np.where(np.arange(len(features)) % 12 == 5, 1, 0)

    pipeline = build_feature_pipeline(
        config, decision, LogisticRegression(max_iter=200, random_state=0)
    )
    pipeline.fit(features, target)

    assert set(pipeline.named_steps) == {"preprocessor", "model"}
    scores = pipeline.predict_proba(features)[:, 1]
    assert scores.shape == (len(features),)
    assert np.isfinite(scores).all()


# --- 回归：identity 缺失率必须以完整训练交易为分母 ----------------------


def test_identity_missing_rate_uses_full_train_population(tmp_path) -> None:
    """identity 列的缺失率分母是**全部训练交易**，未匹配的连接键计为缺失。

    回归背景：旧实现只统计 identity 表中属于 train 的匹配行，把分母缩到了子表大小，
    导致 identity 列的真实缺失率被严重低估（真实数据上少算了约 72 个百分点），
    本应被 ``max_missing_rate`` 排除的列全部留在了特征清单里。
    """
    from tests.cw003_fixtures import Cw003FixtureBuilder

    # identity 只覆盖 20% 交易 -> 左连接后每个 identity 列至少有 80% 缺失。
    _, config = Cw003FixtureBuilder(tmp_path, identity_coverage=0.2).build()
    lookup = load_partition_lookup(config)
    stats = accumulate_train_stats(config, lookup, chunk_size=17)

    train_rows = config.expected_split_sizes["train"]
    train_ids = set(lookup[lookup == "train"].index)
    identity_ids = set(
        pd.read_csv(config.paths["identity"], usecols=[config.join_key])[config.join_key]
    )
    matched = len(train_ids & identity_ids)
    expected_missing_rate = 1.0 - matched / train_rows

    for name in ("id_num", "id_cat"):
        column = stats[name]
        assert column.n_rows == train_rows, (
            f"{name} 的分母应是全部训练交易 {train_rows}，实际 {column.n_rows}"
        )
        assert column.missing_rate == pytest.approx(expected_missing_rate, abs=1e-9), name
        # 覆盖率低于 1 - max_missing_rate 时，该列必然被缺失率规则排除。
        assert column.missing_rate > config.train_only_filters["max_missing_rate"]


def test_identity_columns_are_excluded_when_coverage_is_low(tmp_path) -> None:
    """覆盖率 20% 时，所有 identity 列都应按缺失率被排除，而不是留在特征清单里。"""
    from tests.cw003_fixtures import Cw003FixtureBuilder

    _, config = Cw003FixtureBuilder(tmp_path, identity_coverage=0.2).build()
    lookup = load_partition_lookup(config)
    decision = decide_columns(config, accumulate_train_stats(config, lookup, chunk_size=17))

    excluded = {item["name"]: item["reason_code"] for item in decision.excluded}
    assert excluded.get("id_num") == "missing_rate_above_threshold"
    assert excluded.get("id_cat") == "missing_rate_above_threshold"
    assert "id_num" not in decision.numeric
    assert "id_cat" not in decision.categorical


def test_identity_columns_kept_when_coverage_is_high(tmp_path) -> None:
    """全量覆盖时 identity 列不应被缺失率规则误伤。"""
    from tests.cw003_fixtures import Cw003FixtureBuilder

    _, config = Cw003FixtureBuilder(tmp_path, identity_coverage=1.0).build()
    lookup = load_partition_lookup(config)
    decision = decide_columns(config, accumulate_train_stats(config, lookup, chunk_size=17))

    assert "id_num" in decision.numeric
    assert "id_cat" in decision.categorical
    assert all(
        item["reason_code"] != "missing_rate_above_threshold"
        for item in decision.excluded
        if item["name"].startswith("id_")
    )


# --- 回归：有界唯一值集合在跨块重复时不得误判 --------------------------


def _make_stats(max_cardinality: int) -> tuple[dict, int]:
    """构造单列统计容器，用于直接驱动 _track_distinct。"""
    from ai_detector.features.ieee_cis import ColumnTrainStats

    stats = {
        "col": ColumnTrainStats(name="col", source_file="f", kind="categorical")
    }
    stats["col"]._distinct_values = set()
    return stats, max_cardinality + 1


def test_distinct_at_cap_with_cross_chunk_repeats_is_not_censored() -> None:
    """恰好等于上限、且跨分块大量重复时，必须报出真实唯一值数且**不**标记截断。

    回归背景：旧实现用"本块唯一值个数 > 剩余额度"直接判定截断，没有先扣除已在集合里的
    重复值。一个真实只有 3 类、却反复出现在多个分块里的列会被误报为
    ``n_distinct=cap``、``distinct_is_censored=True``，并被错误排除。
    """
    stats, cap = _make_stats(max_cardinality=4)  # cap = 5

    # 5 个不同的值（恰好等于 cap），分 4 块喂入，块间大量重复。
    chunks = [
        ["a", "b"],
        ["b", "c"],
        ["c", "d"],
        ["d", "e", "a", "b", "c"],
    ]
    for chunk in chunks:
        _track_distinct(stats["col"], pd.Series(chunk), cap)

    assert stats["col"].distinct_is_censored is False
    assert len(stats["col"]._distinct_values) == 5 == cap


def test_distinct_below_cap_with_cross_chunk_repeats_is_not_censored() -> None:
    """真实 3 类、上限 4（验收者给出的探针场景）：必须报 3 类、不截断、可保留。"""
    stats, cap = _make_stats(max_cardinality=3)  # cap = 4

    for chunk in [["a", "b", "c"], ["a", "b", "c"], ["c", "b", "a"], ["a", "a", "b"]]:
        _track_distinct(stats["col"], pd.Series(chunk), cap)

    assert stats["col"].distinct_is_censored is False
    assert len(stats["col"]._distinct_values) == 3
    # 3 <= max_cardinality(3)，因此不应因基数被排除。
    assert not (len(stats["col"]._distinct_values) > 3)


def test_distinct_over_cap_is_censored() -> None:
    """真实超过上限：必须标记截断（与"恰好等于上限"可区分）。"""
    stats, cap = _make_stats(max_cardinality=3)  # cap = 4

    for chunk in [["a", "b"], ["c", "d"], ["e"]]:  # 共 5 个不同值 > cap
        _track_distinct(stats["col"], pd.Series(chunk), cap)

    assert stats["col"].distinct_is_censored is True


def test_distinct_over_cap_detected_in_a_single_chunk() -> None:
    stats, cap = _make_stats(max_cardinality=2)  # cap = 3

    _track_distinct(stats["col"], pd.Series(["a", "b", "c", "d"]), cap)

    assert stats["col"].distinct_is_censored is True


def test_cross_chunk_repeats_do_not_consume_budget_end_to_end(tmp_path) -> None:
    """端到端：分块读取时，低基数类别列不得因跨块重复被判为超过基数上限。"""
    from tests.cw003_fixtures import Cw003FixtureBuilder

    # CatLow 真实 3 类；把上限设为 3，旧实现会在分块读取下误判为截断。
    _, config = Cw003FixtureBuilder(tmp_path, max_cardinality=3).build()
    lookup = load_partition_lookup(config)
    stats = accumulate_train_stats(config, lookup, chunk_size=17)

    catlow = stats["CatLow"]
    assert catlow.distinct_is_censored is False, "跨块重复不应触发截断"
    assert catlow.n_distinct == 3, f"CatLow 真实 3 类，实际报出 {catlow.n_distinct}"

    decision = decide_columns(config, stats)
    assert "CatLow" in decision.categorical, "恰好等于上限的列必须保留"


def test_chunk_size_does_not_change_distinct_counts(tmp_path) -> None:
    """分块大小不得改变唯一值计数与截断判定。"""

    from tests.cw003_fixtures import Cw003FixtureBuilder

    _, config = Cw003FixtureBuilder(tmp_path, max_cardinality=4).build()
    lookup = load_partition_lookup(config)

    reference = None
    for chunk_size in (7, 23, 64, 300):
        stats = accumulate_train_stats(config, lookup, chunk_size=chunk_size)
        snapshot = {
            name: (s.n_distinct, s.distinct_is_censored, s.n_rows, s.n_missing)
            for name, s in stats.items()
        }
        if reference is None:
            reference = snapshot
        else:
            assert snapshot == reference, f"chunk_size={chunk_size} 改变了统计结果"
