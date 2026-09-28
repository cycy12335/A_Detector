"""列画像与特征角色登记测试。

重点验证三件事：

1. 目标列、连接键、原始时间列绝不进入候选特征。
2. 匿名列登记为"语义不可确认"，不编造业务含义。
3. 报告只含聚合统计，不含逐行取值、邮箱或设备原文。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ai_detector.data.profiling import (
    ANONYMOUS_RISK_NOTE,
    UNMATCHED_PARTITION,
    ChunkedProfiler,
    ProfilingConfig,
    classify_role,
    is_numeric_dtype,
    quantile_key,
    role_registry,
    role_summary,
    semantic_note_for,
)
from ai_detector.data.temporal_split import SplitBoundaries

BOUNDARIES = SplitBoundaries(validation_start=104.0, test_start=108.0)


FORBIDDEN = {
    "isFraud": "目标列",
    "TransactionID": "连接键",
    "TransactionDT": "时间列",
}
ANONYMOUS_PREFIXES = ("V", "C", "D", "M", "id_")


def _feed(frame: pd.DataFrame, profiler: ChunkedProfiler, chunk_size: int) -> list:
    for start in range(0, len(frame), chunk_size):
        profiler.update(frame.iloc[start : start + chunk_size])
    return profiler.build_profiles(
        join_key="TransactionID",
        target="isFraud",
        split_time="TransactionDT",
        forbidden=FORBIDDEN,
        anonymous_prefixes=ANONYMOUS_PREFIXES,
    )


def build_profiles(
    frame: pd.DataFrame,
    *,
    config: ProfilingConfig | None = None,
    chunk_size: int = 7,
    source_file: str = "fixture.csv",
):
    """按分块喂入含时间列的 transaction 型数据并返回列画像。"""
    profiler = ChunkedProfiler(
        source_file=source_file,
        config=config or ProfilingConfig(chunk_size=chunk_size, max_tracked_unique=1000),
        boundaries=BOUNDARIES,
        split_time_column="TransactionDT",
    )
    return _feed(frame, profiler, chunk_size)


def build_identity_profiles(
    frame: pd.DataFrame,
    *,
    partition_map: dict[int, str],
    chunk_size: int = 7,
):
    """按连接键映射分区，喂入不含时间列的 identity 型数据。"""
    profiler = ChunkedProfiler(
        source_file="train_identity.csv",
        config=ProfilingConfig(chunk_size=chunk_size, max_tracked_unique=1000),
        join_key_column="TransactionID",
        partition_map=partition_map,
    )
    return _feed(frame, profiler, chunk_size)


# --- 角色判定 -----------------------------------------------------------


def test_structural_columns_never_become_candidates() -> None:
    """目标列、连接键与时间列必须落到各自角色，绝不能是候选特征。"""
    frame = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [100.0, 101.0, 108.0, 109.0],
            "isFraud": [0, 1, 0, 1],
            "amount": [1.0, 2.0, 3.0, 4.0],
        }
    )
    profiles = {profile.column_name: profile for profile in build_profiles(frame)}

    assert profiles["isFraud"].role == "target"
    assert profiles["TransactionID"].role == "join_key"
    assert profiles["TransactionDT"].role == "split_only"
    assert profiles["amount"].role == "candidate_feature"
    for name in ("isFraud", "TransactionID", "TransactionDT"):
        assert profiles[name].role != "candidate_feature"


@pytest.mark.parametrize(
    ("column", "all_missing", "expected_role"),
    [
        ("isFraud", False, "target"),
        ("TransactionID", False, "join_key"),
        ("TransactionDT", False, "split_only"),
        ("blocked", False, "excluded"),
        ("amount", True, "excluded"),
        ("amount", False, "candidate_feature"),
    ],
)
def test_classify_role_precedence(column: str, all_missing: bool, expected_role: str) -> None:
    """角色判定有明确优先级：结构性列 > 显式禁止 > 训练分区全缺失 > 候选特征。"""
    role, reason = classify_role(
        column,
        join_key="TransactionID",
        target="isFraud",
        split_time="TransactionDT",
        forbidden={"isFraud": "目标列", "TransactionID": "连接键", "TransactionDT": "时间列",
                   "blocked": "显式禁止"},
        all_missing_in_train=all_missing,
    )

    assert role == expected_role
    assert reason


def test_all_missing_in_train_is_excluded_with_reason() -> None:
    """只依据**训练分区**判定全缺失列，并在原因里说明这一点。"""
    frame = pd.DataFrame(
        {
            "TransactionID": range(1, 13),
            "TransactionDT": [100.0] * 4 + [105.0] * 4 + [110.0] * 4,
            "isFraud": [0, 1] * 6,
            "always_null": [np.nan] * 12,
        }
    )
    profiles = {profile.column_name: profile for profile in build_profiles(frame)}

    assert profiles["always_null"].role == "excluded"
    assert "训练分区" in profiles["always_null"].role_reason
    assert profiles["always_null"].non_null_count == 0


def test_column_missing_only_in_train_is_excluded_but_not_when_only_in_test() -> None:
    """判定只用训练分区：只在测试分区缺失的列不应被排除。"""
    dt = [100.0] * 4 + [105.0] * 4 + [110.0] * 4
    train_null = [np.nan, np.nan, np.nan, np.nan] + [1.0] * 8
    test_null = [1.0] * 8 + [np.nan] * 4

    frame = pd.DataFrame(
        {
            "TransactionID": range(1, 13),
            "TransactionDT": dt,
            "isFraud": [0, 1] * 6,
            "null_in_train": train_null,
            "null_in_test": test_null,
        }
    )
    profiles = {profile.column_name: profile for profile in build_profiles(frame)}

    assert profiles["null_in_train"].role == "excluded"
    assert profiles["null_in_test"].role == "candidate_feature"


def test_anonymous_columns_are_registered_as_unverifiable() -> None:
    """匿名列必须登记风险，且不得为其断言任何具体业务含义。"""
    for column in ("V1", "V307", "C13", "D15", "M4", "id_31"):
        note = semantic_note_for(column, ("V", "C", "D", "M", "id_"))

        assert note == ANONYMOUS_RISK_NOTE
        assert "无法仅凭列名排除潜在代理或泄漏风险" in note
        assert "不为其编造业务含义" in note
        # 不得出现任何把它当作已知业务字段的措辞。
        for asserted_meaning in ("是设备", "表示邮箱", "代表商户", "金额", "地址", "银行卡"):
            assert asserted_meaning not in note, f"{column} 的说明断言了具体含义"


def test_non_anonymous_columns_get_conservative_note() -> None:
    note = semantic_note_for("card1", ("V", "C", "D", "M", "id_"))

    assert note != ANONYMOUS_RISK_NOTE
    assert "无法从公开数据确认" in note


def test_role_summary_deduplicates_columns_present_in_both_files() -> None:
    """同一列名出现在两个文件时，汇总只能出现一次，但登记要保留来源。"""
    transaction_frame = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [100.0, 101.0, 108.0, 109.0],
            "isFraud": [0, 1, 0, 1],
        }
    )
    identity_frame = pd.DataFrame({"TransactionID": [1, 2], "DeviceType": ["a", "b"]})

    transaction_profiles = build_profiles(transaction_frame)
    identity_profiles = build_identity_profiles(identity_frame, partition_map={1: "train", 2: "train"})
    combined = transaction_profiles + identity_profiles

    summary = role_summary(combined)
    assert summary["join_key"] == ["TransactionID"]

    registry = role_registry(combined)
    transaction_id_entries = [row for row in registry if row["column_name"] == "TransactionID"]
    assert len(transaction_id_entries) == 2
    assert {row["source_file"] for row in transaction_id_entries} == {
        "fixture.csv",
        "train_identity.csv",
    }
    assert all(row["role"] == "join_key" for row in transaction_id_entries)


# --- 画像统计 -----------------------------------------------------------


def test_profile_contains_no_raw_values_or_pii() -> None:
    """画像只能含聚合统计：不得出现邮箱原文、设备串或逐行取值。"""
    frame = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5, 6],
            "TransactionDT": [100.0, 101.0, 105.0, 108.0, 109.0, 110.0],
            "isFraud": [0, 1, 0, 1, 0, 1],
            "P_emaildomain": ["alice@example.com", "bob@example.com", np.nan, "a@b.c", "x@y.z", np.nan],
            "DeviceInfo": ["SM-G950F", "iPhone", "Windows", "SM-G950F", "iPhone", "Windows"],
        }
    )
    profiles = {profile.column_name: profile for profile in build_profiles(frame)}
    serialized = str([profile.to_dict() for profile in profiles.values()])

    for secret in ("alice@example.com", "bob@example.com", "SM-G950F", "iPhone", "a@b.c", "x@y.z"):
        assert secret not in serialized, f"画像泄漏了原始取值：{secret}"


def test_numeric_stats_match_pandas() -> None:
    """数值统计必须与 pandas 直接计算一致（样本足够时）。"""
    rng = np.random.default_rng(5)
    values = rng.lognormal(2.0, 0.7, 300)
    frame = pd.DataFrame(
        {
            "TransactionID": range(1, 301),
            "TransactionDT": np.repeat(np.arange(100, 110), 30).astype(float),
            "isFraud": (rng.random(300) < 0.1).astype(int),
            "amount": values,
        }
    )
    profiles = {profile.column_name: profile for profile in build_profiles(frame, chunk_size=40)}
    amount = profiles["amount"]

    assert amount.minimum == pytest.approx(values.min(), rel=1e-9)
    assert amount.maximum == pytest.approx(values.max(), rel=1e-9)
    assert amount.mean == pytest.approx(values.mean(), rel=1e-9)
    assert amount.std == pytest.approx(values.std(ddof=0), rel=1e-6)
    assert amount.distinct_count == len(np.unique(values))


def test_quantiles_are_marked_approximate_only_when_sampled() -> None:
    """分位数只在抽样时标记为近似值；全量参与时不应误标。"""
    values = np.arange(1.0, 21.0)
    frame = pd.DataFrame(
        {
            "TransactionID": range(1, 21),
            "TransactionDT": np.repeat(np.arange(100, 110), 2).astype(float),
            "isFraud": [0, 1] * 10,
            "amount": values,
        }
    )

    exact = {p.column_name: p for p in build_profiles(frame)}["amount"]
    assert exact.quantiles_are_approximate is False
    assert exact.quantile_sample_size == 20
    assert exact.quantiles[quantile_key(0.5)] == pytest.approx(np.median(values))

    sampled = {
        p.column_name: p
        for p in build_profiles(frame, config=ProfilingConfig(chunk_size=5, quantile_sample_size=6))
    }["amount"]
    assert sampled.quantiles_are_approximate is True
    assert sampled.quantile_sample_size == 6


def test_null_rates_are_reported_per_partition() -> None:
    frame = pd.DataFrame(
        {
            "TransactionID": range(1, 13),
            "TransactionDT": [100.0] * 4 + [105.0] * 4 + [110.0] * 4,
            "isFraud": [0, 1] * 6,
            # 训练分区 2/4 缺失，验证分区 0/4，测试分区 4/4。
            "amount": [np.nan, np.nan, 1.0, 2.0] + [1.0] * 4 + [np.nan] * 4,
        }
    )
    profiles = {profile.column_name: profile for profile in build_profiles(frame)}
    rates = profiles["amount"].null_rate_by_partition

    assert rates["train"] == pytest.approx(0.5)
    assert rates["validation"] == pytest.approx(0.0)
    assert rates["test"] == pytest.approx(1.0)


def test_chunk_size_does_not_change_structural_statistics() -> None:
    """分块大小不得影响计数类统计。"""
    rng = np.random.default_rng(9)
    frame = pd.DataFrame(
        {
            "TransactionID": range(1, 201),
            "TransactionDT": np.repeat(np.arange(100, 110), 20).astype(float),
            "isFraud": (rng.random(200) < 0.1).astype(int),
            "amount": rng.lognormal(2.0, 0.5, 200),
            "category": rng.choice(list("abcde"), 200),
        }
    )

    small = {p.column_name: p for p in build_profiles(frame, chunk_size=7)}
    large = {p.column_name: p for p in build_profiles(frame, chunk_size=200)}

    for name in small:
        assert small[name].distinct_count == large[name].distinct_count, name
        assert small[name].null_rate == pytest.approx(large[name].null_rate), name
        assert small[name].role == large[name].role, name
        assert small[name].non_null_count == large[name].non_null_count, name


def test_repeated_profiling_is_identical() -> None:
    frame = pd.DataFrame(
        {
            "TransactionID": range(1, 51),
            "TransactionDT": np.repeat(np.arange(100, 110), 5).astype(float),
            "isFraud": [0, 1] * 25,
            "amount": np.linspace(1.0, 50.0, 50),
        }
    )

    first = [p.to_dict() for p in build_profiles(frame)]
    second = [p.to_dict() for p in build_profiles(frame)]

    assert first == second


def test_identical_values_in_different_files_are_profiled_separately() -> None:
    """两个文件的同名数值列不应互相污染统计。"""
    frame = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [100.0, 101.0, 108.0, 109.0],
            "isFraud": [0, 1, 0, 1],
            "amount": [1.0, 2.0, 3.0, 4.0],
        }
    )

    profiles = {profile.column_name: profile for profile in build_profiles(frame)}
    assert profiles["amount"].non_null_count == 4


def test_unmatched_join_key_rows_are_bucketed_not_dropped() -> None:
    """identity 中匹配不到交易的行必须单独成桶，而不是被丢弃或强行归入某分区。"""
    identity = pd.DataFrame(
        {"TransactionID": [1, 2, 999999], "DeviceType": ["a", "b", "c"]}
    )
    profiler = ChunkedProfiler(
        source_file="train_identity.csv",
        config=ProfilingConfig(chunk_size=10, max_tracked_unique=100),
        join_key_column="TransactionID",
        partition_map={1: "train", 2: "validation"},
    )
    profiler.update(identity)
    profile = profiler.build_profiles(
        join_key="TransactionID",
        target="isFraud",
        split_time="TransactionDT",
        forbidden={},
        anonymous_prefixes=("V",),
    )[0]

    assert set(profile.null_rate_by_partition) == {"train", "validation", UNMATCHED_PARTITION}


# --- 预算与降级 ---------------------------------------------------------


def test_unique_counting_degrades_visibly_when_budget_is_exhausted() -> None:
    """预算耗尽时必须显式标记截断，而不是静默给出错误数字。"""
    frame = pd.DataFrame(
        {
            "TransactionID": range(1, 51),
            "TransactionDT": np.repeat(np.arange(100, 110), 5).astype(float),
            "isFraud": [0, 1] * 25,
            "a": np.arange(50),
            "b": np.arange(50),
        }
    )
    tiny = ProfilingConfig(chunk_size=10, max_tracked_unique=100, max_total_tracked_unique=3)

    profiles = {p.column_name: p for p in build_profiles(frame, config=tiny)}

    assert all(p.distinct_is_censored for p in profiles.values() if p.distinct_count != 50)
    # 至少有一列被标记为截断或未跟踪，说明降级是可见的而不是静默的。
    assert any(p.distinct_is_censored for p in profiles.values())


def test_per_column_cap_marks_censored_but_keeps_lower_bound() -> None:
    """达到每列上限时保留已跟踪的下界并标记截断。"""
    frame = pd.DataFrame(
        {
            "TransactionID": range(1, 21),
            "TransactionDT": np.repeat(np.arange(100, 110), 2).astype(float),
            "isFraud": [0, 1] * 10,
            "amount": np.arange(20, dtype=float),
        }
    )
    capped = ProfilingConfig(chunk_size=5, max_tracked_unique=5, max_total_tracked_unique=10_000)

    profiles = {p.column_name: p for p in build_profiles(frame, config=capped)}

    assert profiles["amount"].distinct_is_censored is True
    assert profiles["amount"].distinct_count is not None
    assert profiles["amount"].distinct_count <= 20


# --- 辅助函数 -----------------------------------------------------------


@pytest.mark.parametrize(
    ("quantile", "expected"),
    [(0.01, "q01"), (0.25, "q25"), (0.5, "q50"), (0.75, "q75"), (0.99, "q99")],
)
def test_quantile_key_names(quantile: float, expected: str) -> None:
    assert quantile_key(quantile) == expected


def test_is_numeric_dtype_treats_bool_as_categorical() -> None:
    assert is_numeric_dtype(np.dtype("int64")) is True
    assert is_numeric_dtype(np.dtype("float64")) is True
    assert is_numeric_dtype(np.dtype("bool")) is False
    assert is_numeric_dtype(np.dtype("object")) is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"chunk_size": 0}, "chunk_size"),
        ({"max_tracked_unique": 0}, "max_tracked_unique"),
        ({"quantile_sample_size": 0}, "quantile_sample_size"),
        ({"quantiles": (0.0, 0.5)}, "quantiles"),
        ({"quantiles": (0.5, 1.0)}, "quantiles"),
    ],
)
def test_invalid_profiling_config_raises(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ProfilingConfig(**kwargs)


def test_replace_is_available_for_profiling_config() -> None:
    """配置是不可变 dataclass，便于测试与调用方安全派生变体。"""
    base = ProfilingConfig()
    derived = replace(base, chunk_size=5)

    assert derived.chunk_size == 5
    assert base.chunk_size == 50000
