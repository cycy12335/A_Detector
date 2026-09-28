"""时间并列安全切分测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_detector.data.temporal_split import (
    PARTITIONS,
    TemporalSplitError,
    assert_split_is_consistent,
    assign_partition,
    build_split_manifest,
    compute_boundaries,
)


def make_inputs(
    *, n_per_time: int = 10, n_times: int = 20, seed: int = 0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """构造每个时间点含 ``n_per_time`` 笔交易的小数据集。"""
    rng = np.random.default_rng(seed)
    times = np.repeat(np.arange(1, n_times + 1), n_per_time)
    ids = np.arange(1000, 1000 + len(times))
    fraud = (rng.random(len(times)) < 0.1).astype(int)
    return pd.Series(ids), pd.Series(times), pd.Series(fraud)


def test_partitions_are_disjoint_and_complete() -> None:
    ids, times, fraud = make_inputs()
    manifest, rows = build_split_manifest(ids, times, fraud)

    assert set(rows["partition"]) == set(PARTITIONS)
    assert rows["TransactionID"].is_unique
    assert len(rows) == len(ids)
    assert set(rows["TransactionID"]) == set(ids)
    assert sum(manifest.sizes.values()) == len(ids)
    assert_split_is_consistent(rows)


def test_same_transaction_dt_never_spans_partitions() -> None:
    """时间并列保护：同一时间点的所有交易必须落在同一分区。"""
    ids, times, fraud = make_inputs(n_per_time=7, n_times=15)
    _, rows = build_split_manifest(ids, times, fraud)

    per_time = rows.groupby("TransactionDT")["partition"].nunique()
    assert (per_time == 1).all(), "存在跨分区的 TransactionDT"

    # 每个时间点 7 笔，若某分区被拆开，其行数就不会是 7 的整数倍。
    for name in PARTITIONS:
        assert (rows["partition"] == name).sum() % 7 == 0


def test_tie_protection_wins_over_exact_ratio() -> None:
    """并列保护优先于比例精确：粗粒度时间下比例会偏离，但不得拆开时间点。"""
    ids, times, fraud = make_inputs(n_per_time=25, n_times=12)
    manifest, rows = build_split_manifest(ids, times, fraud)

    assert set(rows["partition"]) == set(PARTITIONS)
    assert all(size > 0 for size in manifest.sizes.values())
    per_time = rows.groupby("TransactionDT")["partition"].nunique()
    assert (per_time == 1).all()


def test_ratios_are_close_to_target_when_times_are_fine_grained() -> None:
    """时间点足够多时，实际比例应精确命中目标。"""
    rng = np.random.default_rng(3)
    times = np.sort(rng.integers(1, 5000, size=2000))
    ids = np.arange(1, 2001)
    fraud = (rng.random(2000) < 0.05).astype(int)

    manifest, rows = build_split_manifest(pd.Series(ids), pd.Series(times), pd.Series(fraud))

    assert manifest.actual_ratios["train"] == pytest.approx(0.6)
    assert manifest.actual_ratios["validation"] == pytest.approx(0.2)
    assert manifest.actual_ratios["test"] == pytest.approx(0.2)
    assert_split_is_consistent(rows)


def test_unsorted_input_produces_identical_split() -> None:
    """输入乱序不得改变切分结果。"""
    ids, times, fraud = make_inputs()
    rng = np.random.default_rng(11)
    permutation = rng.permutation(len(ids))

    shuffled_manifest, shuffled_rows = build_split_manifest(
        ids.iloc[permutation].reset_index(drop=True),
        times.iloc[permutation].reset_index(drop=True),
        fraud.iloc[permutation].reset_index(drop=True),
    )
    sorted_manifest, sorted_rows = build_split_manifest(ids, times, fraud)

    assert shuffled_manifest.boundaries == sorted_manifest.boundaries
    assert shuffled_manifest.sizes == sorted_manifest.sizes
    pd.testing.assert_frame_equal(shuffled_rows, sorted_rows)


def test_repeated_runs_are_identical() -> None:
    ids, times, fraud = make_inputs()
    first = build_split_manifest(ids, times, fraud)
    second = build_split_manifest(ids, times, fraud)

    assert first[0].to_dict() == second[0].to_dict()
    pd.testing.assert_frame_equal(first[1], second[1])


def test_label_rates_are_reported_per_partition() -> None:
    ids, times, fraud = make_inputs()
    manifest, _ = build_split_manifest(ids, times, fraud)

    assert set(manifest.label_rates) == set(PARTITIONS)
    for rate in manifest.label_rates.values():
        assert 0.0 <= rate <= 1.0


def test_no_time_travel_between_partitions() -> None:
    ids, times, fraud = make_inputs()
    _, rows = build_split_manifest(ids, times, fraud)

    bounds = rows.groupby("partition")["TransactionDT"].agg(["min", "max"])
    assert bounds.loc["train", "max"] < bounds.loc["validation", "min"]
    assert bounds.loc["validation", "max"] < bounds.loc["test", "min"]


def test_assign_partition_uses_boundary_time_values() -> None:
    """边界值本身必须归入后一个分区（``<`` 语义），不允许含糊。"""
    from ai_detector.data.temporal_split import SplitBoundaries

    boundaries = SplitBoundaries(validation_start=10.0, test_start=20.0)
    values = pd.Series([9.0, 10.0, 19.0, 20.0, 21.0])

    assert assign_partition(values, boundaries).tolist() == [
        "train",
        "validation",
        "validation",
        "test",
        "test",
    ]


def test_duplicate_transaction_id_raises() -> None:
    ids, times, fraud = make_inputs()
    ids = ids.copy()
    ids.iloc[1] = ids.iloc[0]

    with pytest.raises(TemporalSplitError, match="重复"):
        build_split_manifest(ids, times, fraud)


def test_null_transaction_id_raises() -> None:
    ids, times, fraud = make_inputs()
    ids = ids.astype("float64")
    ids.iloc[0] = np.nan

    with pytest.raises(TemporalSplitError, match="TransactionID 存在空值"):
        build_split_manifest(ids, times, fraud)


def test_non_finite_time_raises() -> None:
    ids, times, fraud = make_inputs()
    times = times.astype("float64")
    times.iloc[0] = np.inf

    with pytest.raises(TemporalSplitError, match="非有限值"):
        build_split_manifest(ids, times, fraud)


def test_too_few_distinct_times_raises() -> None:
    ids = pd.Series([1, 2, 3, 4])
    times = pd.Series([5.0, 5.0, 5.0, 5.0])
    fraud = pd.Series([0, 1, 0, 1])

    with pytest.raises(TemporalSplitError, match="不足以切出三个非空分区"):
        build_split_manifest(ids, times, fraud)


def test_empty_input_raises() -> None:
    with pytest.raises(TemporalSplitError, match="输入为空"):
        build_split_manifest(pd.Series([], dtype=int), pd.Series([], dtype=float), pd.Series([], dtype=int))


@pytest.mark.parametrize(
    ("train_ratio", "validation_ratio"),
    [(0.0, 0.2), (0.6, 0.0), (0.8, 0.3), (1.0, 0.0)],
)
def test_invalid_ratios_raise(train_ratio: float, validation_ratio: float) -> None:
    with pytest.raises(TemporalSplitError):
        build_split_manifest(
            *make_inputs(), train_ratio=train_ratio, validation_ratio=validation_ratio
        )


def test_validator_detects_spanned_partition() -> None:
    """校验器必须能发现人为构造的"同一时间跨分区"，否则它保护不了切分。"""
    ids, times, fraud = make_inputs(n_per_time=4, n_times=12)
    _, rows = build_split_manifest(ids, times, fraud)

    broken = rows.copy()
    broken.loc[broken.index[0], "partition"] = "test"
    if broken.groupby("TransactionDT")["partition"].nunique().max() == 1:
        pytest.skip("构造的破坏未产生跨分区时间点")

    with pytest.raises(TemporalSplitError, match="跨分区"):
        assert_split_is_consistent(broken)


def test_compute_boundaries_is_pure_function_of_inputs() -> None:
    ids, times, fraud = make_inputs()
    del ids, fraud

    first = compute_boundaries(times, train_ratio=0.6, validation_ratio=0.2)
    second = compute_boundaries(times.to_numpy(), train_ratio=0.6, validation_ratio=0.2)

    assert first == second
