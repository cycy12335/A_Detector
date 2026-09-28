"""时间切分测试：不重叠、不重复、不发生时间穿越。"""

from __future__ import annotations

import pandas as pd
import pytest

from ai_detector.data.schema import TARGET_COLUMN, TIMESTAMP_COLUMN
from ai_detector.features.split import SplitError, assert_split_is_valid, time_based_split


def make_frame(n_rows: int = 100) -> pd.DataFrame:
    """构造时间严格递增的小型数据框，每 10 行一个正例。"""
    return pd.DataFrame(
        {
            "transaction_id": [f"TX{i:04d}" for i in range(n_rows)],
            TIMESTAMP_COLUMN: pd.date_range("2025-01-01", periods=n_rows, freq="min"),
            TARGET_COLUMN: [1 if i % 10 == 0 else 0 for i in range(n_rows)],
        }
    )


def test_split_has_expected_sizes() -> None:
    split = time_based_split(make_frame(100), train_ratio=0.6, validation_ratio=0.2)

    assert split.sizes() == {"train": 60, "validation": 20, "test": 20}


def test_split_preserves_chronological_order() -> None:
    """三个分区的整体顺序必须是训练 -> 验证 -> 测试。"""
    split = time_based_split(make_frame(100))

    assert split.train[TIMESTAMP_COLUMN].max() <= split.validation[TIMESTAMP_COLUMN].min()
    assert split.validation[TIMESTAMP_COLUMN].max() <= split.test[TIMESTAMP_COLUMN].min()


def test_split_has_no_index_overlap_and_no_lost_rows() -> None:
    """三份合起来必须恰好等于原数据，不重不漏。"""
    source = make_frame(97)
    split = time_based_split(source)

    combined = pd.concat([split.train, split.validation, split.test])
    assert len(combined) == len(source)
    assert set(combined["transaction_id"]) == set(source["transaction_id"])
    assert combined["transaction_id"].is_unique


def test_boundaries_are_reported_in_order() -> None:
    split = time_based_split(make_frame(100))
    boundaries = split.boundaries()

    assert boundaries["train_start"] < boundaries["train_end"]
    assert boundaries["train_end"] <= boundaries["validation_start"]
    assert boundaries["validation_end"] <= boundaries["test_start"]
    assert boundaries["test_start"] < boundaries["test_end"]


def test_fraud_rates_are_reported_per_split() -> None:
    split = time_based_split(make_frame(100))
    rates = split.fraud_rates()

    assert set(rates) == {"train", "validation", "test"}
    assert all(0.0 <= rate <= 1.0 for rate in rates.values())


def test_unsorted_input_is_sorted_before_splitting() -> None:
    """输入乱序时，切分仍必须先按时间排序，否则会把未来数据放进训练集。"""
    source = make_frame(100).sample(frac=1.0, random_state=0)
    split = time_based_split(source)

    assert split.train[TIMESTAMP_COLUMN].is_monotonic_increasing
    assert split.train[TIMESTAMP_COLUMN].max() <= split.test[TIMESTAMP_COLUMN].min()


def test_too_few_rows_raises() -> None:
    with pytest.raises(SplitError, match="过小"):
        time_based_split(make_frame(4))


@pytest.mark.parametrize(
    ("train_ratio", "validation_ratio"),
    [(0.0, 0.2), (0.6, 0.0), (0.8, 0.3), (1.0, 0.0)],
)
def test_invalid_ratios_raise(train_ratio: float, validation_ratio: float) -> None:
    with pytest.raises(SplitError):
        time_based_split(make_frame(100), train_ratio=train_ratio, validation_ratio=validation_ratio)


def test_empty_frame_raises() -> None:
    with pytest.raises(SplitError):
        time_based_split(make_frame(0))


def test_missing_timestamp_column_raises() -> None:
    with pytest.raises(SplitError, match="时间列"):
        time_based_split(pd.DataFrame({"value": [1, 2, 3]}))


def test_validator_detects_time_travel() -> None:
    """校验器必须能发现人为构造的时间穿越，否则它无法保护流程。"""
    source = make_frame(60)
    split = time_based_split(source, train_ratio=0.5, validation_ratio=0.25)
    broken = type(split)(
        train=split.validation,
        validation=split.train,
        test=split.test,
    )

    with pytest.raises(SplitError, match="时间穿越"):
        assert_split_is_valid(broken)


def test_validator_detects_row_count_mismatch() -> None:
    source = make_frame(60)
    split = time_based_split(source, train_ratio=0.5, validation_ratio=0.25)
    truncated = type(split)(train=split.train.iloc[:1], validation=split.validation, test=split.test)

    with pytest.raises(SplitError, match="不等于源数据行数"):
        assert_split_is_valid(truncated, source=source)
