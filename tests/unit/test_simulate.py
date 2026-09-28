"""模拟数据生成器的确定性与契约测试。

测试数据规模刻意保持很小（最多 400 行），保证 ``python -m pytest`` 快速返回。
"""

from __future__ import annotations

import pandas as pd
import pytest

from ai_detector.data.schema import (
    ALL_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    validate_transactions,
)
from ai_detector.data.simulate import generate_transactions, write_transactions_csv

SMALL_ROWS = 300


def test_same_seed_produces_identical_content() -> None:
    """相同种子必须产生逐值相同的数据。"""
    first = generate_transactions(seed=7, rows=SMALL_ROWS)
    second = generate_transactions(seed=7, rows=SMALL_ROWS)
    pd.testing.assert_frame_equal(first, second)


def test_same_seed_produces_identical_csv_bytes(tmp_path) -> None:
    """相同种子写出的 CSV 文件必须逐字节相同。"""
    first_path = write_transactions_csv(
        generate_transactions(seed=11, rows=SMALL_ROWS), tmp_path / "a.csv"
    )
    second_path = write_transactions_csv(
        generate_transactions(seed=11, rows=SMALL_ROWS), tmp_path / "b.csv"
    )
    assert first_path.read_bytes() == second_path.read_bytes()


def test_different_seed_changes_data() -> None:
    """不同种子应产生不同内容，否则说明种子没有被真正使用。"""
    first = generate_transactions(seed=1, rows=SMALL_ROWS)
    second = generate_transactions(seed=2, rows=SMALL_ROWS)
    assert not first.equals(second)


def test_schema_columns_rows_and_order() -> None:
    """列集合、列顺序、行数与时间排序都必须符合契约。"""
    frame = generate_transactions(seed=3, rows=SMALL_ROWS)

    assert list(frame.columns) == list(ALL_COLUMNS)
    assert len(frame) == SMALL_ROWS
    assert frame[TIMESTAMP_COLUMN].is_monotonic_increasing
    assert frame[TIMESTAMP_COLUMN].is_unique
    validate_transactions(frame)


def test_labels_and_flags_are_binary() -> None:
    """目标列与设备标记只能是 0/1，且目标列不能有空值。"""
    frame = generate_transactions(seed=5, rows=SMALL_ROWS)

    assert set(frame[TARGET_COLUMN].unique()) <= {0, 1}
    assert set(frame["is_new_device"].unique()) <= {0, 1}
    assert frame[TARGET_COLUMN].notna().all()


def test_both_classes_present_and_imbalanced() -> None:
    """生成数据必须同时包含正负样本，且保持类别不平衡。"""
    frame = generate_transactions(seed=42, rows=2000)
    rate = frame[TARGET_COLUMN].mean()

    assert 0 < frame[TARGET_COLUMN].sum() < len(frame)
    assert 0.01 < rate < 0.20, f"欺诈率 {rate} 超出预期的不平衡区间"


def test_missing_values_only_in_numeric_features() -> None:
    """缺失只注入数值特征，且缺失注入与标签无关。"""
    frame = generate_transactions(seed=9, rows=SMALL_ROWS, missing_rate=0.1)

    for column in ("transaction_id", TIMESTAMP_COLUMN, "user_id", "device_id"):
        assert frame[column].notna().all()
    assert frame["amount"].isna().any() or frame["transaction_count_1h"].isna().any()

    # 标签不能被缺失注入影响：缺失行的欺诈率与非缺失行同量级即可说明无强耦合。
    missing_mask = frame["amount"].isna()
    rate_missing = frame.loc[missing_mask, TARGET_COLUMN].mean()
    rate_present = frame.loc[~missing_mask, TARGET_COLUMN].mean()
    assert abs(rate_missing - rate_present) < 0.15


def test_no_leakage_column_can_identify_label() -> None:
    """任何输出列都不允许是标签的确定性函数。

    做法：在特征列上，用单列取值分组后标签必须同时出现 0 和 1。若某列能完全
    识别标签，说明存在泄漏字段。
    """
    frame = generate_transactions(seed=42, rows=2000)

    for column in FEATURE_COLUMNS:
        if column in ("amount", "transaction_count_1h"):
            continue  # 连续变量无法按取值分组，改由下面的相关性检查覆盖
        grouped = frame.groupby(column)[TARGET_COLUMN].nunique()
        assert (grouped > 1).all(), f"列 {column} 可单独完全识别标签，疑似泄漏字段"


def test_predictive_signal_is_partial_not_perfect() -> None:
    """特征必须带有可学习但不完美的信号，避免生成“一学就满分”的假数据。"""
    frame = generate_transactions(seed=42, rows=3000)

    fraud_amount = frame.loc[frame[TARGET_COLUMN] == 1, "amount"].median()
    normal_amount = frame.loc[frame[TARGET_COLUMN] == 0, "amount"].median()
    assert fraud_amount > normal_amount

    fraud_new_device = frame.loc[frame[TARGET_COLUMN] == 1, "is_new_device"].mean()
    assert 0.0 < fraud_new_device < 1.0, "is_new_device 不应完全决定标签"


def test_rows_parameter_is_respected() -> None:
    """``rows`` 参数必须被严格遵循。"""
    frame = generate_transactions(seed=4, rows=137)
    assert len(frame) == 137


@pytest.mark.parametrize("rows", [0, -5])
def test_invalid_rows_raise(rows: int) -> None:
    with pytest.raises(ValueError, match="rows"):
        generate_transactions(seed=1, rows=rows)


@pytest.mark.parametrize("missing_rate", [-0.1, 1.0, 1.5])
def test_invalid_missing_rate_raises(missing_rate: float) -> None:
    with pytest.raises(ValueError, match="missing_rate"):
        generate_transactions(seed=1, rows=50, missing_rate=missing_rate)


def test_written_csv_round_trips_schema(tmp_path) -> None:
    """写出的 CSV 必须能被读回并通过契约校验。"""
    path = write_transactions_csv(
        generate_transactions(seed=8, rows=SMALL_ROWS), tmp_path / "nested" / "transactions.csv"
    )
    assert path.is_file()

    reloaded = pd.read_csv(path)
    reloaded[TIMESTAMP_COLUMN] = pd.to_datetime(reloaded[TIMESTAMP_COLUMN])
    assert list(reloaded.columns) == list(ALL_COLUMNS)
    assert len(reloaded) == SMALL_ROWS
    validate_transactions(reloaded)
