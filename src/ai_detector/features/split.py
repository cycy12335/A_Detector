"""按时间先后切分训练集、验证集和测试集。

本模块只提供时间切分。P0 阶段禁止随机切分：随机切分会让同一时间窗口内的
交易同时出现在训练集和测试集，从而掩盖真实场景中的时间外泛化差距。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ai_detector.data.schema import TARGET_COLUMN, TIMESTAMP_COLUMN


class SplitError(ValueError):
    """数据不足以切分或切分结果违反时间约束时抛出。"""


@dataclass(frozen=True)
class TimeSplit:
    """三个按时间先后排列、互不重叠的数据分区。"""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }

    def boundaries(self) -> dict[str, str | None]:
        """各分区的起止时间（ISO 8601）。空分区返回 ``None``。"""

        def edge(frame: pd.DataFrame, which: str) -> str | None:
            if frame.empty:
                return None
            value = frame[TIMESTAMP_COLUMN].min() if which == "start" else frame[TIMESTAMP_COLUMN].max()
            return value.isoformat()

        return {
            "train_start": edge(self.train, "start"),
            "train_end": edge(self.train, "end"),
            "validation_start": edge(self.validation, "start"),
            "validation_end": edge(self.validation, "end"),
            "test_start": edge(self.test, "start"),
            "test_end": edge(self.test, "end"),
        }

    def fraud_rates(self) -> dict[str, float]:
        return {
            "train": float(self.train[TARGET_COLUMN].mean()),
            "validation": float(self.validation[TARGET_COLUMN].mean()),
            "test": float(self.test[TARGET_COLUMN].mean()),
        }


def time_based_split(
    frame: pd.DataFrame,
    *,
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
    timestamp_column: str = TIMESTAMP_COLUMN,
) -> TimeSplit:
    """按时间升序做位置切分。

    先按时间排序，再按行数切分为前 ``train_ratio``、接着 ``validation_ratio``、
    其余为测试集。位置切分保证三个区间的行互不重叠。

    Args:
        frame: 交易数据框。
        train_ratio: 训练集比例。
        validation_ratio: 验证集比例。
        timestamp_column: 时间列名。

    Returns:
        :class:`TimeSplit`。

    Raises:
        SplitError: 数据为空、时间列缺失，或任一区间不足一行。
    """
    if timestamp_column not in frame.columns:
        raise SplitError(f"缺少时间列：{timestamp_column}")
    if frame.empty:
        raise SplitError("数据为空，无法切分")
    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise SplitError(
            f"切分比例非法：train={train_ratio}, validation={validation_ratio}"
        )

    ordered = frame.sort_values(timestamp_column, kind="stable").reset_index(drop=True)
    n_rows = len(ordered)
    n_train = int(n_rows * train_ratio)
    n_validation = int(n_rows * validation_ratio)

    if min(n_train, n_validation, n_rows - n_train - n_validation) < 1:
        raise SplitError(f"数据量 {n_rows} 过小，无法按当前比例切出三个非空区间")

    split = TimeSplit(
        train=ordered.iloc[:n_train].reset_index(drop=True),
        validation=ordered.iloc[n_train : n_train + n_validation].reset_index(drop=True),
        test=ordered.iloc[n_train + n_validation :].reset_index(drop=True),
    )
    assert_split_is_valid(split, source=ordered, timestamp_column=timestamp_column)
    return split


def assert_split_is_valid(
    split: TimeSplit,
    *,
    source: pd.DataFrame | None = None,
    timestamp_column: str = TIMESTAMP_COLUMN,
) -> None:
    """校验切分的三个不变量。

    1. 分区内时间单调不减。
    2. 分区之间不发生时间穿越：训练集最晚时间不晚于验证集最早时间，验证集同理。
    3. 三个分区行数之和等于 ``source`` 行数（若提供），即不重不漏。
    """
    ordered_frames = (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    )

    for name, part in ordered_frames:
        if part.empty:
            raise SplitError(f"分区 {name} 为空")
        if not part[timestamp_column].is_monotonic_increasing:
            raise SplitError(f"分区 {name} 未按时间升序排列")

    first_end = split.train[timestamp_column].max()
    second_start = split.validation[timestamp_column].min()
    second_end = split.validation[timestamp_column].max()
    third_start = split.test[timestamp_column].min()

    if first_end > second_start:
        raise SplitError(
            f"训练集与验证集发生时间穿越：train 最晚 {first_end} > validation 最早 {second_start}"
        )
    if second_end > third_start:
        raise SplitError(
            f"验证集与测试集发生时间穿越：validation 最晚 {second_end} > test 最早 {third_start}"
        )

    if source is not None:
        total = sum(len(part) for _, part in ordered_frames)
        if total != len(source):
            raise SplitError(f"分区行数之和 {total} 不等于源数据行数 {len(source)}")
