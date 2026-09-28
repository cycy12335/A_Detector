"""交易数据的列契约。

集中声明“哪些列是模型输入、哪些列绝不能进入模型”，避免在多个模块中重复
字符串字面量导致的泄漏风险。

约定：
- ``transaction_id`` / ``user_id`` / ``device_id`` 是标识列，禁止作为特征。
- ``timestamp`` 是时间列，用于切分与排序，禁止作为特征。
- ``is_fraud`` 是目标列，只能作为标签，禁止作为特征。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd

TRANSACTION_ID_COLUMN = "transaction_id"
TIMESTAMP_COLUMN = "timestamp"
USER_ID_COLUMN = "user_id"
DEVICE_ID_COLUMN = "device_id"
TARGET_COLUMN = "is_fraud"

#: 标识列，禁止作为模型输入。
ID_COLUMNS: tuple[str, ...] = (
    TRANSACTION_ID_COLUMN,
    USER_ID_COLUMN,
    DEVICE_ID_COLUMN,
)

#: 数值型特征。
NUMERIC_FEATURES: tuple[str, ...] = ("amount", "transaction_count_1h")

#: 类别型特征。布尔标记按类别处理，走 OneHotEncoder 分支。
CATEGORICAL_FEATURES: tuple[str, ...] = ("is_new_device",)

#: 显式声明的模型输入列。新增特征必须同时修改这里和 ``validate_transactions``。
FEATURE_COLUMNS: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

#: 模拟数据 CSV 的完整列顺序。
ALL_COLUMNS: tuple[str, ...] = (
    TRANSACTION_ID_COLUMN,
    TIMESTAMP_COLUMN,
    USER_ID_COLUMN,
    DEVICE_ID_COLUMN,
    "amount",
    "transaction_count_1h",
    "is_new_device",
    TARGET_COLUMN,
)


class SchemaError(ValueError):
    """数据不符合列契约或取值约定时抛出。"""


def forbidden_feature_columns() -> frozenset[str]:
    """返回绝不允许出现在模型输入中的列名集合。"""
    return frozenset(ID_COLUMNS + (TIMESTAMP_COLUMN, TARGET_COLUMN))


def find_forbidden_features(columns: Iterable[str]) -> list[str]:
    """返回 ``columns`` 中混入的禁用列（按输入顺序）。"""
    forbidden = forbidden_feature_columns()
    return [column for column in columns if column in forbidden]


def assert_feature_contract(columns: Sequence[str] | None = None) -> None:
    """确认特征列表不含 ID、时间列和目标列。

    在构建预处理器前调用，使列声明错误尽早暴露，而不是等到指标异常才发现。
    """
    declared = tuple(FEATURE_COLUMNS) if columns is None else tuple(columns)
    offenders = find_forbidden_features(declared)
    if offenders:
        raise SchemaError(
            "特征列表包含禁止使用的列（目标列/ID列/时间列）：" + ", ".join(offenders)
        )


def validate_transactions(
    frame: pd.DataFrame,
    *,
    expected_columns: Sequence[str] | None = None,
    min_rows: int = 1,
    require_sorted: bool = True,
) -> None:
    """校验交易数据框的列、取值与时间顺序；不合法时抛出 :class:`SchemaError`。

    只做契约校验，不修改输入。
    """
    expected = tuple(ALL_COLUMNS if expected_columns is None else expected_columns)
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise SchemaError("缺少必需列：" + ", ".join(missing))

    if len(frame) < min_rows:
        raise SchemaError(f"数据行数 {len(frame)} 少于最小值 {min_rows}")

    for column in NUMERIC_FEATURES:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise SchemaError(f"数值特征 {column} 的 dtype 不是数值类型")

    for column in ID_COLUMNS:
        if frame[column].isna().any():
            raise SchemaError(f"标识列 {column} 不允许为空")

    if frame[TIMESTAMP_COLUMN].isna().any():
        raise SchemaError(f"时间列 {TIMESTAMP_COLUMN} 不允许为空")
    if not pd.api.types.is_datetime64_any_dtype(frame[TIMESTAMP_COLUMN]):
        raise SchemaError(f"时间列 {TIMESTAMP_COLUMN} 必须是 datetime64 类型")

    for column in (TARGET_COLUMN, "is_new_device"):
        values = set(pd.unique(frame[column].dropna()))
        if not values <= {0, 1}:
            raise SchemaError(f"列 {column} 的取值必须属于 {{0, 1}}，实际为 {sorted(values)}")
        if frame[column].isna().any():
            raise SchemaError(f"列 {column} 不允许为空")

    if require_sorted:
        timestamps = frame[TIMESTAMP_COLUMN]
        if not timestamps.is_monotonic_increasing:
            raise SchemaError(f"数据未按 {TIMESTAMP_COLUMN} 升序排列")
