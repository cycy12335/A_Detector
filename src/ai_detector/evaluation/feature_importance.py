"""把 LightGBM 的逐维重要性聚合回**源特征**。

## 为什么需要聚合

模型看到的不是 199 个源列，而是 348 个转换维度：数值列被插补并标准化，缺失的数值列还会
额外产生一个"缺失指示"维度；类别列被 one-hot 展开成若干维度。直接把 348 维重要性贴出去，
读者既看不懂 `categorical__M1_T`，也会把同一个源列的多个 one-hot 维度当成多个特征。

因此本模块把每个转换维度**映射回它唯一所属的源列**，再求和。映射不是靠解析特征名字符串，
而是靠**已拟合预处理器自身的结构**（``output_indices_``、``MissingIndicator.features_``、
``OneHotEncoder.categories_``），因此类别取值里含下划线之类的字符也不会映射错。

## 边界

- 只做 ``gain`` 与 ``split`` 两种重要性。**不做** SHAP、Permutation Importance、PDP，
  也不做概率校准。
- 输出只有源列名与聚合数值，**不含任何 one-hot 类别取值**、完整类别集合或逐行贡献。
- 重要性衡量的是"模型在训练数据上用了哪些列来降低损失"，**不是**因果关系、业务含义或
  合规解释；匿名列（``V*``/``C*``/``D*``/``M*``/``id_*``）不得据此被赋予业务语义。
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

#: 支持的稀疏分支名（与 CW-003 的 ``build_preprocessor`` 一致）。
NUMERIC_BRANCH = "numeric"
CATEGORICAL_BRANCH = "categorical"
MISSING_INDICATOR_PREFIX = "missingindicator_"

#: 聚合守恒容差。gain 是浮点求和，重新分组会带来结合律级别的微小误差。
CONSERVATION_ATOL = 1e-6
CONSERVATION_RTOL = 1e-12


class ImportanceAggregationError(ValueError):
    """重要性无法映射回源特征，或映射结果与预处理器输出维度不一致。"""


def transformed_dim_sources(
    preprocessor: Any,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> list[str]:
    """返回长度为转换维度的列表，第 ``i`` 项是第 ``i`` 个转换维度所属的源列名。

    结构依据（全部来自**已拟合**对象，不做字符串解析）：

    - ``ColumnTransformer.output_indices_`` 给出每个分支占用的维度区间；
    - 数值分支的输出顺序是"插补后的数值列（按输入顺序）+ 缺失指示列（按
      ``MissingIndicator.features_`` 的顺序）"；
    - 类别分支的输出顺序是"按输入列顺序，每列的 ``OneHotEncoder.categories_`` 依次展开"。

    Raises:
        ImportanceAggregationError: 预处理器未拟合、分支缺失或分支宽度与结构推算不符。
    """
    numeric_columns = list(numeric_columns)
    categorical_columns = list(categorical_columns)
    output_indices = getattr(preprocessor, "output_indices_", None)
    if not output_indices:
        raise ImportanceAggregationError("预处理器尚未拟合，无法建立维度到源列的映射")

    # ``transformers_`` 是 ``(name, transformer, columns)`` 三元组的列表（不是字典），
    # 且会包含 ``remainder`` 条目。
    try:
        fitted = {str(item[0]): item[1] for item in preprocessor.transformers_}
    except (AttributeError, TypeError, ValueError, IndexError) as error:  # pragma: no cover
        raise ImportanceAggregationError(f"无法读取预处理器的已拟合分支：{error}") from error

    n_dims = int(sum(block.stop - block.start for block in output_indices.values()))
    mapping: list[str | None] = [None] * n_dims

    def _block(name: str) -> slice:
        block = output_indices.get(name)
        if block is None:
            raise ImportanceAggregationError(f"预处理器缺少 {name} 分支")
        return block  # type: ignore[return-value]

    def _fill(block: slice, names: Sequence[str]) -> None:
        if block.stop - block.start != len(names):
            raise ImportanceAggregationError(
                f"分支宽度 {block.stop - block.start} 与推算的 {len(names)} 不一致"
            )
        for offset, source in enumerate(names):
            index = block.start + offset
            if mapping[index] is not None:
                raise ImportanceAggregationError(f"维度 {index} 被映射了两次")
            mapping[index] = source

    if NUMERIC_BRANCH in fitted:
        block = _block(NUMERIC_BRANCH)
        branch = fitted[NUMERIC_BRANCH]
        imputer = branch.named_steps["imputer"]
        indicator = getattr(imputer, "indicator_", None)
        indicator_sources: list[str] = []
        if indicator is not None:
            features = np.asarray(getattr(indicator, "features_", []), dtype=int)
            for position in features.tolist():
                if not 0 <= position < len(numeric_columns):
                    raise ImportanceAggregationError(
                        f"缺失指示器指向越界的数值列下标 {position}"
                    )
                indicator_sources.append(numeric_columns[position])
        _fill(block, [*numeric_columns, *indicator_sources])

    if CATEGORICAL_BRANCH in fitted:
        branch = fitted[CATEGORICAL_BRANCH]
        encoder = branch.named_steps["encoder"]
        categories = list(getattr(encoder, "categories_", []))
        if len(categories) != len(categorical_columns):
            raise ImportanceAggregationError(
                f"one-hot 编码器记录了 {len(categories)} 列，与输入的 "
                f"{len(categorical_columns)} 个类别列不符"
            )
        expanded = [
            source
            for source, values in zip(categorical_columns, categories)
            for _ in values
        ]
        _fill(_block(CATEGORICAL_BRANCH), expanded)

    unresolved = [index for index, value in enumerate(mapping) if value is None]
    if unresolved:
        raise ImportanceAggregationError(
            f"有 {len(unresolved)} 个转换维度未能映射回源特征，示例下标：{unresolved[:5]}"
        )
    return [str(value) for value in mapping]


def aggregate_by_source(
    source_names: Sequence[str], values: Sequence[float] | np.ndarray
) -> dict[str, float]:
    """按源列名求和。每个维度只属于一个源列，因此这是无损的重新分组。

    Raises:
        ImportanceAggregationError: 长度不一致。
    """
    if len(source_names) != len(values):
        raise ImportanceAggregationError(
            f"维度名与数值长度不一致：{len(source_names)} vs {len(values)}"
        )
    totals: dict[str, float] = {}
    for name, value in zip(source_names, values):
        totals[name] = totals.get(name, 0.0) + float(value)
    return totals


def assert_conserved(
    per_dimension_total: float,
    aggregated: dict[str, float],
    *,
    label: str,
    atol: float = CONSERVATION_ATOL,
    rtol: float = CONSERVATION_RTOL,
) -> dict[str, float]:
    """校验聚合前后总量守恒。

    Returns:
        ``{"per_dimension_total", "aggregated_total", "absolute_difference", "tolerance"}``。

    Raises:
        ImportanceAggregationError: 超出声明容差。
    """
    aggregated_total = float(sum(aggregated.values()))
    difference = aggregated_total - float(per_dimension_total)
    tolerance = max(atol, rtol * abs(float(per_dimension_total)))
    if not abs(difference) <= tolerance:
        raise ImportanceAggregationError(
            f"{label} 聚合前后总量不守恒：{aggregated_total} != {per_dimension_total} "
            f"（绝对差 {difference}，容差 {tolerance}）"
        )
    return {
        "per_dimension_total": float(per_dimension_total),
        "aggregated_total": aggregated_total,
        "absolute_difference": difference,
        "tolerance": tolerance,
        "conserved": True,
    }


def build_importance_frame(
    *,
    source_names: Sequence[str],
    per_dimension_sources: Sequence[str],
    gain_values: Sequence[float] | np.ndarray,
    split_values: Sequence[float] | np.ndarray,
    kinds: dict[str, str],
    top_n: int | None = None,
) -> pd.DataFrame:
    """构造聚合到源特征的重要性表（只含源列名与数值，不含类别取值）。

    排序键 ``(gain 降序, split 降序, 源列名升序)`` 全部确定，因此重复运行顺序一致。

    Raises:
        ImportanceAggregationError: 维度映射与源特征清单不一致。
    """
    known = set(source_names)
    unknown = sorted(set(per_dimension_sources) - known)
    if unknown:
        raise ImportanceAggregationError(f"维度映射出现未知源列：{unknown[:5]}")

    gain = aggregate_by_source(per_dimension_sources, gain_values)
    split = aggregate_by_source(per_dimension_sources, split_values)
    total_gain = float(sum(gain.values()))

    rows = [
        {
            "source_feature": name,
            "kind": kinds.get(name, "unknown"),
            "aggregated_gain": float(gain.get(name, 0.0)),
            "aggregated_split": int(round(float(split.get(name, 0.0)))),
            "gain_share": (float(gain.get(name, 0.0)) / total_gain) if total_gain else 0.0,
        }
        for name in source_names
    ]
    frame = pd.DataFrame(rows).sort_values(
        by=["aggregated_gain", "aggregated_split", "source_feature"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    frame = frame.reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    if top_n is not None:
        frame = frame.head(int(top_n)).reset_index(drop=True)
    return frame
