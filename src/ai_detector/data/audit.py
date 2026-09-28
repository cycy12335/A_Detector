"""审计编排与产物生成。

产出四类可再生产物：

```text
artifacts/reports/cw002_ieee_cis/
├── audit.json              # 机器可读，含指纹、Schema、连接、切分、角色、限制、环境版本
├── audit.md                # 面向面试官的可读摘要（区分事实 / 决策 / 无法确认项）
├── column_profile.csv      # 聚合列画像（每列一行，无逐行取值）
└── split_manifest.json     # 切分算法版本、边界、行数、比例、标签率
```

行级分区表写入 ``data/processed/ieee_cis/``（已被 Git 忽略），与上面的汇总产物分开：
汇总产物要能长期保存，行级表体积大且可随时重建。

环境版本复用 ``evaluation.report.environment_versions``：报告的环境字段只应有一处实现，
否则两个工作包会给出互相矛盾的环境记录。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ai_detector import __version__
from ai_detector.data.ieee_cis import (
    IeeeCisConfig,
    JoinAudit,
    MissingDataError,
    TransactionScan,
    audit_join,
    check_required_files,
    fingerprint_required_files,
    iter_chunks,
    scan_transactions,
)
from ai_detector.data.profiling import (
    ANONYMOUS_RISK_NOTE,
    UNMATCHED_PARTITION,
    ChunkedProfiler,
    ColumnProfile,
    quantile_key,
    role_registry,
    role_summary,
)
from ai_detector.data.temporal_split import (
    PARTITIONS,
    TemporalSplitManifest,
    assert_split_is_consistent,
    build_split_manifest,
)
from ai_detector.evaluation.report import environment_versions, jsonable

WORK_PACKAGE = "CW-002"
DATASET_ID = "ieee_cis_fraud_detection"

#: 数据与方法限制。必须与结果一起给出，避免读者把公开匿名数据当作生产数据。
DATA_LIMITATIONS: tuple[str, ...] = (
    "TransactionDT 是相对时间偏移，其参考起点未在公开数据中说明；本报告只保留原始数值，"
    "不把它转换为任何日历日期，也不推断采集周期。",
    "V*/C*/D*/M*/id_* 等列语义匿名，无法仅凭列名排除潜在代理变量或未知泄漏；"
    "本工作包只登记风险，不声称已完成泄漏排除。",
    "分位数来自固定种子的分块随机子样本，是近似值；均值与标准差由精确累计和得出。",
    "唯一值计数有每列上限与全局预算；达到上限时 distinct_is_censored 为 true，"
    "此时 distinct_count 只是下界，distinct_count 为 null 表示该列未跟踪。",
    "时间并列保护会占用整组时间点，因此实际分区比例可能与 60/20/20 目标略有偏差；"
    "边界安全优先于比例精确。",
    "identity 只覆盖部分交易，缺失 identity 不构成数据错误；本工作包不因 identity 缺失"
    "而删除或过滤任何交易。",
    "IEEE-CIS 是匿名化的公开竞赛数据，不能视为真实银行生产数据；"
    "其上的任何结论都不代表真实线上风控效果。",
    "数据受 Kaggle 竞赛规则约束，本仓库不包含、不转发任何逐行数据。",
)

#: 无法从公开匿名数据确认的事项。
UNVERIFIABLE_ITEMS: tuple[str, ...] = (
    "TransactionDT 的真实起始日历时间与采集时区。",
    "匿名列（V*、C*、D*、M*、id_*）的真实业务含义与生成机制。",
    "数据是否已完成去标识化、是否存在残余可识别信息。",
    "竞赛数据的采样方式，以及它与真实交易总体分布之间的偏差。",
    "identity 与 transaction 之间除 TransactionID 之外是否存在其他关联结构。",
    "历史欺诈标签的标注口径、漏标率与复核流程。",
)


@dataclass(frozen=True)
class AuditResult:
    """一次审计运行的产物位置与核心结论。"""

    audit_json: Path
    audit_markdown: Path
    column_profile_csv: Path
    split_manifest_json: Path
    row_partitions_csv: Path | None
    report: dict[str, Any]


def _profile_file(
    config: IeeeCisConfig,
    *,
    path: Path,
    source_file: str,
    boundaries: Any = None,
    split_time_column: str | None = None,
    join_key_column: str | None = None,
    partition_map: dict[int, str] | None = None,
) -> list[ColumnProfile]:
    """分块读取一个文件并产出其列画像。"""
    profiler = ChunkedProfiler(
        source_file=source_file,
        config=config.profiling,
        boundaries=boundaries,
        split_time_column=split_time_column,
        join_key_column=join_key_column,
        partition_map=partition_map,
    )
    for chunk in iter_chunks(path, chunk_size=config.profiling.chunk_size):
        profiler.update(chunk)

    return profiler.build_profiles(
        join_key=config.join_key,
        target=config.target,
        split_time=config.split_time,
        forbidden=config.forbidden_features,
        anonymous_prefixes=config.anonymous_prefixes,
    )


def _profile_to_row(profile: ColumnProfile, quantiles: tuple[float, ...]) -> dict[str, Any]:
    """把列画像摊平成 CSV 的一行。"""
    row: dict[str, Any] = {
        "column_name": profile.column_name,
        "source_file": profile.source_file,
        "original_dtype": profile.original_dtype,
        "is_numeric": profile.is_numeric,
        "role": profile.role,
        "role_reason": profile.role_reason,
        "semantic_note": profile.semantic_note,
        "total_rows": profile.total_rows,
        "non_null_count": profile.non_null_count,
        "null_rate": profile.null_rate,
        "non_finite_count": profile.non_finite_count,
        "distinct_count": profile.distinct_count,
        "distinct_is_censored": profile.distinct_is_censored,
        "minimum": profile.minimum,
        "maximum": profile.maximum,
        "mean": profile.mean,
        "std": profile.std,
        "quantiles_are_approximate": profile.quantiles_are_approximate,
        "quantile_sample_size": profile.quantile_sample_size,
    }
    for quantile in quantiles:
        key = quantile_key(quantile)
        row[key] = profile.quantiles.get(key)
    for partition in PARTITIONS:
        row[f"null_rate_{partition}"] = profile.null_rate_by_partition.get(partition)
    row[f"null_rate_{UNMATCHED_PARTITION}"] = profile.null_rate_by_partition.get(
        UNMATCHED_PARTITION
    )
    return row


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def _render_markdown(report: dict[str, Any]) -> str:
    """生成面向面试官的可读摘要。

    刻意按"已验证事实 / 设计决策 / 无法确认事项"三分，避免把推断写成事实。
    """
    dataset = report["dataset"]
    files = report["files"]
    join_audit = report["join_audit"]
    split = report["split"]
    roles = report["feature_roles"]["by_role"]

    lines: list[str] = [
        f"# IEEE-CIS 数据审计摘要（{report['work_package']}）",
        "",
        f"- 数据集：{dataset['display_name']}（`{dataset['dataset_id']}`）",
        f"- 数据提供方：{dataset['provider']}；竞赛组织方：{dataset['organizer']}",
        f"- 官方数据页：{dataset['official_data_url']}",
        f"- 官方规则页：{dataset['official_rules_url']}",
        f"- 生成时间（UTC）：{report['generated_at']}",
        f"- 生成工具版本：ai_detector {report['package_version']}",
        "",
        "## 已验证事实",
        "",
        "以下项目由本仓库代码在本地原始文件上直接计算得出，可用指纹追溯。",
        "",
        "### 文件指纹",
        "",
        "| 逻辑名 | 相对路径 | 字节数 | SHA-256 |",
        "| --- | --- | --- | --- |",
    ]
    for name, info in files.items():
        lines.append(
            f"| `{name}` | `{info['relative_path']}` | {info['size_bytes']:,} | `{info['sha256']}` |"
        )

    lines.extend(
        [
            "",
            "### Schema 与连接审计",
            "",
            f"- transaction 行数：{report['schema_audit']['transaction']['n_rows']:,}，"
            f"列数：{report['schema_audit']['transaction']['n_columns']}",
            f"- identity 行数：{join_audit['identity_rows']:,}",
            f"- identity 唯一键数：{join_audit['identity_unique_keys']:,}",
            f"- 被 identity 覆盖的交易数：{join_audit['matched_transaction_rows']:,}"
            f"（覆盖率 {join_audit['identity_coverage_rate']:.4%}）",
            f"- identity 孤立键数（无法匹配交易）：{join_audit['orphan_identity_keys']:,}",
            f"- 左连接后交易行数：{join_audit['rows_after_left_join']:,}"
            f"（不增不减：{join_audit['transaction_rows_preserved']}）",
            f"- 目标列取值：{report['schema_audit']['transaction']['observed_target_values']}",
            "",
            "### 时间切分",
            "",
            f"- 算法版本：`{split['algorithm_version']}`，策略：`{report['split']['strategy']}`",
            f"- 切分边界：validation_start = {split['boundaries']['validation_start']}，"
            f"test_start = {split['boundaries']['test_start']}（原始 TransactionDT 数值）",
            f"- 边界处并列交易数：{split['tie_groups_at_boundary']}",
            "",
            "| 分区 | 行数 | 实际比例 | 目标比例 | 标签率 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    targets = {
        "train": report["split"]["train_ratio_target"],
        "validation": report["split"]["validation_ratio_target"],
        "test": 1.0
        - report["split"]["train_ratio_target"]
        - report["split"]["validation_ratio_target"],
    }
    for name in PARTITIONS:
        lines.append(
            f"| {name} | {split['sizes'][name]:,} | {split['actual_ratios'][name]:.4%} | "
            f"{targets[name]:.0%} | {split['label_rates'][name]:.4%} |"
        )

    lines.extend(
        [
            "",
            "### 特征角色登记",
            "",
            f"- `target`：{roles.get('target', [])}",
            f"- `join_key`：{roles.get('join_key', [])}",
            f"- `split_only`：{roles.get('split_only', [])}",
            f"- `excluded`：{roles.get('excluded', [])}",
            f"- `candidate_feature`：{len(roles.get('candidate_feature', []))} 列"
            "（不在此工作包做取舍）",
            "",
            "目标列、连接键与原始时间列均已验证不在候选特征中。"
            "完整登记（含来源文件）见 `audit.json` 的 `feature_roles.registry` 与"
            " `column_profile.csv`；同一列名可能来自两个文件，"
            "例如 `TransactionID` 同时是两者的连接键。",
            "",
            "## 设计决策",
            "",
            "- **两遍分块读取**：第一遍只读 3 个结构性列完成 Schema、唯一性与连接审计，"
            "第二遍读全部列做聚合画像。内存占用与文件宽度基本解耦。",
            "- **时间并列安全切分**：边界取时间值而非行号，同一 `TransactionDT` 的所有交易"
            "必然进入同一分区；代价是实际比例可能与目标比例略有偏差。",
            "- **左连接为主表保行**：以 transaction 为主表，identity 仅左连接，"
            "不因 identity 缺失删除任何交易；identity 重复键会直接报错而不是放大行数。",
            "- **角色判定只用训练分区**：`excluded` 中的数据驱动规则只看训练分区的缺失率，"
            "不使用验证集或测试集的统计，避免把评估分布泄漏进特征选择。",
            "- **唯一值计数有界**：每列与全局都有上限，触顶时标记 `distinct_is_censored`，"
            "宁可报告下界也不静默给出错误数字。",
            "- **只输出聚合统计**：列画像不含任何逐行取值，也不输出邮箱、设备串等原始内容。",
            "",
            "## 无法从公开匿名数据确认的事项",
            "",
        ]
    )
    for item in report["unverifiable_items"]:
        lines.append(f"- {item}")

    lines.extend(["", "## 数据与方法限制", ""])
    for item in report["limitations"]:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## 环境版本",
            "",
            "| 组件 | 版本 |",
            "| --- | --- |",
        ]
    )
    for key, value in report["environment"].items():
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "> 本报告描述的是公开匿名竞赛数据的工程审计结果，不是生产风控系统，"
            "也不代表任何真实银行的线上效果。",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(config: IeeeCisConfig, *, write_row_partitions: bool = True) -> AuditResult:
    """执行完整审计并写出四类产物。

    Raises:
        MissingDataError: 原始文件缺失（由 :func:`check_required_files` 抛出）。
    """
    check_required_files(config)

    started = time.perf_counter()
    fingerprints = fingerprint_required_files(config)

    scan_started = time.perf_counter()
    scan, transaction_ids, transaction_dt, is_fraud = scan_transactions(config)
    scan_seconds = time.perf_counter() - scan_started

    join_started = time.perf_counter()
    join_audit: JoinAudit = audit_join(config, transaction_ids)
    join_seconds = time.perf_counter() - join_started

    split_started = time.perf_counter()
    manifest, row_partitions = build_split_manifest(
        pd.Series(transaction_ids),
        pd.Series(transaction_dt),
        pd.Series(is_fraud),
        train_ratio=config.train_ratio,
        validation_ratio=config.validation_ratio,
    )
    assert_split_is_consistent(row_partitions)
    split_seconds = time.perf_counter() - split_started

    profile_started = time.perf_counter()
    transaction_profiles = _profile_file(
        config,
        path=config.transaction_path,
        source_file=config.transaction_path.name,
        boundaries=manifest.boundaries,
        split_time_column=config.split_time,
    )
    partition_map = dict(
        zip(
            row_partitions["TransactionID"].to_numpy(dtype=np.int64).tolist(),
            row_partitions["partition"].tolist(),
            strict=True,
        )
    )
    identity_profiles = _profile_file(
        config,
        path=config.identity_path,
        source_file=config.identity_path.name,
        join_key_column=config.join_key,
        partition_map=partition_map,
    )
    profile_seconds = time.perf_counter() - profile_started

    profiles = transaction_profiles + identity_profiles
    roles = role_summary(profiles)

    # 结构性列绝不能出现在候选特征中——这里是最后的硬校验。
    forbidden_in_candidates = [
        column
        for column in (config.target, config.join_key, config.split_time)
        if column in roles["candidate_feature"]
    ]
    if forbidden_in_candidates:
        raise AssertionError(
            f"目标列/连接键/时间列混入候选特征：{forbidden_in_candidates}；角色登记存在漏洞"
        )

    report: dict[str, Any] = {
        "work_package": WORK_PACKAGE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "dataset": config.meta.to_dict(),
        # 每个文件只记录指纹类事实；耗时是分阶段测量的，单独放在 timing_seconds，
        # 不按文件摊派（两个文件被扫描多遍，摊派会给出误导性的单文件耗时）。
        "files": {
            name: fingerprint.to_dict() for name, fingerprint in fingerprints.items()
        },
        "schema_audit": {
            "transaction": scan.to_dict(),
            "required_columns": [config.join_key, config.split_time, config.target],
            "identity_required_columns": [config.join_key],
        },
        "join_audit": join_audit.to_dict(),
        "split": {
            "strategy": config.split_strategy,
            **manifest.to_dict(),
        },
        "feature_roles": {
            "by_role": roles,
            # 逐 (来源文件, 列) 的完整登记：TransactionID 等列同时存在于两个文件，
            # 只按列名汇总会丢失来源维度。
            "registry": role_registry(profiles),
            "forbidden_features": config.forbidden_features,
            "anonymous_columns_note": ANONYMOUS_RISK_NOTE,
            "anonymous_column_count": sum(
                1 for profile in profiles if profile.semantic_note == ANONYMOUS_RISK_NOTE
            ),
        },
        "profiling": {
            "row_count_transaction": len(transaction_profiles),
            "row_count_identity": len(identity_profiles),
            "chunk_size": config.profiling.chunk_size,
            "max_tracked_unique": config.profiling.max_tracked_unique,
            "max_total_tracked_unique": config.profiling.max_total_tracked_unique,
            "quantile_sample_size": config.profiling.quantile_sample_size,
            "quantiles_are_approximate": True,
        },
        "timing_seconds": {
            "scan_transaction": round(scan_seconds, 3),
            "join_audit": round(join_seconds, 3),
            "temporal_split": round(split_seconds, 3),
            "column_profiling": round(profile_seconds, 3),
            "total": round(time.perf_counter() - started, 3),
        },
        "limitations": list(DATA_LIMITATIONS),
        "unverifiable_items": list(UNVERIFIABLE_ITEMS),
        "environment": environment_versions(),
    }

    audit_json = _write_json(report, config.outputs["audit_json"])

    markdown_path = config.outputs["audit_markdown"]
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")

    profile_path = config.outputs["column_profile_csv"]
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_rows = [
        _profile_to_row(profile, config.profiling.quantiles) for profile in profiles
    ]
    pd.DataFrame(profile_rows).to_csv(profile_path, index=False)

    split_manifest_path = _write_json(
        {
            "work_package": WORK_PACKAGE,
            "dataset_id": config.meta.dataset_id,
            "algorithm_version": manifest.algorithm_version,
            "strategy": config.split_strategy,
            "file_fingerprints": {
                name: fingerprint.to_dict() for name, fingerprint in fingerprints.items()
            },
            "split_time_column": config.split_time,
            "split_time_note": (
                "TransactionDT 为相对时间偏移，参考起点未公开；"
                "边界以原始数值记录，不转换为日历日期。"
            ),
            "target_ratio": {
                "train": config.train_ratio,
                "validation": config.validation_ratio,
                "test": 1.0 - config.train_ratio - config.validation_ratio,
            },
            "boundaries": manifest.boundaries.to_dict(),
            "sizes": manifest.sizes,
            "actual_ratios": manifest.actual_ratios,
            "label_rates": manifest.label_rates,
            "n_distinct_times": manifest.n_distinct_times,
            "tie_groups_at_boundary": manifest.tie_groups_at_boundary,
            "assertions": {
                "partitions_non_empty": True,
                "transaction_ids_disjoint": True,
                "no_transaction_dt_spans_partitions": True,
                "no_time_travel_between_partitions": True,
            },
            "limitations": list(DATA_LIMITATIONS[:5]),
            "environment": environment_versions(),
        },
        config.outputs["split_manifest"],
    )

    row_partitions_csv: Path | None = None
    if write_row_partitions:
        processed_dir = config.outputs["processed_dir"]
        processed_dir.mkdir(parents=True, exist_ok=True)
        row_partitions_csv = processed_dir / "split_manifest_rows.csv"
        row_partitions.to_csv(row_partitions_csv, index=False)

    return AuditResult(
        audit_json=audit_json,
        audit_markdown=markdown_path,
        column_profile_csv=profile_path,
        split_manifest_json=split_manifest_path,
        row_partitions_csv=row_partitions_csv,
        report=report,
    )


# 便于调用方判断异常类型，无需再导入 ieee_cis 模块。
__all__ = [
    "AuditResult",
    "DATA_LIMITATIONS",
    "DATASET_ID",
    "MissingDataError",
    "UNVERIFIABLE_ITEMS",
    "WORK_PACKAGE",
    "TemporalSplitManifest",
    "TransactionScan",
    "run_audit",
]
