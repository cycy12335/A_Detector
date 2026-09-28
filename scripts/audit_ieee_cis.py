"""命令行入口：审计本地 IEEE-CIS 原始数据。

用法（项目根目录下执行）::

    python scripts/audit_ieee_cis.py --config configs/ieee_cis.yaml

原始数据必须由使用者本人同意 Kaggle 条款后手动放置到配置指定的目录。本脚本
不下载数据、不访问网络、不读取任何凭据；文件缺失时会列出缺失项与放置步骤并失败。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 让脚本在未执行 `pip install -e .` 时也能从源码树直接运行。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_detector.data.audit import run_audit  # noqa: E402
from ai_detector.data.ieee_cis import IeeeCisConfig, MissingDataError  # noqa: E402

DEFAULT_CONFIG = "configs/ieee_cis.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="审计本地 IEEE-CIS 原始数据并生成四类产物（不访问网络）。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(DEFAULT_CONFIG),
        help=f"配置文件路径（相对项目根目录），默认 {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--skip-row-partitions",
        action="store_true",
        help="不写出行级分区表（data/processed/ieee_cis/split_manifest_rows.csv）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = IeeeCisConfig.from_yaml(args.config)

    try:
        result = run_audit(config, write_row_partitions=not args.skip_row_partitions)
    except MissingDataError as error:
        print(str(error), file=sys.stderr)
        return 2

    report = result.report
    join_audit = report["join_audit"]
    split = report["split"]

    print(f"数据集：{report['dataset']['display_name']}（{report['dataset']['dataset_id']}）")
    print("文件指纹：")
    for name, info in report["files"].items():
        print(f"  {name}: {info['relative_path']} | {info['size_bytes']:,} 字节")
        print(f"    SHA-256 {info['sha256']}")
    print(
        f"transaction：{report['schema_audit']['transaction']['n_rows']:,} 行，"
        f"{report['schema_audit']['transaction']['n_columns']} 列"
    )
    print(
        f"identity：{join_audit['identity_rows']:,} 行，"
        f"覆盖率 {join_audit['identity_coverage_rate']:.4%}，"
        f"孤立键 {join_audit['orphan_identity_keys']:,}，"
        f"左连接保行 {join_audit['transaction_rows_preserved']}"
    )
    print(
        f"切分（{split['algorithm_version']}）：validation_start={split['boundaries']['validation_start']}，"
        f"test_start={split['boundaries']['test_start']}"
    )
    for name in ("train", "validation", "test"):
        print(
            f"  {name:<10} 行数={split['sizes'][name]:<8} "
            f"比例={split['actual_ratios'][name]:.4%} 标签率={split['label_rates'][name]:.4%}"
        )
    roles = report["feature_roles"]["by_role"]
    print(
        f"特征角色：候选特征 {len(roles['candidate_feature'])} 列，"
        f"排除 {len(roles['excluded'])} 列；"
        f"target/join_key/split_only = {roles['target']}/{roles['join_key']}/{roles['split_only']}"
    )
    print(f"耗时（秒）：{report['timing_seconds']}")
    print("产物：")
    for path in (
        result.audit_json,
        result.audit_markdown,
        result.column_profile_csv,
        result.split_manifest_json,
    ):
        print(f"  {config.relative(path)}")
    if result.row_partitions_csv is not None:
        print(f"  {config.relative(result.row_partitions_csv)}（行级分区表，已被 Git 忽略）")
    print("提醒：IEEE-CIS 是匿名公开竞赛数据，指标与结论不代表真实银行生产效果。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
