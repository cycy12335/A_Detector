"""命令行入口：在真实 IEEE-CIS 数据上运行 CW-003 三类基线实验。

用法（项目根目录下执行）::

    python scripts/train_ieee_baselines.py --config configs/ieee_cis_baseline.yaml

前置条件（开工门禁，任一不满足即以退出码 2 失败并提示置为 blocked）：

1. CW-002 的两个原始文件存在且 SHA-256 与 CW-002 审计记录一致；
2. CW-002 的行级切分清单存在且行数、分区规模、边界一致。

本脚本不下载数据、不访问网络、不读取任何凭据；不做超参数搜索；不使用 test 选择任何内容。
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

from ai_detector.data.modeling import (  # noqa: E402
    DataGateError,
    ModelingConfig,
    ModelingContractError,
)
from ai_detector.ieee_training import run_training  # noqa: E402

DEFAULT_CONFIG = "configs/ieee_cis_baseline.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在真实 IEEE-CIS 数据上运行 CW-003 的规则 / Dummy / Logistic Regression 基线。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(DEFAULT_CONFIG),
        help=f"配置文件路径（相对项目根目录），默认 {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="读取分块行数；默认使用配置/代码内的固定值，仅影响内存与耗时，不影响结论",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = ModelingConfig.from_yaml(args.config)
    except Exception as error:  # noqa: BLE001 - 配置错误需要原样暴露给使用者
        print(f"配置加载失败：{error}", file=sys.stderr)
        return 2

    kwargs = {} if args.chunk_size is None else {"chunk_size": args.chunk_size}
    try:
        result = run_training(config, **kwargs)
    except DataGateError as error:
        print(str(error), file=sys.stderr)
        print(
            "请把任务状态置为 blocked：不得替换文件、重算切分或用 fixture 冒充真实实验。",
            file=sys.stderr,
        )
        return 2
    except ModelingContractError as error:
        print(f"数据契约校验失败：{error}", file=sys.stderr)
        return 3

    report = result.report
    print(f"数据集：{report['dataset']['id']}（CW-003 真实公开数据基线）")
    print("数据门禁与切分：")
    for name, info in report["data"]["fingerprints"].items():
        print(f"  {name}: {info['relative_path']} | {info['size_bytes']:,} 字节 | 指纹与 CW-002 一致")
    print(
        f"  切分来源 CW-002 清单（{report['data']['split_algorithm_version']}）| "
        f"边界 validation_start={report['dataset']['boundaries']['validation_start']}、"
        f"test_start={report['dataset']['boundaries']['test_start']}"
    )
    for name in ("train", "validation", "test"):
        partition = report["partitions"][name]
        print(f"  {name:<11} 行数={partition['rows']:,} 标签率={partition['label_rate']:.4%}")

    features = report["features"]
    print(
        f"特征：允许清单 transaction {features['allowlist']['train_transaction.csv']} 列 / "
        f"identity {features['allowlist']['train_identity.csv']} 列；"
        f"训练期排除 {features['decision']['excluded_total']} 列；"
        f"最终 {features['decision']['kept_total']} 列 → 转换后 {features['transformed_dimension']} 维"
        f"（{features['output_format']}）"
    )

    print("测试集一次性评估（阈值冻结后）：")
    print(
        f"  {'基线':<22}{'阈值':>6}{'Precision':>11}{'Recall':>9}{'F1':>8}"
        f"{'ROC-AUC':>10}{'PR-AUC':>9}{'FPR':>8}"
    )
    for key in ("rule_tail_anomaly", "dummy", "logistic_regression"):
        metrics = report["baselines"][key]["test"]["metrics"]
        print(
            f"  {key:<22}{metrics['threshold']:>6.2f}{metrics['precision']:>11.4f}"
            f"{metrics['recall']:>9.4f}{metrics['f1']:>8.4f}{metrics['roc_auc']:>10.4f}"
            f"{metrics['pr_auc']:>9.4f}{metrics['false_positive_rate']:>8.4f}"
        )

    print("相对成本场景（研究用相对单位，非货币）：")
    for key in ("rule_tail_anomaly", "dummy", "logistic_regression"):
        for name, payload in report["baselines"][key]["test"]["cost_scenarios"].items():
            print(
                f"  {key:<22}{name:<12} FP:FN={payload['fp_cost']:.0f}:{payload['fn_cost']:.0f} "
                f"阈值={payload['frozen_threshold']:.2f} "
                f"总成本={payload['test_total_cost']:,.0f} "
                f"每笔={payload['test_average_cost']:.6f}"
            )

    print(f"耗时（秒）：{report['timings_seconds']} | 合计 {report['total_seconds']}")
    print("产物：")
    for path in (
        result.report_json,
        result.report_markdown,
        result.threshold_curve_csv,
        result.feature_manifest_json,
        *result.model_paths.values(),
    ):
        print(f"  {config.relative(path)}")
    print("提醒：公开匿名竞赛数据的结果不代表真实银行生产效果；报告未设置任何效果门槛。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
