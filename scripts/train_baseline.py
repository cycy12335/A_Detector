"""命令行入口：训练基线模型并输出评估报告。

用法（项目根目录下执行）::

    python scripts/train_baseline.py --config configs/baseline.yaml

数据顺序约束：训练集拟合预处理与模型，验证集选阈值，测试集只评估一次。
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

from ai_detector.config import BaselineConfig  # noqa: E402
from ai_detector.training import run_training  # noqa: E402

DEFAULT_CONFIG = "configs/baseline.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练 CW-001 基线模型并生成 JSON 报告。")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(DEFAULT_CONFIG),
        help=f"配置文件路径（相对项目根目录），默认 {DEFAULT_CONFIG}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BaselineConfig.from_yaml(args.config)
    result = run_training(config)

    report = result.report
    data_summary = report["data"]
    split_summary = report["split"]

    print(f"输入数据：{data_summary['input_path']}")
    print(f"总行数：{data_summary['n_rows']}，整体欺诈率：{data_summary['fraud_rate']:.4%}")
    print(f"切分策略：{split_summary['strategy']}")
    for name, size in split_summary["sizes"].items():
        boundaries = split_summary["boundaries"]
        start = boundaries[f"{name}_start"]
        end = boundaries[f"{name}_end"]
        rate = split_summary["fraud_rates"][name]
        print(f"  {name:<10} 行数={size:<6} 欺诈率={rate:.4%}  {start} ~ {end}")

    print("验证集选出的阈值与测试集指标：")
    for key, model in report["models"].items():
        metrics = report["test_metrics"][key]
        print(
            f"  {key:<20} 阈值={model['threshold']['threshold']:.2f} "
            f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f} "
            f"ROC-AUC={metrics['roc_auc']:.4f} PR-AUC={metrics['pr_auc']:.4f} "
            f"FPR={metrics['false_positive_rate']:.4f}"
        )

    print(f"阈值选择规则：{next(iter(report['models'].values()))['threshold']['rule']}")
    print(f"JSON 报告：{result.report_path}")
    for key, path in result.model_paths.items():
        print(f"模型产物（{key}）：{path}")
    print("提醒：以上指标来自模拟数据的时间外测试集，不代表真实风控效果。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
