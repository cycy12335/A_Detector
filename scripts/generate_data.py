"""命令行入口：生成确定性模拟交易数据。

用法（项目根目录下执行）::

    python scripts/generate_data.py --seed 42 --rows 5000 --output data/generated/transactions.csv

输出为**模拟数据**，不代表任何真实机构或真实交易，详见
``ai_detector.data.simulate`` 的模块说明。
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

from ai_detector.config import resolve_path  # noqa: E402
from ai_detector.data.simulate import (  # noqa: E402
    DEFAULT_MISSING_RATE,
    DEFAULT_ROWS,
    DEFAULT_SEED,
    DEFAULT_START_TIMESTAMP,
    generate_transactions,
    write_transactions_csv,
)

DEFAULT_OUTPUT = "data/generated/transactions.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="生成确定性模拟交易数据（模拟数据，非真实交易）。"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="生成行数")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help=f"输出 CSV 路径（相对项目根目录），默认 {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--start-timestamp",
        default=DEFAULT_START_TIMESTAMP,
        help="首笔交易时间（ISO 8601）",
    )
    parser.add_argument(
        "--missing-rate",
        type=float,
        default=DEFAULT_MISSING_RATE,
        help="数值特征上的缺失注入比例，用于验证缺失值处理路径",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    frame = generate_transactions(
        seed=args.seed,
        rows=args.rows,
        start_timestamp=args.start_timestamp,
        missing_rate=args.missing_rate,
    )
    output_path = write_transactions_csv(frame, resolve_path(args.output))

    fraud_rate = float(frame["is_fraud"].mean())
    print(f"已生成模拟交易数据：{output_path}")
    print(f"行数：{len(frame)}，列数：{frame.shape[1]}")
    print(f"时间范围：{frame['timestamp'].min()} ~ {frame['timestamp'].max()}")
    print(f"欺诈样本：{int(frame['is_fraud'].sum())}（比例 {fraud_rate:.4%}）")
    print("提醒：这是模拟数据，指标不可外推到真实风控效果。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
