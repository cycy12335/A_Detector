"""命令行入口：在真实 IEEE-CIS 数据上运行 CW-004 LightGBM 挑战者实验。

用法（项目根目录下执行）::

    python scripts/train_ieee_lightgbm.py --config configs/ieee_cis_lightgbm.yaml

前置条件（开工门禁，任一不满足即以退出码 2 失败并提示置为 blocked）：

1. CW-003 协议任务状态为 ``passed``；
2. CW-003 建模配置与 feature manifest 的 SHA-256 与门禁期望一致；
3. CW-003 的 report.json 与三类正式模型产物存在；
4. CW-002 的两个原始文件、行级切分清单与边界仍与 CW-003 数据门禁一致。

本脚本不下载数据、不访问网络、不读取任何凭据；不做超参数搜索；不做 SHAP / Permutation /
PDP / 概率校准；不重新评估 CW-003 三基线的 test。
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

from ai_detector.data.modeling import DataGateError  # noqa: E402
from ai_detector.lightgbm_training import (  # noqa: E402
    DeterminismError,
    FeatureContractError,
    LightGbmConfig,
    PrerequisiteGateError,
    run_lightgbm_training,
)
from ai_detector.models.lightgbm_challenger import LightGbmSpecError  # noqa: E402

DEFAULT_CONFIG = "configs/ieee_cis_lightgbm.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在真实 IEEE-CIS 数据上运行 CW-004 固定参数 LightGBM 挑战者实验。"
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
        help="读取分块行数；默认使用代码内的固定值，仅影响内存与耗时，不影响结论",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = LightGbmConfig.from_yaml(args.config)
    except Exception as error:  # noqa: BLE001 - 配置错误需要原样暴露给使用者
        print(f"配置加载失败：{error}", file=sys.stderr)
        return 2

    kwargs = {} if args.chunk_size is None else {"chunk_size": args.chunk_size}
    try:
        result = run_lightgbm_training(config, **kwargs)
    except (PrerequisiteGateError, DataGateError) as error:
        print(str(error), file=sys.stderr)
        print(
            "请把任务状态置为 blocked：不得替换文件、重算切分或降低门禁。",
            file=sys.stderr,
        )
        return 2
    except LightGbmSpecError as error:
        print(f"LightGBM 规格或依赖失败：{error}", file=sys.stderr)
        print(
            "若为 LightGBM 无法安装/导入，请把任务状态置为 blocked；"
            "不得源码编译、降级 Python、新建替代环境或改用其他模型框架。",
            file=sys.stderr,
        )
        return 4
    except DeterminismError as error:
        print(str(error), file=sys.stderr)
        print("候选 A/B 不确定，任务应失败并保留差异证据；不得开放 test。", file=sys.stderr)
        return 5
    except FeatureContractError as error:
        print(f"特征契约校验失败：{error}", file=sys.stderr)
        return 3

    report = result.report
    gates = report["prerequisite_gates"]
    model = report["model"]
    early = report["candidate_runs"]["B"]["early_stopping"]
    determinism = report["candidate_runs"]["determinism"]
    metrics = report["test_evaluation"]["metrics"]

    print("前置门禁：")
    print(
        f"  CW-003 协议任务 {gates['prerequisite_task']['status']} | "
        f"配置指纹一致 {gates['cw003_config']['match']} | "
        f"feature manifest 一致 {gates['cw003_feature_manifest']['match']}"
    )
    print("数据门禁与切分：")
    for name, info in report["data"]["fingerprints"].items():
        print(f"  {name}: {info['relative_path']} | {info['size_bytes']:,} 字节 | 与 CW-002 一致")
    for name in ("train", "validation", "test"):
        partition = report["partitions"][name]
        print(f"  {name:<11} 行数={partition['rows']:,} 标签率={partition['label_rate']:.4%}")

    features = report["features"]["contract"]
    print(
        f"特征契约：{features['kept_numeric']} 数值 + {features['kept_categorical']} 类别 = "
        f"{features['kept_total']} 列 → {features['expected_transformed_dimension']} 维 "
        f"({features['output_format']})，与已验收 manifest 逐项一致："
        f"{features['matches_cw003_manifest']}"
    )

    print(
        f"模型：{model['type']} | LightGBM {model['lightgbm_version']} | CPU | "
        f"n_jobs={model['n_jobs']} | 未做超参数搜索"
    )
    print(
        f"  参数：{ {k: model['params'][k] for k in sorted(model['params'])} }"
    )
    print(
        f"  scale_pos_weight={report['candidate_runs']['B']['scale_pos_weight']:.6f}"
        "（只用 train 标签）"
    )
    print(
        f"  早停：metric={early['metric']} rounds={early['stopping_rounds']} "
        f"best_iteration={early['best_iteration']} 触发={early['early_stopping_fired']}"
    )

    print("两阶段确定性：")
    print(
        f"  容差 atol={determinism['tolerance']['absolute']} "
        f"rtol={determinism['tolerance']['relative']} | "
        f"比较项 {len(determinism['compared_items'])} 个 | 通过={determinism['passed']} | "
        f"失败项={determinism['failed_items'] or '无'}"
    )
    lifecycle = report["evaluation"]["lifecycle_requirements"]
    print(
        f"  test 生命周期：首条事件为冻结 {lifecycle['first_event_is_freeze']} | "
        f"冻结后 features 访问 {lifecycle['n_feature_accesses_after_freeze']} 次 | "
        f"labels 访问 {lifecycle['n_label_accesses_after_freeze']} 次"
    )

    print("测试集一次性评估（阈值冻结后）：")
    print(
        f"  {'模型':<22}{'阈值':>6}{'Precision':>11}{'Recall':>9}{'F1':>8}"
        f"{'ROC-AUC':>10}{'PR-AUC':>9}{'FPR':>8}"
    )
    for row in report["comparison"]:
        print(
            f"  {row['model_key']:<22}{row['frozen_threshold']:>6.2f}"
            f"{row['precision']:>11.4f}{row['recall']:>9.4f}{row['f1']:>8.4f}"
            f"{row['roc_auc']:>10.4f}{row['pr_auc']:>9.4f}"
            f"{row['false_positive_rate']:>8.4f}"
        )
    print(
        f"  混淆矩阵：TP {metrics['true_positive']:,} / FP {metrics['false_positive']:,} / "
        f"TN {metrics['true_negative']:,} / FN {metrics['false_negative']:,}"
    )

    print("相对成本场景（研究用相对单位，非货币）：")
    for name, payload in report["test_evaluation"]["cost_scenarios"].items():
        print(
            f"  {name:<12} FP:FN={payload['fp_cost']:.0f}:{payload['fn_cost']:.0f} "
            f"阈值={payload['frozen_threshold']:.2f} "
            f"总成本={payload['test_total_cost']:,.0f} "
            f"每笔={payload['test_average_cost']:.6f}"
        )

    importance = report["feature_importance"]
    print(f"聚合重要性：{importance['n_source_features']} 个源特征（gain/split），守恒校验：")
    print(
        f"  gain 绝对差 {importance['conservation']['gain']['absolute_difference']:.3e} | "
        f"split 绝对差 {importance['conservation']['split']['absolute_difference']:.3e}"
    )
    print("  不含 one-hot 类别取值；不做 SHAP / Permutation / PDP；不赋予匿名列业务含义。")

    print(f"耗时（秒）：{report['timings_seconds']} | 合计 {report['total_seconds']}")
    print(
        f"峰值内存：{report['resources']['peak_memory_mib']} MiB"
        f"（{report['resources']['peak_memory_measurement']}）"
    )
    print("产物：")
    for path in (
        result.report_json,
        result.report_markdown,
        result.threshold_curve_csv,
        result.comparison_csv,
        result.feature_importance_csv,
        result.bundle_path,
    ):
        print(f"  {config.relative(path)}")
    print(
        "提醒：公开匿名竞赛数据的结果不代表真实银行生产效果；模型输出是分数而非校准概率；"
        "三基线读取自 CW-003 报告，未重新评估其 test。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
