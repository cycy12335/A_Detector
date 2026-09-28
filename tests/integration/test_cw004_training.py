"""CW-004 端到端集成测试。

在**项目自建的合成夹具**上跑完整实验：前置门禁 → 复用 CW-002 切分 → train-only 列决策 →
契约核对 → 装配 → 候选运行 A/B → 确定性核对 → 冻结 → 一次受控 test → 四模型对比 →
聚合重要性 → 推理 bundle → 五类报告产物。

重点守住五条纪律：

1. 特征契约、预处理与 ``scale_pos_weight`` 只来自 train；
2. 候选 A/B 全程不接触 test，确定性比较通过后才冻结；
3. test 恰好一次 features 读取 + 恰好一次 labels 读取，日志由真实控制逻辑产生；
4. CW-003 三基线只从已验收报告读取，未被重新评估；
5. 报告与产物不含绝对路径、凭据、逐行取值或类别原值。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ai_detector.data.modeling import (
    HoldoutAccessTracker,
    HoldoutLeakageError,
    assemble_partitions,
    load_partition_lookup,
)
from ai_detector.evaluation.feature_importance import (
    aggregate_by_source,
    transformed_dim_sources,
)
from ai_detector.features.ieee_cis import accumulate_train_stats, build_preprocessor, decide_columns
from ai_detector.lightgbm_training import (
    build_comparison_frame,
    compare_determinism,
    run_candidate,
    run_lightgbm_training,
)
from ai_detector.models.lightgbm_challenger import (
    LGBM_KEY,
    LightGbmBundle,
    compute_scale_pos_weight,
    load_bundle,
)

REQUIRED_REPORT_KEYS = {
    "work_package",
    "generated_at",
    "seed",
    "dataset",
    "config",
    "prerequisite_gates",
    "data",
    "partitions",
    "features",
    "model",
    "candidate_runs",
    "thresholds",
    "evaluation",
    "test_evaluation",
    "cw003_baseline_source",
    "comparison",
    "feature_importance",
    "inference_bundle",
    "resources",
    "timings_seconds",
    "limitations",
    "environment",
}


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """跑一次完整 CW-004 实验，供本模块多个测试复用。"""
    from tests.cw004_fixtures import Cw004FixtureBuilder

    workspace = tmp_path_factory.mktemp("cw004")
    config_path, config, cw003 = Cw004FixtureBuilder(workspace).build_cw004()
    result = run_lightgbm_training(config, chunk_size=64)
    return {
        "result": result,
        "config": config,
        "config_path": config_path,
        "cw003": cw003,
        "workspace": workspace,
    }


# --- 门禁与契约 ---------------------------------------------------------


def test_report_carries_every_required_section(trained: dict) -> None:
    assert REQUIRED_REPORT_KEYS <= set(trained["result"].report)
    assert trained["result"].report["work_package"] == "CW-004"


def test_prerequisite_gates_recorded_and_passed(trained: dict) -> None:
    gates = trained["result"].report["prerequisite_gates"]
    assert gates["all_passed"] is True
    assert gates["prerequisite_task"]["status"] == "passed"
    assert gates["cw003_config"]["match"] is True
    assert gates["cw003_config"]["sha256"] == gates["cw003_config"]["expected_sha256"]
    assert gates["cw003_feature_manifest"]["match"] is True
    assert gates["cw003_report"]["sha256"]
    assert all(item["exists"] for item in gates["cw003_required_artifacts"])


def test_feature_contract_matches_cw003_manifest(trained: dict) -> None:
    config = trained["config"]
    contract = trained["result"].report["features"]["contract"]
    manifest = json.loads(
        (trained["workspace"] / "out" / "feature_manifest.json").read_text(encoding="utf-8")
    )

    assert contract["matches_cw003_manifest"] is True
    assert contract["order_matches"] is True
    assert contract["kept_numeric"] == len(manifest["kept"]["numeric"])
    assert contract["kept_categorical"] == len(manifest["kept"]["categorical"])
    assert contract["kept_total"] == config.expected_source_features
    assert contract["expected_transformed_dimension"] == manifest["transformed_dimension"]
    assert contract["output_format"] == manifest["output_format"] == "sparse_csr"


def test_forbidden_features_never_reach_the_model(trained: dict) -> None:
    report = trained["result"].report
    forbidden = set(report["features"]["forbidden_features"])
    assert forbidden == {"isFraud", "TransactionID", "TransactionDT"}

    bundle = load_bundle(trained["result"].bundle_path)
    assert not (set(bundle.source_features) & forbidden)
    for name in bundle.transformed_feature_names:
        tail = name.split("__", 1)[-1]
        for banned in forbidden:
            assert tail != banned and not tail.startswith(f"{banned}_"), name

    # 逐维重要性表的键同样不得出现禁止字段。
    frame = pd.read_csv(trained["result"].feature_importance_csv)
    assert not (set(frame["source_feature"]) & forbidden)
    assert "identity_present" not in set(frame["source_feature"])


def test_derived_identity_present_is_not_consumed_by_the_model(trained: dict) -> None:
    """CW-003 的 remainder='drop' 语义被原样保留：派生字段不进入模型。"""
    report = trained["result"].report
    assert report["features"]["identity_present_consumed_by_model"] is False
    bundle = load_bundle(trained["result"].bundle_path)
    assert "identity_present" not in bundle.source_features
    assert "identity_present" not in bundle.transformed_feature_names


# --- train-only 边界 ----------------------------------------------------


def _perturb_partition(config, lookup: pd.Series, partition: str) -> None:
    """破坏指定分区中的特征值；train 分区保持不变。"""
    from tests.cw003_fixtures import CW003_CATEGORICAL, CW003_NUMERIC

    path = config.paths["transaction"]
    frame = pd.read_csv(path)
    ids = lookup[lookup == partition].index
    mask = frame["TransactionID"].isin(ids)
    for column in CW003_NUMERIC:
        frame.loc[mask, column] = 1e9
    for column in CW003_CATEGORICAL:
        frame.loc[mask, column] = "NEVER_SEEN_CATEGORY"
    frame.to_csv(path, index=False)


def _preprocessor_fingerprint(pipeline) -> dict:
    numeric = pipeline.named_transformers_.get("numeric")
    categorical = pipeline.named_transformers_.get("categorical")
    fingerprint: dict = {}
    if numeric is not None:
        fingerprint["median"] = np.asarray(numeric.named_steps["imputer"].statistics_).tolist()
        fingerprint["mean"] = np.asarray(numeric.named_steps["scaler"].mean_).tolist()
        fingerprint["scale"] = np.asarray(numeric.named_steps["scaler"].scale_).tolist()
        indicator = getattr(numeric.named_steps["imputer"], "indicator_", None)
        fingerprint["indicator_features"] = (
            [] if indicator is None else np.asarray(indicator.features_).tolist()
        )
    if categorical is not None:
        encoder = categorical.named_steps["encoder"]
        fingerprint["categories"] = [list(map(str, values)) for values in encoder.categories_]
    return fingerprint


def test_held_out_perturbation_changes_nothing_train_derived(cw004_fixture) -> None:
    """扰动 validation/test 后：保留列、训练期统计、预处理参数与 scale_pos_weight 全不变。"""
    _, config, _ = cw004_fixture.build_cw004()
    cw003 = config.modeling
    lookup = load_partition_lookup(cw003)

    before_stats = accumulate_train_stats(cw003, lookup, chunk_size=64)
    before_decision = decide_columns(cw003, before_stats)
    assembled = assemble_partitions(cw003, lookup, before_decision, chunk_size=64)
    pipeline = build_preprocessor(cw003, before_decision)
    pipeline.fit(assembled.features("train", list(before_decision.kept)), None)
    before_fingerprint = _preprocessor_fingerprint(pipeline)
    tracker = HoldoutAccessTracker()
    before_weight = compute_scale_pos_weight(assembled.target("train", tracker=tracker))

    for partition in ("validation", "test"):
        _perturb_partition(cw003, lookup, partition)

    after_stats = accumulate_train_stats(cw003, lookup, chunk_size=64)
    after_decision = decide_columns(cw003, after_stats)
    assembled_after = assemble_partitions(cw003, lookup, after_decision, chunk_size=64)
    pipeline_after = build_preprocessor(cw003, after_decision)
    pipeline_after.fit(
        assembled_after.features("train", list(after_decision.kept)), None
    )

    assert after_decision.numeric == before_decision.numeric
    assert after_decision.categorical == before_decision.categorical
    for name, stats in before_stats.items():
        assert after_stats[name].to_dict() == stats.to_dict(), name
    assert _preprocessor_fingerprint(pipeline_after) == before_fingerprint
    after_weight = compute_scale_pos_weight(assembled_after.target("train", tracker=tracker))
    assert after_weight == before_weight


def test_scale_pos_weight_matches_train_label_counts(trained: dict) -> None:
    report = trained["result"].report
    train = report["partitions"]["train"]
    rows = train["rows"]
    positives = round(train["label_rate"] * rows)
    expected = (rows - positives) / positives
    assert report["candidate_runs"]["B"]["scale_pos_weight"] == pytest.approx(expected)


def test_candidate_runs_declare_train_validation_scope_only(trained: dict) -> None:
    runs = trained["result"].report["candidate_runs"]
    assert runs["design"]["n_runs"] == 2
    assert runs["design"]["partition_scope"] == ["train", "validation"]
    assert runs["design"]["test_accessed"] is False
    for label in ("A", "B"):
        assert runs[label]["test_accessed"] is False
        assert runs[label]["partition_scope"] == ["train", "validation"]


def test_early_stopping_only_sees_validation(trained: dict) -> None:
    report = trained["result"].report
    model = report["model"]
    assert model["early_stopping"]["validation_partition"] == "validation"
    assert model["early_stopping"]["metric"] == "binary_logloss"
    assert model["early_stopping"]["stopping_rounds"] == 50
    assert model["early_stopping"]["first_metric_only"] is True

    for label in ("A", "B"):
        early = report["candidate_runs"][label]["early_stopping"]
        assert early["metric"] == "binary_logloss"
        assert early["stopping_rounds"] == 50
        assert early["first_metric_only"] is True
        assert early["n_estimators_cap"] == model["params"]["n_estimators"]
        assert 0 <= early["best_iteration"] <= model["params"]["n_estimators"]
        assert early["trees_kept"] <= early["rounds_run"] <= model["params"]["n_estimators"]
        # 早停证据必须来自 validation 上的评估历史。
        assert early["history_length"] == early["rounds_run"] >= 1
        assert early["best_score"] is not None


def test_no_hyperparameter_search_and_params_frozen(trained: dict) -> None:
    report = trained["result"].report
    model = report["model"]
    assert model["hyperparameter_search"] is False
    assert model["device"] == "cpu"
    assert model["n_jobs"] == 1
    assert model["params"]["n_estimators"] == 1000
    assert model["params"]["learning_rate"] == 0.05
    assert model["params"]["num_leaves"] == 31
    assert model["params"]["random_state"] == report["seed"]
    assert model["lightgbm_version"] not in ("", "unavailable", "unknown")


# --- 两阶段确定性与 test 生命周期 ---------------------------------------


def test_determinism_comparison_passed_within_declared_tolerance(trained: dict) -> None:
    determinism = trained["result"].report["candidate_runs"]["determinism"]
    assert determinism["passed"] is True
    assert determinism["failed_items"] == []
    assert determinism["tolerance"] == {"absolute": 1e-12, "relative": 0.0}
    assert determinism["test_opened"] is False
    assert determinism["items"]["validation_scores"]["max_absolute_difference"] is not None


def test_determinism_comparison_rejects_perturbed_run(trained: dict) -> None:
    """把真实运行的载荷扰动一点点，比较必须失败——证明比较不是恒真。"""
    runs = trained["result"].report["candidate_runs"]
    assert runs["A"]["validation_score_summary"]["n"] > 0

    left = {
        "kept_numeric": runs["A"]["kept_numeric"],
        "kept_categorical": runs["A"]["kept_categorical"],
        "transformed_dimension": runs["A"]["transformed_dimension"],
        "output_format": runs["A"]["output_format"],
        "scale_pos_weight": runs["A"]["scale_pos_weight"],
        "best_iteration": runs["A"]["best_iteration"],
        "validation_scores": np.zeros(runs["A"]["validation_score_summary"]["n"]),
        "frozen_thresholds": runs["A"]["frozen_thresholds"],
        "validation_confusion_matrix": runs["A"]["validation_confusion_matrix"],
        "aggregated_gain_importance": {
            item["source_feature"]: item["aggregated_gain"]
            for item in runs["A"]["top_gain_source_features"]
        },
        "aggregated_split_importance": {
            item["source_feature"]: 0 for item in runs["A"]["top_gain_source_features"]
        },
    }
    right = {**left, "validation_scores": np.full(left["validation_scores"].size, 0.5)}
    result = compare_determinism(left, right, absolute_tolerance=1e-12)
    assert result["passed"] is False
    assert "validation_scores" in result["failed_items"]

    shifted = {**left, "best_iteration": left["best_iteration"] + 1}
    assert compare_determinism(left, shifted, absolute_tolerance=1e-12)["passed"] is False


def test_test_lifecycle_is_exactly_one_features_and_one_labels_access(trained: dict) -> None:
    lifecycle = trained["result"].report["evaluation"]["lifecycle_requirements"]
    assert lifecycle["first_event_is_freeze"] is True
    assert lifecycle["n_feature_accesses_after_freeze"] == 1
    assert lifecycle["n_label_accesses_after_freeze"] == 1
    assert lifecycle["report_generation_accessed_test"] is False
    assert lifecycle["importance_computation_accessed_test"] is False
    assert lifecycle["bundle_save_accessed_test"] is False

    access = trained["result"].report["evaluation"]["test_access"]
    assert access["frozen"] is True
    assert access["n_test_accesses"] == 2
    entries = access["entries"]
    assert entries[0]["event"] == "thresholds_frozen"
    assert [entry["what"] for entry in entries[1:]] == ["features", "labels"]
    for entry in entries[1:]:
        assert entry["after_freeze"] is True
        assert entry["purpose"] == "single_shot_evaluation"
        assert entry["sequence"] > entries[0]["sequence"]


def test_freeze_reason_records_model_and_thresholds(trained: dict) -> None:
    reason = trained["result"].report["evaluation"]["test_access"]["frozen_reason"]
    best_iteration = trained["result"].report["candidate_runs"]["B"]["best_iteration"]
    assert "冻结" in reason
    assert f"best_iteration={best_iteration}" in reason


@pytest.fixture(scope="module")
def assembled_only(trained: dict) -> dict:
    """只装配、不训练：供生命周期门禁的负向/正向探针复用。"""
    cw003 = trained["config"].modeling
    lookup = load_partition_lookup(cw003)
    decision = decide_columns(cw003, accumulate_train_stats(cw003, lookup, chunk_size=64))
    return {
        "assembled": assemble_partitions(cw003, lookup, decision, chunk_size=64),
        "decision": decision,
        "cw003": cw003,
    }


def test_pre_freeze_test_feature_access_raises_and_logs_nothing(assembled_only: dict) -> None:
    """**冻结前读取 test 特征必须失败，且不返回任何数据、不产生任何日志。**

    这是首轮验收指出的缺陷的回归测试：原实现下 ``features("test", ...)`` 在冻结前
    会成功返回数据且 tracker 仍为空。
    """
    assembled = assembled_only["assembled"]
    tracker = HoldoutAccessTracker()
    columns = list(assembled_only["decision"].kept)

    with pytest.raises(HoldoutLeakageError, match="在阈值冻结之前尝试访问 test"):
        assembled.features("test", columns, tracker=tracker)
    assert tracker.entries == [], "失败的读取不得留下任何日志条目"
    assert tracker.frozen is False

    # train / validation 不受影响：不需要追踪器，也不产生日志。
    assert len(assembled.features("train", columns)) > 0
    assert len(assembled.features("validation", columns)) > 0
    assert tracker.entries == []


def test_test_feature_access_requires_a_tracker(assembled_only: dict) -> None:
    """没有追踪器就没有冻结检查与记账，因此该入口拒绝返回 test 数据。"""
    assembled = assembled_only["assembled"]
    columns = list(assembled_only["decision"].kept)

    with pytest.raises(HoldoutLeakageError, match="必须提供生命周期追踪器"):
        assembled.features("test", columns)
    # 即使先冻结，缺少追踪器依然不可读——不存在"绕过记账"的成功路径。
    HoldoutAccessTracker().freeze(reason="探针")
    with pytest.raises(HoldoutLeakageError, match="必须提供生命周期追踪器"):
        assembled.features("test", columns)


def test_test_partition_cannot_be_taken_whole(assembled_only: dict) -> None:
    """``assembled["test"]`` 整体取出必须被拒绝：第二条未受保护的读取路径已被关闭。"""
    assembled = assembled_only["assembled"]

    with pytest.raises(HoldoutLeakageError, match="不得经 assembled\\['test'\\] 整体取出"):
        assembled["test"]
    # 结构性元信息仍然可用（列名、行数、连接键），但它们不含任何取值。
    assert assembled.structural_row_count("test") > 0
    assert assembled.structural_columns("test")
    assert assembled.structural_ids("test")
    assert assembled_only["cw003"].target not in assembled.structural_columns("test")
    # train / validation 整体取出不受影响。
    assert len(assembled["train"]) > 0


def test_post_freeze_test_features_read_once_and_logged_once(assembled_only: dict) -> None:
    """冻结后允许一次成功读取，且日志由读取入口自身产生（不是调用方补写）。"""
    assembled = assembled_only["assembled"]
    tracker = HoldoutAccessTracker()
    columns = list(assembled_only["decision"].kept)

    tracker.freeze(reason="集成测试：冻结后读取探针")
    frame = assembled.features("test", columns, tracker=tracker)

    assert len(frame) == assembled_only["cw003"].expected_split_sizes["test"]
    assert list(frame.columns) == columns
    accesses = [entry for entry in tracker.entries if entry["event"] == "test_access"]
    assert len(accesses) == 1, "受控读取必须恰好产生一条日志"
    assert accesses[0]["what"] == "features"
    assert accesses[0]["purpose"] == "single_shot_evaluation"
    assert accesses[0]["after_freeze"] is True


def test_test_labels_cannot_be_read_before_freezing(assembled_only: dict) -> None:
    """生命周期测试：冻结前读取 test 标签必须失败，且不产生访问日志。"""
    assembled = assembled_only["assembled"]
    cw003 = assembled_only["cw003"]
    tracker = HoldoutAccessTracker()

    assert assembled.structural_ids("test")
    with pytest.raises(HoldoutLeakageError, match="在阈值冻结之前尝试访问 test"):
        assembled.target("test", tracker=tracker)
    assert tracker.entries == []

    tracker.freeze(reason="集成测试：手动冻结")
    assert assembled.target("test", tracker=tracker).size == cw003.expected_split_sizes["test"]
    assert [entry["event"] for entry in tracker.entries] == ["thresholds_frozen", "test_access"]


def test_official_flow_has_no_manual_tracking_or_raw_test_access() -> None:
    """结构性回归：正式流程里不得再出现"先取数据、后手工 tracker.read"或裸取 test 整帧。"""
    import ast
    from pathlib import Path as _Path

    for module in ("src/ai_detector/lightgbm_training.py", "src/ai_detector/ieee_training.py"):
        source = _Path(module).read_text(encoding="utf-8")
        assert "tracker.read(" not in source, f"{module} 手工写访问日志"
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                if node.value.id != "assembled":
                    continue
                index = node.slice
                text = ast.unparse(index)
                assert text not in {"PARTITION_TEST", "'test'", '"test"'}, (
                    f"{module} 仍存在对 test 整帧的裸读取：assembled[{text}]"
                )
        # test 特征只允许经受控入口取得。
        assert "features(\n        PARTITION_TEST" in source or "features(PARTITION_TEST" in source
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr != "features":
                    continue
                first = node.args[0] if node.args else None
                if first is None or ast.unparse(first) not in {"PARTITION_TEST", "'test'", '"test"'}:
                    continue
                assert any(kw.arg == "tracker" for kw in node.keywords), (
                    f"{module} 读取 test 特征时未传 tracker"
                )


def test_test_partition_target_is_sealed_until_freeze(trained: dict) -> None:
    cw003 = trained["config"].modeling
    lookup = load_partition_lookup(cw003)
    decision = decide_columns(cw003, accumulate_train_stats(cw003, lookup, chunk_size=64))
    assembled = assemble_partitions(cw003, lookup, decision, chunk_size=64)

    assert cw003.target not in assembled.structural_columns("test")
    assert cw003.target in assembled["train"].columns


def test_pre_freeze_stage_reads_no_test_labels(trained: dict) -> None:
    evaluation = trained["result"].report["evaluation"]
    assert evaluation["pre_freeze_structural_only"]["test_labels_read_before_freeze"] is False
    assert evaluation["pre_freeze_structural_only"]["candidate_runs_accessed_test"] is False
    assert set(evaluation["pre_freeze_structural_only"]["label_rates_computed_for"]) == {
        "train",
        "validation",
    }
    assert trained["result"].report["partitions"]["test"]["label_rate_computed_after_freeze"] is True


# --- 阈值与 test 指标 ---------------------------------------------------


def test_threshold_curve_is_lightgbm_validation_only(trained: dict) -> None:
    config = trained["config"]
    curve = pd.read_csv(trained["result"].threshold_curve_csv)
    grid = config.modeling.threshold_grid()

    assert set(curve["model_key"]) == {LGBM_KEY}
    assert set(curve["partition"]) == {"validation"}
    assert len(curve) == len(grid)
    for column in (
        "threshold",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "predicted_positive_rate",
    ):
        assert column in curve.columns


def test_frozen_f1_threshold_is_the_curve_argmax(trained: dict) -> None:
    """冻结阈值必须等于"仅用 validation 曲线按声明 tie-break 选出的"那一个。"""
    curve = pd.read_csv(trained["result"].threshold_curve_csv)
    best = max(
        curve.to_dict(orient="records"),
        key=lambda row: (row["f1"], row["recall"], row["threshold"]),
    )
    frozen = trained["result"].report["thresholds"]["frozen"]
    assert frozen["selected_f1_threshold"] == pytest.approx(best["threshold"])
    assert trained["result"].report["thresholds"]["source_partition"] == "validation"
    assert trained["result"].report["thresholds"]["grid_size"] == len(
        trained["config"].modeling.threshold_grid()
    )


def test_cost_scenarios_are_frozen_on_validation_and_applied_to_test(trained: dict) -> None:
    report = trained["result"].report
    frozen = report["thresholds"]["frozen"]["cost_scenarios"]
    tested = report["test_evaluation"]["cost_scenarios"]

    assert set(frozen) == set(tested) == {"fp_fn_1_1", "fp_fn_1_5", "fp_fn_1_20"}
    for name, payload in tested.items():
        assert payload["frozen_threshold"] == frozen[name]["selected_threshold"]
        assert payload["fp_cost"] == frozen[name]["fp_cost"]
        assert payload["fn_cost"] == frozen[name]["fn_cost"]
        matrix = payload["test_confusion_matrix"]
        assert payload["test_total_cost"] == pytest.approx(
            matrix["false_positive"] * payload["fp_cost"]
            + matrix["false_negative"] * payload["fn_cost"]
        )
        assert payload["test_average_cost"] == pytest.approx(
            payload["test_total_cost"] / report["partitions"]["test"]["rows"]
        )


def test_higher_fn_cost_never_lowers_the_threshold(trained: dict) -> None:
    frozen = trained["result"].report["thresholds"]["frozen"]["cost_scenarios"]
    assert frozen["fp_fn_1_1"]["selected_threshold"] <= frozen["fp_fn_1_20"]["selected_threshold"]


def test_test_metrics_are_complete_and_uncalibrated_scores(trained: dict) -> None:
    payload = trained["result"].report["test_evaluation"]
    metrics = payload["metrics"]
    rows = trained["result"].report["partitions"]["test"]["rows"]

    for key in (
        "threshold",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "predicted_positive_rate",
        "roc_auc",
        "pr_auc",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
    ):
        assert key in metrics, key
    assert metrics["n_samples"] == rows
    assert metrics["threshold"] == pytest.approx(
        trained["result"].report["thresholds"]["frozen"]["selected_f1_threshold"]
    )
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert payload["score_semantics"] == "模型分数（score），不是已校准概率"
    assert metrics["true_positive"] + metrics["false_negative"] == metrics["n_positive"]


def test_no_effect_threshold_is_declared(trained: dict) -> None:
    """报告必须显式声明不设效果门槛，且不宣称生产效果。"""
    markdown = trained["result"].report_markdown.read_text(encoding="utf-8")
    limitations = " ".join(trained["result"].report["limitations"])
    assert "不设置任何效果门槛" in markdown
    assert "概率校准" in limitations
    assert "生产" in limitations


# --- 三基线来源化对比 ---------------------------------------------------


def test_cw003_baselines_are_read_not_re_evaluated(trained: dict) -> None:
    config = trained["config"]
    source = trained["result"].report["cw003_baseline_source"]
    cw003_report_path = Path(config.cw003["report_path"])
    cw003_report = json.loads(cw003_report_path.read_text(encoding="utf-8"))

    assert source["source_work_package"] == "CW-003"
    assert source["re_evaluated"] is False
    assert source["source_report_sha256"]
    assert set(source["baselines"]) == set(config.baseline_keys)

    for key in config.baseline_keys:
        recorded = source["baselines"][key]
        original = cw003_report["baselines"][key]
        assert recorded["test"]["metrics"] == original["test"]["metrics"]
        assert recorded["thresholds"] == original["thresholds"]
        assert recorded["fit_seconds"] == original["fit_seconds"]

    # CW-003 的报告文件在本工作包运行后必须逐字节未变。
    assert (
        hashlib.sha256(cw003_report_path.read_bytes()).hexdigest()
        == source["source_report_sha256"]
    )


def test_comparison_frame_has_four_models_with_provenance(trained: dict) -> None:
    config = trained["config"]
    comparison = pd.read_csv(trained["result"].comparison_csv)
    assert set(comparison["model_key"]) == {*config.baseline_keys, LGBM_KEY}
    assert len(comparison) == 4

    by_key = comparison.set_index("model_key")
    for key in config.baseline_keys:
        assert by_key.loc[key, "metrics_source"] == "cw003_report_json_read_only"
        assert by_key.loc[key, "source_work_package"] == "CW-003"
        assert by_key.loc[key, "source_report_sha256"] == (
            trained["result"].report["cw003_baseline_source"]["source_report_sha256"]
        )
    assert by_key.loc[LGBM_KEY, "metrics_source"] == "cw004_single_shot_test_evaluation"
    assert by_key.loc[LGBM_KEY, "train_seconds_source"] == "cw004_candidate_run_b"

    for column in ("pr_auc", "roc_auc", "f1", "precision", "recall", "false_positive_rate",
                   "frozen_threshold", "train_seconds", "complexity_note"):
        assert column in comparison.columns


def test_comparison_metrics_match_the_report(trained: dict) -> None:
    report = trained["result"].report
    comparison = pd.read_csv(trained["result"].comparison_csv).set_index("model_key")
    metrics = report["test_evaluation"]["metrics"]
    assert comparison.loc[LGBM_KEY, "roc_auc"] == pytest.approx(metrics["roc_auc"])
    assert comparison.loc[LGBM_KEY, "pr_auc"] == pytest.approx(metrics["pr_auc"])
    assert comparison.loc[LGBM_KEY, "f1"] == pytest.approx(metrics["f1"])


def test_comparison_frame_is_a_pure_function_of_its_inputs(trained: dict) -> None:
    """对比表由已记录的数据重建后必须逐字段一致（不存在手工填写的数字）。"""
    report = trained["result"].report
    rebuilt = build_comparison_frame(
        trained["config"],
        report["cw003_baseline_source"],
        report["test_evaluation"],
        lightgbm_train_seconds=report["candidate_runs"]["B"]["timings_seconds"]["lightgbm_fit"],
    )
    recorded = pd.read_csv(trained["result"].comparison_csv)
    pd.testing.assert_frame_equal(rebuilt, recorded)


def test_no_correlation_claimed_as_causation(trained: dict) -> None:
    markdown = trained["result"].report_markdown.read_text(encoding="utf-8")
    assert "不构成因果解释" in markdown or "不是因果关系" in markdown


# --- 聚合重要性 ---------------------------------------------------------


def test_importance_aggregates_back_to_source_features(trained: dict) -> None:
    report = trained["result"].report
    frame = pd.read_csv(trained["result"].feature_importance_csv)
    bundle = load_bundle(trained["result"].bundle_path)

    assert len(frame) == report["feature_importance"]["n_source_features"]
    assert set(frame["source_feature"]) == set(bundle.source_features)
    # 每个源特征一行：数值 + 类别 = kept_total，且与 bundle 的分类清单一致。
    assert len(frame) == report["features"]["contract"]["kept_total"]
    assert len(bundle.numeric_features) == report["features"]["contract"]["kept_numeric"]
    assert len(bundle.categorical_features) == report["features"]["contract"]["kept_categorical"]
    assert set(frame["kind"]) <= {"numeric", "categorical"}
    assert frame["rank"].tolist() == list(range(1, len(frame) + 1))
    assert report["feature_importance"]["category_values_exposed"] is False


def test_importance_conservation_is_verified(trained: dict) -> None:
    conservation = trained["result"].report["feature_importance"]["conservation"]
    for key in ("gain", "split"):
        entry = conservation[key]
        assert entry["conserved"] is True
        assert abs(entry["absolute_difference"]) <= entry["tolerance"]


def test_importance_recomputation_matches_the_written_table(trained: dict) -> None:
    """用 bundle 里的模型重新计算 gain/split 并聚合，必须与写入的 CSV 一致。"""
    config = trained["config"]
    cw003 = config.modeling
    lookup = load_partition_lookup(cw003)
    decision = decide_columns(cw003, accumulate_train_stats(cw003, lookup, chunk_size=64))
    assembled = assemble_partitions(cw003, lookup, decision, chunk_size=64)
    tracker = HoldoutAccessTracker()
    run = run_candidate("probe", config, decision, assembled, tracker)

    sources = transformed_dim_sources(run.preprocessor, decision.numeric, decision.categorical)
    expected_gain = aggregate_by_source(sources, run.gain_by_dimension)
    frame = pd.read_csv(trained["result"].feature_importance_csv).set_index("source_feature")
    for name, value in expected_gain.items():
        assert frame.loc[name, "aggregated_gain"] == pytest.approx(value)


def test_importance_frame_leaks_no_category_values(trained: dict) -> None:
    frame = pd.read_csv(trained["result"].feature_importance_csv)
    blob = trained["result"].feature_importance_csv.read_text(encoding="utf-8")
    assert "NEVER_SEEN_CATEGORY" not in blob
    assert set(frame.columns) == {
        "rank",
        "source_feature",
        "kind",
        "aggregated_gain",
        "aggregated_split",
        "gain_share",
    }
    # 表里只允许出现源列名本身，不得出现 one-hot 展开后的列名。
    for value in frame["source_feature"]:
        assert "__" not in value, f"出现转换后特征名而非源列名：{value}"


def test_importance_declares_non_causal_limits(trained: dict) -> None:
    importance = trained["result"].report["feature_importance"]
    assert importance["forbidden_analyses"]
    joined = " ".join(importance["interpretation_limits"])
    assert "因果" in joined
    assert "业务" in joined
    for banned in ("SHAP", "Permutation", "PDP"):
        assert banned in " ".join(importance["forbidden_analyses"])


# --- 推理 bundle --------------------------------------------------------


def test_bundle_roundtrip_scores_and_thresholds_are_identical(trained: dict) -> None:
    bundle = load_bundle(trained["result"].bundle_path)
    config = trained["config"]
    cw003 = config.modeling
    lookup = load_partition_lookup(cw003)
    decision = decide_columns(cw003, accumulate_train_stats(cw003, lookup, chunk_size=64))
    assembled = assemble_partitions(cw003, lookup, decision, chunk_size=64)

    frame = assembled.features("validation", list(decision.kept))
    before = bundle.score_batch(frame)
    after = load_bundle(trained["result"].bundle_path).score_batch(frame)

    assert np.array_equal(before, after)
    assert bundle.frozen_thresholds == load_bundle(trained["result"].bundle_path).frozen_thresholds
    assert bundle.best_iteration == trained["result"].report["candidate_runs"]["B"]["best_iteration"]
    assert bundle.scale_pos_weight == pytest.approx(
        trained["result"].report["candidate_runs"]["B"]["scale_pos_weight"]
    )


def test_bundle_scores_are_bounded_scores_not_probabilities_claims(trained: dict) -> None:
    bundle = load_bundle(trained["result"].bundle_path)
    config = trained["config"]
    cw003 = config.modeling
    lookup = load_partition_lookup(cw003)
    decision = decide_columns(cw003, accumulate_train_stats(cw003, lookup, chunk_size=64))
    assembled = assemble_partitions(cw003, lookup, decision, chunk_size=64)
    scores = bundle.score_batch(assembled.features("validation", list(decision.kept)))

    assert scores.ndim == 1
    assert np.isfinite(scores).all()
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    assert "不是已校准" in " ".join(bundle.notes)


def test_bundle_rejects_missing_and_extra_columns(trained: dict) -> None:
    from ai_detector.models.lightgbm_challenger import LightGbmSpecError

    bundle = load_bundle(trained["result"].bundle_path)
    config = trained["config"]
    cw003 = config.modeling
    lookup = load_partition_lookup(cw003)
    decision = decide_columns(cw003, accumulate_train_stats(cw003, lookup, chunk_size=64))
    assembled = assemble_partitions(cw003, lookup, decision, chunk_size=64)
    frame = assembled.features("validation", list(decision.kept))

    with pytest.raises(LightGbmSpecError, match="缺少"):
        bundle.score_batch(frame.drop(columns=[frame.columns[0]]))
    with pytest.raises(LightGbmSpecError, match="多出"):
        bundle.score_batch(frame.assign(extra_probe=1.0))
    with pytest.raises(LightGbmSpecError, match="只接受带列名的 DataFrame"):
        bundle.score_batch(frame.to_numpy())


def test_bundle_contains_no_raw_data(trained: dict) -> None:
    bundle = load_bundle(trained["result"].bundle_path)
    assert bundle.describe()["contains_raw_data"] is False

    for name, value in vars(bundle).items():
        assert not isinstance(value, pd.DataFrame), f"bundle 含有数据框字段：{name}"
        if isinstance(value, np.ndarray):
            assert value.size <= 1000, f"bundle 的 {name} 尺寸像逐行数据：{value.size}"

    # 序列化体积必须远离"原始数据"量级。
    assert trained["result"].bundle_path.stat().st_size < 20 * 1024 * 1024


def test_bundle_records_all_source_fingerprints(trained: dict) -> None:
    bundle = load_bundle(trained["result"].bundle_path)
    fingerprints = bundle.fingerprints
    assert fingerprints["cw004_config_sha256"] == trained["config"].config_sha256
    assert fingerprints["cw003_config_sha256"] == trained["config"].modeling.config_sha256
    assert len(fingerprints["train_transaction_sha256"]) == 64
    assert len(fingerprints["train_identity_sha256"]) == 64
    assert fingerprints["split_algorithm_version"] == "temporal_tie_safe_v1"


def test_bundle_payload_type_is_enforced(trained: dict, tmp_path: Path) -> None:
    from ai_detector.models.lightgbm_challenger import LightGbmSpecError

    path = tmp_path / "not_a_bundle.joblib"
    joblib.dump([1, 2, 3], path)
    with pytest.raises(LightGbmSpecError, match="bundle 类型不符"):
        load_bundle(path)


def test_bundle_class_is_the_same_object_across_import_paths() -> None:
    from ai_detector.models import lightgbm_challenger

    assert LightGbmBundle is lightgbm_challenger.LightGbmBundle


# --- 产物与安全 ---------------------------------------------------------


def test_all_artifacts_exist(trained: dict) -> None:
    result = trained["result"]
    for path in (
        result.report_json,
        result.report_markdown,
        result.threshold_curve_csv,
        result.comparison_csv,
        result.feature_importance_csv,
        result.bundle_path,
    ):
        assert path.is_file(), path
        assert path.stat().st_size > 0, path


def test_report_contains_no_absolute_paths_or_row_level_values(trained: dict) -> None:
    result = trained["result"]
    blobs = {
        "report.json": json.dumps(result.report, ensure_ascii=False),
        "report.md": result.report_markdown.read_text(encoding="utf-8"),
        "threshold_curve.csv": result.threshold_curve_csv.read_text(encoding="utf-8"),
        "comparison.csv": result.comparison_csv.read_text(encoding="utf-8"),
        "feature_importance.csv": result.feature_importance_csv.read_text(encoding="utf-8"),
    }
    for label, blob in blobs.items():
        assert "C:\\" not in blob, label
        assert "D:\\" not in blob, label
        assert "/Users/" not in blob, label
        assert "/home/" not in blob, label
        assert "NEVER_SEEN_CATEGORY" not in blob, label
        for credential in ("password", "token", "cookie", "api_key", "secret"):
            assert credential not in blob.lower(), f"{label} 含疑似凭据字段 {credential}"

    report_blob = blobs["report.json"]
    for row_level_key in ('"predictions"', '"per_row"', '"samples"', '"records"', '"values":'):
        assert row_level_key not in report_blob, f"报告出现行级字段 {row_level_key}"


def test_report_declares_workspace_neutral_paths(trained: dict) -> None:
    """夹具位于临时目录：报告里的路径只能退化为文件名，不得是绝对路径。"""
    report = trained["result"].report
    for path in (
        report["config"]["relative_path"],
        report["prerequisite_gates"]["cw003_report"]["relative_path"],
        report["thresholds"]["curve_written_to"],
        report["feature_importance"]["full_table_written_to"],
        report["inference_bundle"]["relative_path"],
    ):
        assert not Path(path).is_absolute(), path
        assert ":" not in path, path


#: 报告里天然易变的字段：时间戳、各阶段耗时与峰值内存。它们不是结论，
#: 因此比较两次运行时整棵子树一并剥离；**任何其它字段的差异都会让断言失败**。
VOLATILE_REPORT_KEYS = frozenset(
    {
        "generated_at",
        "source_report_generated_at",
        "total_seconds",
        "timings_seconds",
        "fit_seconds",
        "train_seconds",
        "predict_seconds",
        "peak_memory_mib",
        "resources",
    }
)


def _strip_volatile(payload):
    """递归删除易变字段，返回删除计数（确保确实剥到了东西，而不是静默空转）。"""
    removed = 0
    if isinstance(payload, dict):
        for key in [key for key in payload if key in VOLATILE_REPORT_KEYS]:
            payload.pop(key)
            removed += 1
        for value in payload.values():
            removed += _strip_volatile(value)
    elif isinstance(payload, list):
        for item in payload:
            removed += _strip_volatile(item)
    return removed


def test_training_is_deterministic_across_runs(tmp_path: Path) -> None:
    """两次完整运行（含 A/B 与 test 评估）除生成时间与耗时外必须逐字段一致。"""
    from tests.cw004_fixtures import Cw004FixtureBuilder

    workspace = tmp_path / "repeat"
    _, config, _ = Cw004FixtureBuilder(workspace).build_cw004()
    first = run_lightgbm_training(config, chunk_size=64).report
    second = run_lightgbm_training(config, chunk_size=64).report

    assert _strip_volatile(first) > 0
    assert _strip_volatile(second) > 0

    # 剥离的必须只是易变字段：结论性字段仍然在报告里。
    remaining = json.dumps(first, ensure_ascii=False)
    for conclusion in ("best_iteration", "aggregated_gain", "selected_f1_threshold", "roc_auc"):
        assert conclusion in remaining, conclusion

    assert json.dumps(first, sort_keys=True, ensure_ascii=False, default=str) == json.dumps(
        second, sort_keys=True, ensure_ascii=False, default=str
    )


def test_chunk_size_does_not_change_conclusions(tmp_path: Path) -> None:
    """分块大小只影响内存与耗时：换一个分块重跑，冻结阈值与 test 指标必须不变。"""
    from tests.cw004_fixtures import Cw004FixtureBuilder

    workspace = tmp_path / "chunked"
    _, config, _ = Cw004FixtureBuilder(workspace).build_cw004()
    baseline = run_lightgbm_training(config, chunk_size=64).report
    chunked = run_lightgbm_training(config, chunk_size=17).report

    assert (
        baseline["thresholds"]["frozen"]["selected_f1_threshold"]
        == chunked["thresholds"]["frozen"]["selected_f1_threshold"]
    )
    assert baseline["test_evaluation"]["metrics"] == chunked["test_evaluation"]["metrics"]
    assert baseline["features"]["contract"] == chunked["features"]["contract"]


def test_artifacts_land_only_under_configured_outputs(trained: dict) -> None:
    config = trained["config"]
    result = trained["result"]
    expected = {
        config.outputs["report_json"],
        config.outputs["report_markdown"],
        config.outputs["threshold_curve_csv"],
        config.outputs["comparison_csv"],
        config.outputs["feature_importance_csv"],
        config.outputs["bundle"],
    }
    assert {
        result.report_json,
        result.report_markdown,
        result.threshold_curve_csv,
        result.comparison_csv,
        result.feature_importance_csv,
        result.bundle_path,
    } == expected
