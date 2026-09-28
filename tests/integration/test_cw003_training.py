"""CW-003 端到端集成测试。

在**项目自建的合成夹具**上跑完整实验（门禁 → 复用 CW-002 切分 → train-only 特征决策 →
三类基线 → validation 冻结阈值 → test 一次性评估 → 四类报告 + 三个模型产物）。

重点守住三条纪律：

1. 分区严格复用 CW-002 清单，左连接不丢不增交易。
2. 阈值只在 validation 上选择；test 不参与任何选择，且只被访问一次。
3. 报告不含绝对路径、凭据、逐行取值或禁止字段；重复运行结果一致。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ai_detector.data.modeling import (
    HoldoutAccessTracker,
    HoldoutLeakageError,
    assemble_partitions,
    load_partition_lookup,
    validate_test_partition,
)
from ai_detector.features.ieee_cis import accumulate_train_stats, decide_columns
from ai_detector.ieee_training import (
    RULE_KEY,
    evaluate_frozen_thresholds,
    freeze_thresholds,
    run_training,
)

BASELINE_KEYS = (RULE_KEY, "dummy", "logistic_regression")
REQUIRED_REPORT_KEYS = {
    "work_package",
    "generated_at",
    "seed",
    "dataset",
    "config",
    "data",
    "partitions",
    "features",
    "baselines",
    "risk_register",
    "cost_note",
    "evaluation",
    "timings_seconds",
    "limitations",
    "environment",
}


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """跑一次完整实验，供本模块多个测试复用。"""
    from tests.cw003_fixtures import Cw003FixtureBuilder

    workspace = tmp_path_factory.mktemp("cw003")
    config_path, config = Cw003FixtureBuilder(workspace).build()
    result = run_training(config, chunk_size=64)
    return {"result": result, "config": config, "workspace": workspace}


# --- 门禁与分区 ---------------------------------------------------------


def test_gate_evidence_matches_report(trained: dict) -> None:
    report = trained["result"].report
    fingerprints = report["data"]["fingerprints"]

    assert set(fingerprints) == {"train_transaction.csv", "train_identity.csv"}
    for info in fingerprints.values():
        assert info["match"] is True, "报告记录的门禁指纹必须与 CW-002 一致"
    assert report["data"]["split_algorithm_version"] == "temporal_tie_safe_v1"


def test_partitions_reuse_cw002_manifest(trained: dict) -> None:
    """分区必须与 CW-002 清单逐一对应：不重、不漏、ID 互斥。"""
    config = trained["config"]
    report = trained["result"].report
    manifest = pd.read_csv(config.paths["split_manifest_rows"])

    expected_sizes = manifest["partition"].value_counts().to_dict()
    for name in ("train", "validation", "test"):
        assert report["partitions"][name]["rows"] == expected_sizes[name]

    assert report["partitions"]["invariants"] == {"ids_disjoint": True, "ids_complete": True}
    assert (
        report["partitions"]["train"]["rows"]
        + report["partitions"]["validation"]["rows"]
        + report["partitions"]["test"]["rows"]
        == config.expected_total_rows
    )


def test_left_join_neither_drops_nor_duplicates_transactions(trained: dict) -> None:
    """identity 只覆盖 60% 交易：左连接必须保行，且 identity_present 取值正确。

    **三个分区**都做取值级逐行校验。test 特征只能经受控入口读取，因此本测试创建一个
    **独立的** tracker 并显式 ``freeze()``——这正是"要读 test，就先声明自己处在冻结后的
    评估阶段"的用法：读取由 ``features()`` 自己记账，测试只取连接键与派生列，
    不触碰任何模型特征值，也不访问内部存储。
    """
    config = trained["config"]
    lookup = load_partition_lookup(config)
    decision = decide_columns(config, accumulate_train_stats(config, lookup, chunk_size=64))
    assembled = assemble_partitions(config, lookup, decision, chunk_size=64)

    partitions = ("train", "validation", "test")
    total = sum(assembled.structural_row_count(name) for name in partitions)
    assert total == config.expected_total_rows

    id_sets = [assembled.structural_ids(name) for name in partitions]
    assert sum(len(values) for values in id_sets) == config.expected_total_rows
    for left in range(len(id_sets)):
        for right in range(left + 1, len(id_sets)):
            assert not (id_sets[left] & id_sets[right])

    # --- test 分区：测试内独立 tracker，先冻结，再经受控 features() 读取 ---
    test_tracker = HoldoutAccessTracker()
    test_tracker.freeze(
        reason="CW-003 集成测试：核对左连接不变量，属冻结后的结构性取值校验"
    )
    probe_columns = [config.join_key, "identity_present"]
    test_frame = assembled.features("test", probe_columns, tracker=test_tracker)
    assert [entry["event"] for entry in test_tracker.entries] == [
        "thresholds_frozen",
        "test_access",
    ]
    assert test_tracker.entries[1]["what"] == "features"
    assert test_tracker.entries[1]["purpose"] == "single_shot_evaluation"
    assert test_tracker.entries[1]["after_freeze"] is True
    assert list(test_frame.columns) == probe_columns

    identity_ids = set(pd.read_csv(config.paths["identity"], usecols=[config.join_key])[config.join_key])
    combined = pd.concat(
        [
            assembled[name].loc[:, probe_columns]
            for name in ("train", "validation")
        ]
        + [test_frame],
        ignore_index=True,
    )
    assert len(combined) == config.expected_total_rows
    expected_present = combined[config.join_key].isin(identity_ids).astype(int)
    assert combined["identity_present"].astype(int).equals(expected_present)



# --- 特征契约 -----------------------------------------------------------


def test_forbidden_features_never_reach_the_model(trained: dict) -> None:
    report = trained["result"].report
    kept = report["baselines"]["logistic_regression"]["params"]
    assert kept  # 参数摘要存在

    manifest = json.loads(
        trained["result"].feature_manifest_json.read_text(encoding="utf-8")
    )
    all_kept = set(manifest["kept"]["numeric"]) | set(manifest["kept"]["categorical"])
    assert not all_kept & {"isFraud", "TransactionID", "TransactionDT"}
    assert set(manifest["forbidden_features"]) == {"isFraud", "TransactionID", "TransactionDT"}
    # 允许清单里也不允许出现它们。
    allowlist = {name for names in manifest["allowlist"].values() for name in names}
    assert not allowlist & {"isFraud", "TransactionID", "TransactionDT"}


def test_feature_manifest_records_decisions_and_reasons(trained: dict) -> None:
    manifest = json.loads(trained["result"].feature_manifest_json.read_text(encoding="utf-8"))

    assert manifest["output_format"] == "sparse_csr", "设计矩阵必须保持稀疏"
    assert manifest["transformed_dimension"] > len(manifest["kept"]["numeric"])
    assert manifest["excluded"], "夹具中应有被排除的列"
    for item in manifest["excluded"]:
        assert item["reason_code"] and item["reason"]
        assert "train_missing_rate" in item
    assert manifest["derived_features"]["identity_present"]["enabled"] is True
    assert manifest["derived_features"]["identity_present"]["risk_note"]


# --- 决策只来自 train ---------------------------------------------------


def test_thresholds_are_selected_on_validation_only(trained: dict) -> None:
    """报告中的冻结阈值必须等于"仅用 validation 数据重算"的结果。"""
    config = trained["config"]
    report = trained["result"].report
    lookup = load_partition_lookup(config)
    decision = decide_columns(config, accumulate_train_stats(config, lookup, chunk_size=64))
    assembled = assemble_partitions(config, lookup, decision, chunk_size=64)
    tracker = HoldoutAccessTracker()

    from ai_detector.models.rule_baseline import fit_rule_baseline

    rule_state = fit_rule_baseline(config.rule_baseline, assembled["train"])
    validation_labels = assembled.target("validation", tracker=tracker)

    recomputed = freeze_thresholds(
        config, validation_labels, rule_state.score(assembled["validation"]), model_key=RULE_KEY
    )
    reported = report["baselines"][RULE_KEY]["thresholds"]
    assert recomputed["selected_f1_threshold"] == reported["selected_f1_threshold"]
    for name, payload in recomputed["cost_scenarios"].items():
        assert payload["selected_threshold"] == reported["cost_scenarios"][name]["selected_threshold"]


def test_frozen_thresholds_do_not_depend_on_test_data() -> None:
    """同一份 validation 与两份不同的 test：冻结阈值必须完全相同，只有指标不同。"""
    from ai_detector.data.modeling import ModelingConfig
    from ai_detector.config import PROJECT_ROOT

    config = ModelingConfig.from_yaml(PROJECT_ROOT / "configs" / "ieee_cis_baseline.yaml")
    rng = np.random.default_rng(0)
    validation_labels = np.array([0] * 90 + [1] * 10)
    validation_scores = np.concatenate([rng.random(90) * 0.6, 0.4 + rng.random(10) * 0.6])
    frozen = freeze_thresholds(
        config, validation_labels, validation_scores, model_key="probe"
    )

    test_scores = np.concatenate([rng.random(80) * 0.7, 0.3 + rng.random(20) * 0.7])
    first = evaluate_frozen_thresholds(
        config, np.array([0] * 80 + [1] * 20), test_scores, frozen, model_key="probe"
    )
    second = evaluate_frozen_thresholds(
        config, np.array([0] * 95 + [1] * 5), test_scores, frozen, model_key="probe"
    )

    assert first["threshold"] == second["threshold"] == frozen["selected_f1_threshold"]
    for name in first["cost_scenarios"]:
        assert (
            first["cost_scenarios"][name]["frozen_threshold"]
            == second["cost_scenarios"][name]["frozen_threshold"]
        )
    # 指标确实随 test 标签改变，说明它们用的是同一批冻结阈值而非重新选择。
    assert first["metrics"] != second["metrics"]


def test_test_access_log_is_produced_by_real_phases(trained: dict) -> None:
    """访问日志由追踪器在真实执行阶段产生，且冻结事件排在所有访问之前。"""
    evaluation = trained["result"].report["evaluation"]
    access = evaluation["test_access"]

    assert evaluation["single_shot_test"] is True
    assert evaluation["evaluation_partition"] == "test"
    assert evaluation["validation_used_for"] == [
        "threshold_curve",
        "f1_threshold",
        "cost_thresholds",
    ]
    assert evaluation["test_used_for"] == ["single_frozen_evaluation"]

    assert access["frozen"] is True
    assert access["frozen_reason"], "冻结理由必须记录"
    assert access["n_test_accesses"] >= 1

    events = access["entries"]
    assert events[0]["event"] == "thresholds_frozen", "冻结事件必须是第一条"
    for entry in events[1:]:
        assert entry["event"] == "test_access"
        assert entry["after_freeze"] is True
        assert entry["sequence"] > events[0]["sequence"]
    assert any(e["what"] == "labels" for e in events[1:]), "必须记录 test 标签访问"
    assert all(e["purpose"] == "single_shot_evaluation" for e in events[1:])


def test_pre_freeze_stage_reads_no_test_labels(trained: dict) -> None:
    """冻结阶段只做结构性检查：不读 test 标签，标签率也只对 train/validation 计算。"""
    report = trained["result"].report
    pre_freeze = report["evaluation"]["pre_freeze_structural_only"]

    assert pre_freeze["test_labels_read_before_freeze"] is False
    assert set(pre_freeze["label_rates_computed_for"]) == {"train", "validation"}
    assert report["partitions"]["test"]["label_rate_computed_after_freeze"] is True


def test_test_labels_cannot_be_read_before_freezing(trained: dict) -> None:
    """生命周期测试：冻结前读取 test 标签必须失败。"""
    config = trained["config"]
    lookup = load_partition_lookup(config)
    decision = decide_columns(config, accumulate_train_stats(config, lookup, chunk_size=64))
    assembled = assemble_partitions(config, lookup, decision, chunk_size=64)
    tracker = HoldoutAccessTracker()

    # 结构性访问（连接键集合）在冻结前是允许的。
    assert assembled.structural_ids("test")
    with pytest.raises(HoldoutLeakageError, match="在阈值冻结之前尝试访问 test"):
        assembled.target("test", tracker=tracker)
    with pytest.raises(HoldoutLeakageError, match="在阈值冻结之前尝试访问 test"):
        validate_test_partition(config, assembled, tracker)

    tracker.freeze(reason="单元测试：手动冻结")
    labels = assembled.target("test", tracker=tracker)
    assert labels.size == config.expected_split_sizes["test"]
    assert len(tracker.entries) == 2  # 冻结事件 + 一次标签访问


def test_test_partition_has_no_target_column_after_assembly(trained: dict) -> None:
    """装配阶段即剥离 test 目标列：冻结前代码路径物理上拿不到 test 标签。"""
    config = trained["config"]
    lookup = load_partition_lookup(config)
    decision = decide_columns(config, accumulate_train_stats(config, lookup, chunk_size=64))
    assembled = assemble_partitions(config, lookup, decision, chunk_size=64)

    # test 分区不再允许整体取出：列名走结构性入口（只暴露 schema，不暴露取值）。
    assert config.target not in assembled.structural_columns("test")
    assert config.target in assembled["train"].columns
    assert config.target in assembled["validation"].columns


def test_rule_baseline_quantiles_come_from_train(trained: dict) -> None:
    """规则阈值必须等于训练分区上的分位数。"""
    config = trained["config"]
    report = trained["result"].report
    lookup = load_partition_lookup(config)
    decision = decide_columns(config, accumulate_train_stats(config, lookup, chunk_size=64))
    assembled = assemble_partitions(config, lookup, decision, chunk_size=64)

    params = report["baselines"][RULE_KEY]["params"]
    for field in config.rule_fields():
        expected = float(
            np.quantile(assembled["train"][field].to_numpy(dtype=float), params["quantile"])
        )
        assert params["thresholds"][field] == pytest.approx(expected)


# --- 三类基线 -----------------------------------------------------------


def test_all_three_baselines_ran_with_full_evidence(trained: dict) -> None:
    report = trained["result"].report

    assert set(report["baselines"]) == set(BASELINE_KEYS)
    for key in BASELINE_KEYS:
        entry = report["baselines"][key]
        assert entry["params"], key
        assert entry["fit_seconds"] >= 0
        assert entry["predict_seconds"] >= 0
        assert "convergence" in entry
        thresholds = entry["thresholds"]
        assert 0.0 < thresholds["selected_f1_threshold"] < 1.0
        assert set(thresholds["cost_scenarios"]) == {"fp_fn_1_1", "fp_fn_1_5", "fp_fn_1_20"}
        test = entry["test"]
        assert test["partition"] == "test"
        assert test["frozen_from"] == "validation"
        metrics = test["metrics"]
        for name in ("precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate"):
            assert 0.0 <= metrics[name] <= 1.0, f"{key}.{name}"
        assert metrics["n_samples"] == report["partitions"]["test"]["rows"]


def test_dummy_baseline_is_the_prior_reference(trained: dict) -> None:
    """Dummy 用先验概率：ROC-AUC 应为 0.5，PR-AUC 应等于测试集正例率。"""
    report = trained["result"].report
    metrics = report["baselines"]["dummy"]["test"]["metrics"]

    assert metrics["roc_auc"] == pytest.approx(0.5)
    assert metrics["pr_auc"] == pytest.approx(report["partitions"]["test"]["label_rate"])


def test_logistic_regression_reports_convergence(trained: dict) -> None:
    """收敛状态必须被明确记录，不得留空（那等于把未收敛藏起来）。"""
    convergence = trained["result"].report["baselines"]["logistic_regression"]["convergence"]

    assert convergence["converged"] is not None
    assert convergence["n_iter"], "必须记录迭代次数"
    assert isinstance(convergence["warnings"], list)


def test_cost_scenarios_are_relative_units(trained: dict) -> None:
    report = trained["result"].report

    assert report["cost_note"]["units"] == "研究用相对成本单位"
    for key in BASELINE_KEYS:
        scenarios = report["baselines"][key]["test"]["cost_scenarios"]
        assert set(scenarios) == {"fp_fn_1_1", "fp_fn_1_5", "fp_fn_1_20"}
        for payload in scenarios.values():
            assert payload["test_total_cost"] >= 0
            assert payload["test_average_cost"] is not None
            assert set(payload["test_confusion_matrix"]) == {
                "true_positive",
                "false_positive",
                "true_negative",
                "false_negative",
            }
            matrix = payload["test_confusion_matrix"]
            assert sum(matrix.values()) == report["partitions"]["test"]["rows"]


def test_higher_fn_cost_never_raises_the_threshold(trained: dict) -> None:
    """FN 成本越高，选出的阈值不应上升（更倾向多报警）。"""
    report = trained["result"].report
    for key in ("logistic_regression",):
        scenarios = report["baselines"][key]["thresholds"]["cost_scenarios"]
        ordered = [
            scenarios["fp_fn_1_1"]["selected_threshold"],
            scenarios["fp_fn_1_5"]["selected_threshold"],
            scenarios["fp_fn_1_20"]["selected_threshold"],
        ]
        assert ordered == sorted(ordered, reverse=True), f"{key} 的阈值随 FN 成本单调不增"


# --- 报告安全边界 -------------------------------------------------------


def test_report_contains_no_absolute_paths_or_row_level_values(trained: dict) -> None:
    result = trained["result"]
    report_blob = json.dumps(result.report, ensure_ascii=False)
    manifest_blob = result.feature_manifest_json.read_text(encoding="utf-8")
    markdown = result.report_markdown.read_text(encoding="utf-8")
    curve = result.threshold_curve_csv.read_text(encoding="utf-8")

    for blob, label in (
        (report_blob, "report.json"),
        (manifest_blob, "feature_manifest.json"),
        (markdown, "report.md"),
        (curve, "threshold_curve.csv"),
    ):
        assert "C:\\" not in blob, label
        assert "D:\\" not in blob, label
        assert "/Users/" not in blob, label
        assert "/home/" not in blob, label
        for credential in ("password", "token", "cookie", "api_key", "secret"):
            assert credential not in blob.lower(), f"{label} 含疑似凭据字段 {credential}"

    # 报告不得逐行输出样本或预测：不允许出现行级结构键。
    for row_level_key in ('"predictions"', '"per_row"', '"samples"', '"records"', '"values":'):
        assert row_level_key not in report_blob, f"报告出现行级字段 {row_level_key}"


def test_reports_do_not_leak_raw_category_values(trained: dict) -> None:
    """类别列的真实取值不得出现在报告中（夹具里的类别值可作探针）。"""
    manifest_blob = trained["result"].feature_manifest_json.read_text(encoding="utf-8")
    report_blob = json.dumps(trained["result"].report, ensure_ascii=False)

    for probe in ("NEVER_SEEN_CATEGORY",):
        assert probe not in manifest_blob
        assert probe not in report_blob


def test_threshold_curve_covers_all_baselines_on_validation(trained: dict) -> None:
    curve = pd.read_csv(trained["result"].threshold_curve_csv)

    assert set(curve["model_key"]) == set(BASELINE_KEYS)
    assert set(curve["partition"]) == {"validation"}
    assert len(curve) == 3 * len(trained["config"].threshold_grid())
    for column in ("threshold", "true_positive", "false_positive", "true_negative",
                   "false_negative", "precision", "recall", "f1", "false_positive_rate",
                   "predicted_positive_rate"):
        assert column in curve.columns


# --- 产物与确定性 -------------------------------------------------------


def test_all_artifacts_exist(trained: dict) -> None:
    result = trained["result"]

    for path in (
        result.report_json,
        result.report_markdown,
        result.threshold_curve_csv,
        result.feature_manifest_json,
    ):
        assert path.is_file(), path
    assert set(result.model_paths) == set(BASELINE_KEYS)
    for path in result.model_paths.values():
        assert path.is_file(), path


def test_saved_pipelines_reload_and_predict(trained: dict) -> None:
    """模型产物必须可加载并复用训练期统计量。

    test 特征只能经已冻结的追踪器读取：这里确实要做一次真实的 test 打分，因此显式冻结并
    由 ``features()`` 自动记账（恰好一条 ``features`` 访问）。
    """
    import joblib

    config = trained["config"]
    lookup = load_partition_lookup(config)
    decision = decide_columns(config, accumulate_train_stats(config, lookup, chunk_size=64))
    assembled = assemble_partitions(config, lookup, decision, chunk_size=64)

    tracker = HoldoutAccessTracker()
    tracker.freeze(reason="集成测试：对已保存的 Pipeline 做一次 test 打分，属真实的冻结后读取")
    features = assembled.features("test", list(decision.kept), tracker=tracker)
    assert [
        entry["what"] for entry in tracker.entries if entry["event"] == "test_access"
    ] == ["features"]

    for key in ("dummy", "logistic_regression"):
        pipeline = joblib.load(trained["result"].model_paths[key])
        probabilities = pipeline.predict_proba(features)[:, 1]
        assert probabilities.shape == (len(features),), key
        assert np.isfinite(probabilities).all(), key
        assert ((probabilities >= 0) & (probabilities <= 1)).all(), key


#: 允许在不同运行之间变化的字段：生成时间与各类耗时。
VOLATILE_KEYS = ("generated_at", "timings_seconds", "total_seconds", "fit_seconds", "predict_seconds")


def _strip_volatile(report: dict) -> dict:
    """去掉时间类字段，只保留可复现的结构性结论。

    耗时本来就应当变化；把它从比较中排除，而不是放宽断言强度——
    特征清单、阈值与混淆矩阵仍然逐字段比较。
    """
    payload = json.loads(json.dumps(report, ensure_ascii=False))
    for key in ("generated_at", "timings_seconds", "total_seconds"):
        payload.pop(key, None)
    for entry in payload["baselines"].values():
        for key in ("fit_seconds", "predict_seconds"):
            entry.pop(key, None)
    return payload


def test_training_is_deterministic(trained: dict) -> None:
    """**同一份配置**重复运行：特征清单、阈值、混淆矩阵必须逐字段一致。"""
    config = trained["config"]
    first = run_training(config, chunk_size=64).report
    second = run_training(config, chunk_size=64).report

    assert _strip_volatile(first) == _strip_volatile(second)

    reference = trained["result"].report
    for key in BASELINE_KEYS:
        assert (
            first["baselines"][key]["test"]["metrics"]["f1"]
            == pytest.approx(reference["baselines"][key]["test"]["metrics"]["f1"])
        )


def test_chunk_size_does_not_change_conclusions(trained: dict) -> None:
    """分块大小只影响内存与耗时，不得改变任何结论。"""
    config = trained["config"]
    default = run_training(config, chunk_size=64).report
    tiny = run_training(config, chunk_size=17).report

    assert _strip_volatile(default) == _strip_volatile(tiny)


def test_report_states_no_effect_threshold_and_no_production_claim(trained: dict) -> None:
    """不得出现效果门槛或生产效果声明。"""
    markdown = trained["result"].report_markdown.read_text(encoding="utf-8")

    assert "不是" in markdown and "生产" in markdown
    assert "报告未设置任何「效果必须达到某值」的门槛" in markdown, "必须显式声明不设效果门槛"
    for overclaim in ("生产级", "银行级", "保证达到", "达标"):
        assert overclaim not in markdown
    # 不得出现肯定式的效果门槛措辞。
    for threshold_claim in ("PR-AUC 必须", "F1 必须", "AUC 必须", "指标必须"):
        assert threshold_claim not in markdown
    assert "研究用相对成本单位" in markdown
