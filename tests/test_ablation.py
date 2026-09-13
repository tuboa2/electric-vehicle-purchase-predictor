import numpy as np
import pytest
from evaluation.ablation import AblationHarness
from evaluation.blender import EnsembleBlender


def test_ensemble_blender_hill_climbing():
    np.random.seed(42)
    n = 200
    y_true = np.random.choice([0, 1], n)

    # Model 1: Moderate accuracy
    pred1 = np.clip(0.6 * y_true + 0.4 * np.random.uniform(0, 1, n), 0, 1)
    # Model 2: Slightly different errors
    pred2 = np.clip(0.5 * y_true + 0.5 * np.random.uniform(0, 1, n), 0, 1)
    # Model 3: Random noise
    pred3 = np.random.uniform(0, 1, n)

    oof_dict = {"m1": pred1, "m2": pred2, "m3": pred3}

    weights, best_score = EnsembleBlender.hill_climbing(
        oof_dict=oof_dict,
        y_true=y_true,
        metric_name="roc_auc",
        n_iterations=20,
    )

    assert len(weights) > 0
    assert pytest.approx(sum(weights.values()), rel=1e-3) == 1.0
    # Model 1 and/or Model 2 should receive positive weight, Model 3 (noise) should have low or zero weight
    assert "m1" in weights
    assert weights.get("m3", 0.0) < 0.2


def test_blender_gate_5_evaluation():
    # Pass scenario: gain >= 0.0005
    passed, gain = EnsembleBlender.evaluate_gate_5(best_single_cv=0.8500, blended_cv=0.8520, metric_name="roc_auc")
    assert passed is True
    assert pytest.approx(gain) == 0.0020

    # Fail scenario: gain < 0.0005
    passed_fail, gain_fail = EnsembleBlender.evaluate_gate_5(best_single_cv=0.8500, blended_cv=0.8502, metric_name="roc_auc")
    assert passed_fail is False
    assert pytest.approx(gain_fail) == 0.0002


def test_ablation_harness():
    harness = AblationHarness(metric_name="roc_auc")

    # Mock evaluate_fn: score depends heavily on 'core_features'
    def mock_eval(features: list[str]) -> float:
        score = 0.50
        if "c1" in features and "c2" in features:
            score += 0.30
        if "noise_1" in features:
            score -= 0.02 # harmful
        return score

    feature_blocks = {
        "core_features": ["c1", "c2"],
        "noise_features": ["noise_1"],
    }
    all_features = ["c1", "c2", "noise_1"]

    res = harness.run_feature_ablation(mock_eval, feature_blocks, all_features)

    assert res["baseline_score"] == 0.78
    results_map = {r["block_name"]: r for r in res["ablation_results"]}

    # Dropping core_features should cause massive score drop -> ESSENTIAL
    assert results_map["core_features"]["marginal_contribution"] > 0.1
    assert results_map["core_features"]["status"] == "ESSENTIAL"

    # Dropping noise_features should improve score -> HARMFUL
    assert results_map["noise_features"]["marginal_contribution"] < 0
    assert results_map["noise_features"]["status"] == "HARMFUL"
