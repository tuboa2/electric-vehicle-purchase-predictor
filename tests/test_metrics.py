import numpy as np
import pytest
from evaluation.metrics import MetricRegistry
from evaluation.metric import score as metric_score
from schemas.metric_spec import score as spec_score, get_scorer


def test_roc_auc_standard():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    res = MetricRegistry.roc_auc(y_true, y_pred)
    assert 0.0 <= res <= 1.0
    assert pytest.approx(res, 0.001) == 0.75


def test_roc_auc_2d_input():
    y_true = np.array([0, 1, 0, 1])
    # Shape (N, 2) from model.predict_proba
    y_pred_2d = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.7, 0.3],
        [0.1, 0.9],
    ])
    score_2d = MetricRegistry.roc_auc(y_true, y_pred_2d)
    assert score_2d == 1.0

    # Shape (N, 1)
    y_pred_col = np.array([[0.1], [0.8], [0.3], [0.9]])
    score_col = MetricRegistry.roc_auc(y_true, y_pred_col)
    assert score_col == 1.0


def test_roc_auc_nan_and_inf_handling():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([np.nan, -np.inf, np.inf, 0.8])
    # Should not crash with ValueError or NaN
    res = MetricRegistry.roc_auc(y_true, y_pred)
    assert not np.isnan(res)
    assert 0.0 <= res <= 1.0


def test_roc_auc_string_targets():
    y_true_str = np.array(["No", "No", "Yes", "Yes"])
    y_pred = np.array([0.1, 0.2, 0.8, 0.9])
    res = MetricRegistry.roc_auc(y_true_str, y_pred)
    assert res == 1.0


def test_roc_auc_single_class_edge_case():
    # Only class 0 present
    y_true_zeros = np.array([0, 0, 0, 0])
    y_pred = np.array([0.2, 0.4, 0.6, 0.8])
    res = MetricRegistry.roc_auc(y_true_zeros, y_pred)
    assert res == 0.5


def test_metric_spec_and_metric_interface():
    y_true = np.array([0, 1])
    y_pred = np.array([0.2, 0.7])
    assert metric_score(y_true, y_pred) == 1.0
    assert spec_score(y_true, y_pred) == 1.0

    scorer_fn, direction = get_scorer()
    assert direction == "maximize"
    assert scorer_fn(y_true, y_pred) == 1.0


def test_registry_lookup():
    fn, direction = MetricRegistry.get_metric("roc_auc")
    assert direction == "maximize"
    assert callable(fn)
