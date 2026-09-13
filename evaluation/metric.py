"""
Scout-certified evaluation metric interface for playground-series-s6e9.
Scikit-learn compatible signature: score(y_true, y_pred) -> float
"""

import numpy as np
from evaluation.metrics import MetricRegistry


def score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes competition official metric (ROC-AUC) with strict edge-case protections."""
    return MetricRegistry.roc_auc(y_true, y_pred)
